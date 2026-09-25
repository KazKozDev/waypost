"""Request classifier.

A cascade of three levels:
  L0 — deterministic features (length, tools, images, code). ~0 µs.
       No neural net belongs here: this is reading request fields.
  L1 — static embeddings (potion-multilingual-128M) + a trained head.
       Optional: enabled by ROUTER_ENABLE_L1, and silently falls back to
       L0 if the package is absent.
  L2 — a heavy encoder only when L1 confidence is low. An extension
       point, see classify_l2().

The routing embedding represents the current task, relevant history and
constraints. It is shared by the classifier, predictor and neighbourhood.
The semantic cache uses a separate full-request embedding.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache

from .schemas import Capability, ChatRequest, RequestProfile, Tier
from .request_view import REPRESENTATION_VERSION, bounded, build_request_view
from .embeddings import HashingEncoder

CODE_RE = re.compile(r"```|\bdef \b|\bclass \b|=>|;\s*$|</\w+>", re.M)
MATH_RE = re.compile(r"[∫∑√±≈≤≥]|\\frac|\$\$|\b\d+\s*[\*/\^]\s*\d+")
CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")

# Code does not always arrive in blocks: "write a python function" is
# also code. The hint words catch natural-language requests that CODE_RE
# does not see.
CODE_HINTS = (
    "функци",
    "python",
    "код",
    "сортиров",
    "алгоритм",
    "скрипт",
    "программ",
    "переменн",
    "цикл",
    "массив",
    "дебаг",
    "баг",
    "code",
    "function",
    "script",
    "algorithm",
    "sort",
    "implement",
    "debug",
    "bug",
    "program",
    "javascript",
    "sql",
    "regex",
    "класс",
)

REASONING_HINTS = (
    "почему",
    "объясни",
    "докажи",
    "сравни",
    "проанализируй",
    "спроектируй",
    "why",
    "explain",
    "prove",
    "analyze",
    "design",
    "step by step",
)
EXTRACTION_HINTS = (
    "извлеки",
    "классифицируй",
    "переведи",
    "исправь",
    "список",
    "extract",
    "classify",
    "translate",
    "fix",
    "list",
    "summarize",
)


def estimate_tokens(text: str) -> int:
    """A rough estimate. Cyrillic tokenizes denser than Latin — an exact
    count needs the specific provider tokenizer, but this is enough for
    quota reservation."""
    if not text:
        return 0
    cyr = len(CYRILLIC_RE.findall(text))
    ratio = 2.2 if cyr > len(text) * 0.3 else 3.6
    return int(len(text) / ratio) + 1


def _flatten(req: ChatRequest) -> tuple[str, bool]:
    """Request text + a flag for whether images are present."""
    parts, has_image = [], False
    for m in req.messages:
        if isinstance(m.content, str):
            parts.append(m.content)
        elif isinstance(m.content, list):
            for block in m.content:
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif block.get("type") in ("image_url", "image"):
                    has_image = True
    return "\n".join(parts), has_image


ACTION_PATTERNS = (
    ("translation", r"\btranslat\w*\b|переведи|перевести"),
    ("rewriting", r"\brewrite\b|\brephrase\b|\bproofread\b|\bshorten\b|make it shorter|fix the grammar|сократи|перефразируй|исправь (?:грамматику|текст|опечатки)"),
    ("summarization", r"\bsummari[sz]\w*\b|\bsummary\b|резюмируй|кратко изложи|подведи итог"),
    ("extraction", r"\bextract\w*\b|\bclassify\b|извлеки|классифицируй"),
)
DESIGN_RE = re.compile(r"(?i)\bdesign\b|\barchitect\w*\b|спроектируй|архитектур")
HARD_REASONING_RE = re.compile(r"(?i)distributed|consensus|replicat|распределён|распределен|консенсус|репликац|\btheorem\b|теорем")
DEBUG_RE = re.compile(r"(?i)\bdebug\b|\bbug\b|\bfix\b|handle .*input|баг|исправь|почини|ошибк")
GENERATE_RE = re.compile(r"(?i)\bwrite\b|\bimplement\b|\bcreate\b|\badd\b|напиши|создай|реализуй|добавь")


def classify_l0(req: ChatRequest) -> RequestProfile:
    full_text, has_image = _flatten(req)
    # Admission still sees the full payload, including schemas and tool-call
    # arguments. Selecting a smaller routing view must not undercount context.
    tool_payload = json.dumps(
        {"tools": req.tools, "functions": req.functions,
         "response_format": req.response_format,
         "tool_calls": [m.tool_calls for m in req.messages if m.tool_calls]},
        ensure_ascii=False,
    ) if (req.tools or req.functions or req.response_format or any(m.tool_calls for m in req.messages)) else ""
    n_in = estimate_tokens(full_text) + estimate_tokens(tool_payload)
    n_out = req.max_tokens or min(2048, max(256, n_in // 2))
    caps: set[Capability] = set()
    if req.tools or req.functions:
        caps.add(Capability.TOOLS)
    if (req.response_format or {}).get("type") in ("json_object", "json_schema"):
        caps.add(Capability.JSON)
    if has_image:
        caps.add(Capability.VISION)
    if req.stream:
        caps.add(Capability.STREAM)

    view = build_request_view(req)
    latest = bounded(view.latest, 2300)
    text = bounded(view.intent_text, 3200)
    low = text.lower()
    has_code = bool(CODE_RE.search(text)) or any(h in low for h in CODE_HINTS)
    has_code = has_code or bool(re.search(r"(?i)\btests?\b|тест[ыаов]*\b", text))
    action = next((name for name, pattern in ACTION_PATTERNS
                   if re.search(pattern, latest, re.IGNORECASE)), None)
    design = bool(DESIGN_RE.search(latest))
    reasoning = any(h in low for h in REASONING_HINTS)
    has_math = bool(MATH_RE.search(latest))
    # A translation/summarization of code is a text operation. Conversely,
    # an elliptical edit of previously requested code remains code work.
    if action and not (action == "rewriting" and view.continuation and has_code):
        task, subtype = "extraction", action
    elif design:
        task, subtype = "reasoning", "design"
    elif has_code:
        task = "code"
        if DEBUG_RE.search(latest):
            subtype = "code_debugging"
        elif GENERATE_RE.search(latest):
            subtype = "code_generation"
        else:
            subtype = "code_analysis"
    elif reasoning or has_math:
        task, subtype = "reasoning", "analysis"
    else:
        task, subtype = "chat", "conversation"

    confidence = 0.8 if action or has_code or design else 0.6
    if view.continuation:
        confidence = min(confidence, 0.65)
    active_tokens = estimate_tokens(view.latest)
    score = 0.20 + min(0.30, active_tokens / 12_000)
    score += 0.25 if task == "code" else 0.0
    score += 0.15 if task == "reasoning" else 0.0
    score += 0.10 if has_math and task != "extraction" else 0.0
    score += 0.35 if task == "reasoning" and HARD_REASONING_RE.search(text) else 0.0
    score += 0.10 if req.tools or req.functions else 0.0
    score += 0.15 if has_image else 0.0
    score -= 0.10 if task == "extraction" and active_tokens < 800 else 0.0
    score = max(0.0, min(1.0, score))
    return RequestProfile(
        task_probs={task: confidence, "chat": 1.0 - confidence} if task != "chat" else {"chat": 1.0},
        task_subtype=subtype,
        complexity=score, confidence=confidence, tier=tier_from(score, confidence),
        required_caps=caps, est_input_tokens=n_in, est_output_tokens=n_out,
        language=view.language, classifier_source="rules",
        routing_text=view.embedding_text, representation_version=REPRESENTATION_VERSION,
    )


def tier_from(complexity: float, confidence: float) -> Tier:
    """Asymmetry of errors: underestimating complexity costs more than
    overestimating it (escalation = double quota spend). So low confidence
    raises the tier rather than lowering it."""
    adjusted = complexity + (1.0 - confidence) * 0.15
    if adjusted < 0.35:
        return Tier.S
    if adjusted < 0.65:
        return Tier.M
    return Tier.L


# ---------------------------------------------------------------- L1 hook


@lru_cache(maxsize=1)
def _static_encoder():
    """potion-multilingual-128M: static embeddings, no forward pass,
    multilingual support is mandatory for mixed ru/en traffic."""
    try:
        from model2vec import StaticModel

        return StaticModel.from_pretrained("minishlab/potion-multilingual-128M")
    except Exception:
        return None


@lru_cache(maxsize=2)
def routing_encoder(enable_l1: bool = True):
    encoder = _static_encoder() if enable_l1 else None
    if encoder is None:
        encoder = HashingEncoder()
        name = "hashing:256"
    else:
        name = "model2vec:minishlab/potion-multilingual-128M"
    return encoder, f"{REPRESENTATION_VERSION}|{name}"


def classify(
    req: ChatRequest, enable_l1: bool = False, head_path: str | None = None
) -> RequestProfile:
    profile = classify_l0(req)
    encoder, version = routing_encoder(enable_l1)
    profile.embedding = encoder.encode([profile.routing_text])[0]
    profile.embedder_version = version
    if not enable_l1 or isinstance(encoder, HashingEncoder):
        return profile

    head = _load_head(head_path)
    if (head is None or head.representation_version != REPRESENTATION_VERSION
            or head.dim != len(profile.embedding)):
        return profile  # old heads must be retrained on the new request view

    probs, complexity, conf = head.predict(profile.embedding)
    profile.task_probs = probs
    profile.complexity = complexity
    profile.confidence = conf
    profile.tier = tier_from(complexity, conf)
    profile.classifier_source = "L1"
    return profile


@lru_cache(maxsize=4)
def _load_head(path: str | None):
    """The head is trained on your own logs (scripts/train_head.py).
    Before training it does not exist — and that is a normal state for
    the first weeks."""
    if not path:
        return None
    try:
        from .head import TaskHead

        return TaskHead.load(path)
    except Exception:
        return None


def embed_text(text: str):
    """Embedding for the semantic cache, independent of the L1 classifier.
    Returns None if the model2vec package is not installed."""
    encoder = _static_encoder()
    if encoder is None:
        return None
    return encoder.encode([text[:4000]])[0]
