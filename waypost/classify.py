"""Request classifier.

A cascade of three levels:
  L0 — deterministic features (length, tools, images, code). ~0 µs.
       No neural net belongs here: this is reading request fields.
  L1 — static embeddings (potion-multilingual-128M) + a trained head.
       Optional: enabled by ROUTER_ENABLE_L1, and silently falls back to
       L0 if the package is absent.
  L2 — a heavy encoder only when L1 confidence is low. An extension
       point, see classify_l2().

The embedding is computed once and reused by three consumers: the
classifier head, the semantic cache, and the bandit features.
"""
from __future__ import annotations

import re
from functools import lru_cache

from .schemas import Capability, ChatRequest, RequestProfile, Tier

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


def classify_l0(req: ChatRequest) -> RequestProfile:
    text, has_image = _flatten(req)
    n_in = estimate_tokens(text)
    n_out = req.max_tokens or min(2048, max(256, n_in // 2))

    caps: set[Capability] = set()
    if req.tools or req.functions:
        caps.add(Capability.TOOLS)
    if req.response_format:
        caps.add(Capability.JSON)
    if has_image:
        caps.add(Capability.VISION)
    if req.stream:
        caps.add(Capability.STREAM)

    low = text.lower()
    has_code = bool(CODE_RE.search(text)) or any(h in low for h in CODE_HINTS)
    has_math = bool(MATH_RE.search(text))
    reasoning = any(h in low for h in REASONING_HINTS)
    extraction = any(h in low for h in EXTRACTION_HINTS)

    # Complexity as a continuous value — the tier is derived from it by
    # thresholds.
    score = 0.25
    score += min(0.30, n_in / 12_000)  # length is a signal in itself
    score += 0.20 if has_code else 0.0
    score += 0.15 if has_math else 0.0
    score += 0.15 if reasoning else 0.0
    score += 0.10 if req.tools else 0.0
    score -= 0.15 if (extraction and not reasoning and n_in < 800) else 0.0
    score = max(0.0, min(1.0, score))

    if has_code:
        task = "code"
    elif reasoning:
        task = "reasoning"
    elif extraction:
        task = "extraction"
    else:
        task = "chat"

    return RequestProfile(
        task_probs={task: 0.7, "chat": 0.3} if task != "chat" else {"chat": 1.0},
        complexity=score,
        confidence=0.55,
        tier=tier_from(score, 0.55),
        required_caps=caps,
        est_input_tokens=n_in,
        est_output_tokens=n_out,
        language="ru" if CYRILLIC_RE.search(text) else "en",
        classifier_source="rules",
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


def classify(
    req: ChatRequest, enable_l1: bool = False, head_path: str | None = None
) -> RequestProfile:
    profile = classify_l0(req)
    if not enable_l1:
        return profile

    encoder = _static_encoder()
    if encoder is None:
        return profile  # package not installed — silently stay on rules

    text, _ = _flatten(req)
    profile.embedding = encoder.encode([text[:4000]])[0]

    head = _load_head(head_path)
    if head is None:
        return profile  # head not trained yet: embedding is there, class is not

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
