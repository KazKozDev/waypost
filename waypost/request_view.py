"""Bounded, role-aware input for routing; never rewrites the provider request."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .schemas import ChatMessage, ChatRequest

REPRESENTATION_VERSION = "request-v2"
MAX_ROUTING_CHARS = 4000

FOLLOWUP_RE = re.compile(
    r"(?i)(^(?:и |а |также |теперь |ещё |еще |and |also |now |handle |try again|"
    r"сделай |переделай |добавь |исправь )|"
    r"\b(?:it|that|previous|above|same|её|ее|него|неё|выше|предыдущ\w*)\b)"
)
LANGUAGE_RE = re.compile(
    r"(?i)(?:answer|respond|reply|write|translate[^:\n]{0,60}?\b(?:into|to))\s+(?:in\s+)?"
    r"(english|russian|spanish|french|german|chinese)\b|"
    r"(?:отвечай|ответь|напиши|переведи[^:\n]{0,60}?)\s+на\s+"
    r"(русск\w*|английск\w*|испанск\w*|французск\w*|немецк\w*|китайск\w*)"
)
LANGUAGES = {
    "english": "en", "russian": "ru", "spanish": "es", "french": "fr",
    "german": "de", "chinese": "zh", "русск": "ru", "английск": "en",
    "испанск": "es", "французск": "fr", "немецк": "de", "китайск": "zh",
}


def message_text(message: ChatMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    if isinstance(message.content, list):
        return "\n".join(
            b.get("text", "") for b in message.content
            if isinstance(b, dict) and b.get("type") in ("text", "input_text")
            and isinstance(b.get("text"), str)
        )
    return ""


def bounded(text: str, limit: int) -> str:
    """Preserve both instructions before a source and instructions after it."""
    if len(text) <= limit:
        return text
    marker = "\n[... omitted ...]\n"
    room = limit - len(marker)
    return text[:room // 2] + marker + text[-(room - room // 2):]


def _language(text: str) -> str | None:
    matches = list(LANGUAGE_RE.finditer(text))
    if not matches:
        return None
    value = next(v for v in matches[-1].groups() if v).lower()
    return next((code for name, code in LANGUAGES.items() if value.startswith(name)), None)


@dataclass(frozen=True)
class RequestView:
    latest: str
    history: str
    tool_result: str
    constraints: str
    tools: str
    language: str
    continuation: bool

    @property
    def intent_text(self) -> str:
        # Tool output and system boilerplate are evidence/constraints,
        # not a new user task. Only referential turns inherit an intent.
        return self.latest + ("\n" + self.history if self.continuation else "")

    @property
    def embedding_text(self) -> str:
        sections = [("Current task", bounded(self.latest, 2300))]
        if self.constraints:
            sections.append(("Constraints", bounded(self.constraints, 350)))
        if self.tools:
            sections.append(("Available tools", bounded(self.tools, 300)))
        if self.history:
            sections.append(("Referenced conversation", bounded(self.history, 650)))
        if self.tool_result:
            sections.append(("Tool evidence", bounded(self.tool_result, 200)))
        return "\n\n".join(f"{name}:\n{text}" for name, text in sections)[:MAX_ROUTING_CHARS]


def build_request_view(req: ChatRequest) -> RequestView:
    user_indices = [i for i, m in enumerate(req.messages) if m.role == "user"]
    index = user_indices[-1] if user_indices else len(req.messages) - 1
    latest = message_text(req.messages[index]) if index >= 0 else ""
    continuation = bool(FOLLOWUP_RE.search(bounded(latest, 2300))) and len(user_indices) > 1
    history = ""
    if continuation:
        previous = user_indices[-2]
        # One preceding user turn and its answer, not an unbounded history.
        snippets = [bounded(message_text(req.messages[previous]), 400)]
        answers = [m for m in req.messages[previous + 1:index] if m.role == "assistant"]
        if answers:
            snippets.append(bounded(message_text(answers[-1]), 250))
        history = "\n".join(snippets)
    results = [m for m in req.messages[index + 1:] if m.role in ("tool", "function")]
    tool_result = bounded(message_text(results[-1]), 200) if results else ""

    # Explicit output-language requirements outrank the language of quoted
    # source material. Honour message-role precedence without including the
    # whole system prompt in the task embedding.
    language = None
    for role in ("system", "developer", "user"):
        messages = [m for m in req.messages if m.role == role] if role != "user" else (
            [req.messages[index]] if index >= 0 else []
        )
        for message in reversed(messages):
            language = _language(message_text(message))
            if language:
                break
        if language:
            break
    language = language or ("ru" if re.search(r"[\u0400-\u04ff]", latest) else "en")
    constraints = [f"output_language={language}"]
    if req.response_format:
        constraints.append("response_format=" + bounded(json.dumps(req.response_format, ensure_ascii=False), 220))
    if req.max_tokens is not None:
        constraints.append(f"max_output_tokens={req.max_tokens}")
    if req.tool_choice is not None:
        constraints.append("tool_choice=" + bounded(json.dumps(req.tool_choice, ensure_ascii=False), 100))
    names = []
    for tool in req.tools or req.functions or []:
        function = tool.get("function", tool)
        params = function.get("parameters") or {}
        names.append(str(function.get("name", "")) + "(" + ",".join(params.get("required") or []) + ")")
    return RequestView(latest, history, tool_result, "; ".join(constraints),
                       ", ".join(names), language, continuation)
