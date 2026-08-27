"""Cascade verifier.

The cascade saves quota only under one condition: a cheap-model failure
must be CHEAP TO DETECT. If the check costs as much as calling the strong
model, there is no saving. So almost everything here is deterministic
code, and neural nets plug in optionally and only where no deterministic
check exists (answer groundedness in context).

The cascade math: it pays off while p_fail · cost_L < cost_L − cost_S,
that is, with a failure share below ~30-40%. The share is measured by
telemetry (`Telemetry.cascade_stats`), not by a guess — if it is above
that, the cascade must be turned off, not tuned.

Checks in increasing cost:
  empty / truncated / looping                 — string comparison
  JSON: parse + schema                        — json.loads + mini-validator
  tool_calls: name and required arguments     — check against tools description
  answer language did not match request language — character count
  refusal instead of an answer                — narrow list of phrasings
  context groundedness (RAG)                  — HHEM, optional
  instruction contradiction                   — NLI, optional
"""
from __future__ import annotations

import ast
import json
import re
from typing import Any, Callable, NamedTuple

from .schemas import ChatRequest, RequestProfile

CYRILLIC_RE = re.compile(r"[Ѐ-ӿ]")
LATIN_RE = re.compile(r"[A-Za-z]")
CODE_FENCE_RE = re.compile(r"```")

# A narrow list: a wide one would false-positive on honest answers that
# merely contain the words "I cannot".
REFUSAL_RE = re.compile(
    r"(?i)^(?:\W{0,3})("
    r"i(?:'m| am) (?:sorry|unable)|i can(?:no|')t (?:help|assist|do)"
    r"|as an ai (?:language )?model"
    r"|извин\w+, (?:но )?я не могу|к сожалению, я не могу"
    r"|я не могу (?:помочь|выполнить|ответить)"
    r"|я всего лишь (?:языковая )?модель"
    r")"
)


class VerifyResult(NamedTuple):
    """A NamedTuple, not a dataclass: the caller unpacks the result as a
    (ok, reason) pair."""

    ok: bool
    reason: str = ""


def _content(body: dict[str, Any]) -> str:
    try:
        msg = body["choices"][0]["message"]
        return msg.get("content") or msg.get("reasoning") or msg.get("reasoning_content") or ""
    except (KeyError, IndexError, TypeError):
        return ""


def _tool_calls(body: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        return body["choices"][0]["message"].get("tool_calls") or []
    except (KeyError, IndexError, TypeError, AttributeError):
        return []


def _finish_reason(body: dict[str, Any]) -> str | None:
    try:
        return body["choices"][0].get("finish_reason")
    except (KeyError, IndexError, TypeError):
        return None


# --------------------------------------------------------------- JSON

_TYPES = {
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def validate_schema(value: Any, schema: dict[str, Any]) -> str | None:
    """A mini JSON Schema validator: type, required, properties, enum,
    items.

    A full jsonschema is not needed here: the hot path checks a model
    answer, not user input, and only coarse violations matter. Returns
    the first error description, or None.
    """
    expected = schema.get("type")
    if expected in _TYPES:
        if expected == "number" and isinstance(value, bool):
            return "boolean instead of number"
        if not isinstance(value, _TYPES[expected]):
            return f"expected {expected}, got {type(value).__name__}"
    if (enum := schema.get("enum")) is not None and value not in enum:
        return f"value outside enum: {value!r}"
    if isinstance(value, dict):
        for name in schema.get("required", []):
            if name not in value:
                return f"missing required field {name!r}"
        for name, sub in (schema.get("properties") or {}).items():
            if name in value and (err := validate_schema(value[name], sub)):
                return f"{name}: {err}"
    if isinstance(value, list) and (items := schema.get("items")):
        for i, item in enumerate(value):
            if err := validate_schema(item, items):
                return f"[{i}]: {err}"
    return None


def _schema_of(response_format: dict[str, Any]) -> dict[str, Any] | None:
    if response_format.get("type") != "json_schema":
        return None
    spec = response_format.get("json_schema") or {}
    return spec.get("schema") or None


# ----------------------------------------------------------- degeneration


def is_degenerate(text: str, *, min_len: int = 200, repeat_threshold: int = 5) -> bool:
    """Looping: the same line (or a short tail) repeats.

    Cheap models break into loops noticeably more often than expensive
    ones, and this is exactly the failure the cascade must catch: the
    answer is formally non-empty.
    """
    if len(text) < min_len:
        return False
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        top = max(set(lines), key=lines.count)
        if len(top) > 8 and lines.count(top) >= repeat_threshold:
            return True
    tail = text[-60:].strip()
    return bool(tail) and len(tail) > 12 and text.count(tail) >= repeat_threshold


def _check_code_syntax(content: str) -> str | None:
    """Check syntax of Python code fences."""
    pattern = re.compile(r"```(?:python|py)\n(.*?)```", re.DOTALL | re.IGNORECASE)
    for match in pattern.finditer(content):
        code = match.group(1).strip()
        if not code:
            continue
        try:
            ast.parse(code)
        except SyntaxError as exc:
            return f"syntax_error: {exc.msg}"
    return None


class Verifier:
    """grounding(context, answer) -> 0..1 — an HHEM-like groundedness
    classifier (vectara/hallucination_evaluation_model). nli(premise,
    hypothesis) -> 0..1 — an instruction-following model. Both optional:
    better not to block than to block blindly."""

    def __init__(
        self,
        grounding: Callable[[str, str], float] | None = None,
        grounding_threshold: float = 0.5,
        nli: Callable[[str, str], float] | None = None,
        nli_threshold: float = 0.4,
        check_language: bool = True,
        check_refusal: bool = True,
    ):
        self.grounding = grounding
        self.grounding_threshold = grounding_threshold
        self.nli = nli
        self.nli_threshold = nli_threshold
        self.check_language = check_language
        self.check_refusal = check_refusal

    def verify(
        self, req: ChatRequest, profile: RequestProfile | None, body: dict[str, Any]
    ) -> VerifyResult:
        content = _content(body)
        calls = _tool_calls(body)

        if not content.strip() and not calls:
            return VerifyResult(False, "empty")

        # Truncated by a limit. If the client set the limit, that is
        # their decision, not a model failure.
        if _finish_reason(body) == "length" and req.max_tokens is None:
            return VerifyResult(False, "truncated")

        if req.response_format and (err := self._check_json(req, content)):
            return VerifyResult(False, err)

        if req.tools and calls and (err := self._check_tool_calls(req, calls)):
            return VerifyResult(False, err)

        if content.strip():
            if is_degenerate(content):
                return VerifyResult(False, "degenerate")
            if (
                self.check_refusal
                and len(content) < 400
                and REFUSAL_RE.search(content.strip())
            ):
                return VerifyResult(False, "refusal")
            if (
                self.check_language
                and profile is not None
                and self._language_mismatch(profile, content)
            ):
                return VerifyResult(False, "language_mismatch")
            if code_err := _check_code_syntax(content):
                return VerifyResult(False, code_err)

        if err := self._check_grounding(req, content):
            return VerifyResult(False, err)

        return VerifyResult(True, "")

    # ------------------------------------------------------------ details
    def _check_json(self, req: ChatRequest, content: str) -> str | None:
        rf = req.response_format or {}
        if rf.get("type") not in ("json_object", "json_schema"):
            return None
        try:
            parsed = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            return "invalid_json"
        if (schema := _schema_of(rf)) is not None:
            if validate_schema(parsed, schema):
                return "schema_violation"
        return None

    def _check_tool_calls(
        self, req: ChatRequest, calls: list[dict[str, Any]]
    ) -> str | None:
        """Calling a nonexistent function or with broken arguments is the
        most common way a cheap model silently breaks a pipeline."""
        known = {}
        for t in req.tools or []:
            fn = t.get("function") or {}
            if name := fn.get("name"):
                known[name] = fn.get("parameters") or {}
        for call in calls:
            fn = call.get("function") or {}
            name = fn.get("name")
            if name not in known:
                return "unknown_tool"
            raw = fn.get("arguments")
            if isinstance(raw, str):
                try:
                    args = json.loads(raw or "{}")
                except json.JSONDecodeError:
                    return "invalid_tool_arguments"
            else:
                args = raw or {}
            if validate_schema(args, known[name]):
                return "invalid_tool_arguments"
        return None

    def _language_mismatch(self, profile: RequestProfile, content: str) -> bool:
        """Russian request — English answer. A common failure of cheap
        models. Code and formulas are excluded from the check."""
        if profile.language != "ru":
            return False
        body = CODE_FENCE_RE.split(content)[0]
        cyr = len(CYRILLIC_RE.findall(body))
        lat = len(LATIN_RE.findall(body))
        return lat > 40 and cyr < lat * 0.15

    def _check_grounding(self, req: ChatRequest, content: str) -> str | None:
        if self.grounding is None or not content.strip():
            return None
        context = "\n".join(
            m.content
            for m in req.messages
            if m.role in ("tool", "system") and isinstance(m.content, str)
        )
        if not context.strip():
            return None  # not RAG — nothing to check
        try:
            score = float(self.grounding(context, content))
        except Exception:  # noqa: BLE001
            return None
        return "ungrounded" if score < self.grounding_threshold else None


def map_outcome(
    ok: bool,
    reason: str = "",
    *,
    status: str = "ok",
    error_detail: str | None = None,
) -> tuple[str, str, dict[str, Any]]:
    """Map verification results and status to outcome, source, and detail per Spec v5 B.2."""
    if status in ("rate_limited", "timeout", "upstream_error"):
        return "unknown", "upstream", {"status": status}

    if not ok:
        fail_reason = reason or status or "error"
        if "refusal" in fail_reason:
            return "fail", "hard_check", {"refusal": True}
        detail: dict[str, Any] = {"reason": fail_reason}
        if error_detail:
            detail["error"] = error_detail
        return "fail", "hard_check", detail

    return "pass", "hard_check", {"weak": True}
