"""Prefix discipline for provider-side caching.

Provider-side prefix caching (OpenAI and DeepSeek — automatic from ~1024
tokens, Anthropic — via explicit cache_control markers, Gemini — context
caching) needs no code but does need discipline. It breaks trivially:

  - reorder blocks → different prefix → cache miss;
  - insert a timestamp into the system prompt → the prefix is unique
    every time;
  - switch to another provider → it does not have this prefix at all.

Hence the three rules below: stable ordering (system → tools → static →
history), no reordering within the history, and sticky routing by prefix
hash (see router.Router).
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from .registry import Offering
from .schemas import ChatRequest

# Blocks in order of increasing mutability. System and developer messages
# sit first; conversational turns (user, assistant, tool, function) preserve
# their exact chronological sequence in the history.
ROLE_ORDER = {"system": 0, "developer": 0}


def order_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """System blocks first, everything else in original chronological order.

    A stable sort is mandatory: reordering within the history changes the
    prefix entirely and zeroes out the cache.
    """
    return sorted(messages, key=lambda m: ROLE_ORDER.get(m.get("role"), 1))


def prefix_hash(req: ChatRequest) -> str:
    """Hash of the static zone: system messages + tool descriptions.

    Used for sticky routing: two requests with identical static content
    are best sent to the same provider, even if their session_id differs.
    """
    static = {
        "system": [
            m.content for m in req.messages if m.role in ("system", "developer")
        ],
        "tools": req.tools or req.functions or [],
    }
    blob = json.dumps(static, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _mark(block: dict[str, Any]) -> dict[str, Any]:
    return {**block, "cache_control": {"type": "ephemeral"}}


def apply_cache_points(payload: dict[str, Any], o: Offering) -> dict[str, Any]:
    """Cache points are placed at the END of the static zone.

    Only for providers with explicit mode (prompt_cache: explicit). For
    others an extra field in the request body is a reason to return 400,
    not to speed up.
    """
    if o.prompt_cache != "explicit":
        return payload

    messages = payload.get("messages") or []
    last_static = max(
        (i for i, m in enumerate(messages) if m.get("role") in ("system", "developer")),
        default=-1,
    )
    if last_static >= 0:
        messages[last_static] = _mark(messages[last_static])

    if tools := payload.get("tools"):
        tools[-1] = _mark(tools[-1])
    return payload


def build_payload(
    req: ChatRequest, o: Offering, *, stream: bool = False
) -> dict[str, Any]:
    """Provider request body: router extensions stripped, block order
    stabilized, cache points placed."""
    payload = req.provider_payload(o.model_id)
    if payload.get("messages"):
        payload["messages"] = order_messages(payload["messages"])
    payload = apply_cache_points(payload, o)
    payload["stream"] = stream
    if getattr(req, "thinking_mode", False):
        if "mlx" in o.provider.lower():
            payload["enable_thinking"] = True
            payload["chat_template_kwargs"] = {"enable_thinking": True}
        elif "ollama" in o.provider.lower() or o.is_local:
            payload["options"] = {**payload.get("options", {}), "num_predict": 4096}
    if o.provider == "openrouter":
        # Ask to return the request cost: without this usage.cost is not
        # sent, and the executor's watchdog has nothing to check.
        payload["usage"] = {"include": True}
    return payload
