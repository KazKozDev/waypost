"""Internal and external data schemas.

The external contract is OpenAI-compatible so any existing client works
by changing base_url. The internal one is RequestProfile, passed between
the classifier, router and executor.
"""
from __future__ import annotations

import time
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------
# External contract (OpenAI-compatible)
# --------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: Literal["system", "developer", "user", "assistant", "tool", "function"]
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    function_call: dict[str, Any] | None = None


class ChatRequest(BaseModel):
    # "auto" — let the router choose. An explicit model id bypasses routing.
    model: str = "auto"
    messages: list[ChatMessage]
    temperature: float = 1.0
    top_p: float | None = None
    max_tokens: int | None = None
    stream: bool = False
    stop: list[str] | str | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any | None = None
    functions: list[dict[str, Any]] | None = None
    function_call: Any | None = None
    response_format: dict[str, Any] | None = None
    seed: int | None = None

    # Router extensions (not part of the OpenAI spec, ignored by providers)
    privacy: Literal["default", "strict"] = "default"
    latency_class: Literal["interactive", "batch"] = "interactive"
    profile: Literal[
        "auto", "code_completion", "reasoning", "privacy_only", "balanced"
    ] = "auto"
    session_id: str | None = None
    no_cache: bool = False
    thinking_mode: bool = False
    # A retry with the same key must not run twice: a client whose
    # network dropped must not have to know whether the request arrived.
    idempotency_key: str | None = None

    def provider_payload(self, model_id: str) -> dict[str, Any]:
        """Provider request body: router extensions stripped out."""
        drop = {
            "privacy",
            "latency_class",
            "profile",
            "session_id",
            "no_cache",
            "thinking_mode",
            "idempotency_key",
            "model",
            "stream",
        }
        body = self.model_dump(exclude_none=True, exclude=drop)
        body["model"] = model_id
        return body


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class RouterMeta(BaseModel):
    """Router decision diagnostics. Returned in the response — without
    it, debugging routing turns into guesswork."""

    provider: str | None = None
    model: str | None = None
    task_class: str | None = None
    complexity_tier: str | None = None
    classifier_source: str | None = None
    cache: Literal["miss", "exact", "semantic", "semantic_degraded"] = "miss"
    attempts: int = 0
    fallback_path: list[str] = Field(default_factory=list)
    # Which provider key worked — without this it is impossible to
    # understand why the quota ran out "unexpectedly".
    key_index: int = 0
    hedged: bool = False
    escalated: bool = False
    verify_reason: str | None = None
    # The answer is approximate: served from the semantic cache with a
    # lowered threshold because all providers refused. Telling the
    # client this is fresh is not allowed.
    approximate: bool = False
    policy: dict[str, Any] = Field(default_factory=dict)
    latency_ms: int = 0
    routing_profile: str | None = None
    routing_source: str | None = None
    request_id: str | None = None
    l1_prediction: dict[str, Any] | None = None
    is_exploration: bool = False
    saved_usd: float = 0.0


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[dict[str, Any]]
    usage: Usage = Field(default_factory=Usage)
    router: RouterMeta = Field(default_factory=RouterMeta)


# --------------------------------------------------------------------------
# Internal contract
# --------------------------------------------------------------------------


class Tier(str, Enum):
    S = "S"  # extraction, classification, short rewriting
    M = "M"  # summarization, ordinary dialogue, simple code
    L = "L"  # reasoning, long code, analysis


class Capability(str, Enum):
    TOOLS = "tools"
    JSON = "json"
    VISION = "vision"
    STREAM = "stream"


class RequestProfile(BaseModel):
    """What the classifier tells the router.

    Intentionally a distribution, not a hard label: low confidence must
    push the tier up rather than silently pick the argmax.
    """

    task_probs: dict[str, float] = Field(default_factory=dict)
    complexity: float = 0.5  # 0..1
    confidence: float = 0.5  # 0..1
    tier: Tier = Tier.M
    required_caps: set[Capability] = Field(default_factory=set)
    est_input_tokens: int = 0
    est_output_tokens: int = 0
    language: str = "unknown"
    classifier_source: Literal["rules", "L1", "L2"] = "rules"
    embedding: Any | None = None  # reused by the cache and the bandit

    @property
    def task_class(self) -> str:
        if not self.task_probs:
            return "unknown"
        return max(self.task_probs.items(), key=lambda kv: kv[1])[0]

    @property
    def est_total_tokens(self) -> int:
        return self.est_input_tokens + self.est_output_tokens


class RouterError(Exception):
    """An error the executor can do nothing more about."""

    def __init__(
        self, message: str, status_code: int = 502, meta: RouterMeta | None = None
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.meta = meta or RouterMeta()
