"""What a wrong answer costs here.

Every failure was priced the same. A garbled sentence in a chat reply and
a wrong number in a financial summary both cost "one failed request", so
the router spent the same care on both: the same quality floor, the same
willingness to escalate, the same budget.

That is the wrong shape. Cost of error is a property of the request, not
of the pool, and it should move three things:

  the quality floor  — how bad a model may be before it is not offered
  escalation         — whether a doubtful answer is re-asked or shipped
  the time budget    — how long it is worth spending to get it right

Stakes are stated, not guessed, wherever the caller can state them: an
``x-waypost-stakes`` header or a ``stakes`` field. Where they are not,
they are inferred from what the request is asking for, conservatively —
the inference can only raise stakes above "normal" on positive evidence,
never lower them below what the caller asked for. A router that quietly
decides a request is unimportant is worse than one that treats
everything alike.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .schemas import ChatRequest, Tier

LEVELS = ("low", "normal", "high", "critical")
_ORDER = {name: i for i, name in enumerate(LEVELS)}


@dataclass(frozen=True)
class StakesProfile:
    level: str
    reason: str
    # Added to the tier's quality floor. A high-stakes request refuses
    # models that a low-stakes one would happily accept.
    quality_floor_bonus: float
    # Multiplies the wall-clock budget: it is worth waiting longer for an
    # answer that matters, and not worth it for one that does not.
    deadline_factor: float
    # Multiplies the verifier's willingness to escalate. Below 1.0 a
    # doubtful answer is shipped rather than re-asked.
    escalation_factor: float
    # The minimum tier this request may be routed at, if any.
    min_tier: Tier | None = None


PROFILES = {
    "low": StakesProfile(
        "low", "", quality_floor_bonus=-0.10, deadline_factor=0.7,
        escalation_factor=0.5,
    ),
    "normal": StakesProfile(
        "normal", "", quality_floor_bonus=0.0, deadline_factor=1.0,
        escalation_factor=1.0,
    ),
    "high": StakesProfile(
        "high", "", quality_floor_bonus=0.15, deadline_factor=1.5,
        escalation_factor=1.5, min_tier=Tier.M,
    ),
    "critical": StakesProfile(
        "critical", "", quality_floor_bonus=0.25, deadline_factor=2.5,
        escalation_factor=2.0, min_tier=Tier.L,
    ),
}

# Phrases that say the answer will be acted on rather than read. Narrow
# on purpose: a false "high" costs quota on every matching request, so
# the list holds only wordings that are hard to read any other way.
_HIGH_MARKERS = (
    r"\b(в\s+продакшн|в\s+прод|на\s+прод|боевой\s+сервер)\b",
    r"\b(to|in)\s+production\b",
    r"\b(миграци[яю]|migration)\s+(бд|базы|database|schema)\b",
    r"\b(медицинск|диагноз|дозиров|dosage|diagnosis|prescription)\w*",
    r"\b(юридическ|договор|контракт|legal|contract clause)\w*",
    r"\b(налог|бухгалт|invoice|tax return|payroll)\w*",
    r"\b(security (review|audit)|уязвимост\w*|vulnerabilit\w*)\b",
)
_HIGH_RE = [re.compile(p, re.IGNORECASE) for p in _HIGH_MARKERS]


def _text_of(req: ChatRequest, limit: int = 4000) -> str:
    parts: list[str] = []
    for m in req.messages:
        if isinstance(m.content, str):
            parts.append(m.content)
        elif isinstance(m.content, list):
            parts += [
                b.get("text", "") for b in m.content if isinstance(b, dict)
            ]
    return " ".join(parts)[:limit]


def _raise_to(current: str, floor: str) -> str:
    return floor if _ORDER[floor] > _ORDER[current] else current


def classify_stakes(req: ChatRequest, declared: str | None = None) -> StakesProfile:
    """Decide what a wrong answer costs for this request.

    The declared level is the starting point and the floor: inference may
    raise it, never lower it. A caller who says "low" and then asks for a
    tool call gets the tool-call treatment; a caller who says "critical"
    is never talked down to "normal" by a heuristic.

    ``declared`` defaults to what the request itself carries. Requiring
    the caller to pass a field that is already in the object is a footgun:
    forgetting it silently downgrades every request to "normal".
    """
    if declared is None:
        declared = req.stakes
    level = declared if declared in _ORDER else "normal"
    reasons: list[str] = []
    if declared in _ORDER:
        reasons.append(f"declared {declared}")

    # A tool call is not a message — it is an action the answer will
    # cause. Getting the arguments wrong has consequences outside the
    # conversation.
    if req.tools or req.functions:
        level = _raise_to(level, "high")
        reasons.append("tool call")

    # A schema means a machine will parse this. Malformed output does not
    # get a puzzled human, it gets an exception somewhere downstream.
    if (req.response_format or {}).get("type") in ("json_object", "json_schema"):
        level = _raise_to(level, "high")
        reasons.append("structured output")

    text = _text_of(req)
    for rx in _HIGH_RE:
        if (hit := rx.search(text)) is not None:
            level = _raise_to(level, "high")
            reasons.append(f"subject: {hit.group(0)[:40]}")
            break

    profile = PROFILES[level]
    return StakesProfile(
        level=profile.level,
        reason="; ".join(reasons) or "default",
        quality_floor_bonus=profile.quality_floor_bonus,
        deadline_factor=profile.deadline_factor,
        escalation_factor=profile.escalation_factor,
        min_tier=profile.min_tier,
    )
