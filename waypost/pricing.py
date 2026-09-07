"""Determining whether a model is free or costs money.

The project rule is to spend only free quotas. So the question "is this
model paid" must be settled by a mechanism, not by human memory, and
settled fail-closed: unknown → treat as paid.

Evidence is ranked by strength:

  1. BILLED     — the provider billed in the response (usage.cost > 0).
                  Strongest of all: this is a fact, not a declaration.
  2. API        — prices from GET /models (OpenRouter returns pricing.prompt;
                  some OpenAI-compatible hosts have their own fields).
  3. ID         — a ':free' / '-free' suffix in the model identifier.
  4. MANIFEST   — declared by a human in config/providers.yaml.
  5. UNKNOWN    — declared nowhere → treated as paid.

Merge: any PAID evidence overrides any FREE. An error this way costs an
unavailable model; an error the other way costs money.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

# Suffixes of free variants (OpenRouter convention).
FREE_MARKERS = (":free", "-free")

# Per-token price fields in a GET /models response across hosts.
_PRICE_KEYS = (
    "prompt",
    "completion",
    "input",
    "output",
    "request",
    "image",
    "input_cost_per_token",
    "output_cost_per_token",
    "prompt_token_price",
    "completion_token_price",
    "price_per_input_token",
    "price_per_output_token",
)

# Cost fields for a request already executed (usage / usage.cost_details).
_COST_KEYS = (
    "cost",
    "total_cost",
    "cost_usd",
    "prompt_cost",
    "completion_cost",
    "upstream_inference_cost",
)


class Cost(str, Enum):
    FREE = "free"
    PAID = "paid"
    UNKNOWN = "unknown"


class Source(str, Enum):
    BILLED = "billed"  # a bill in the provider response
    API = "api"  # prices from GET /models
    ID = "id"  # a :free suffix
    MANIFEST = "manifest"  # declared in the manifest
    UNKNOWN = "unknown"  # declared nowhere


_STRENGTH = {
    Source.BILLED: 4,
    Source.API: 3,
    Source.ID: 2,
    Source.MANIFEST: 1,
    Source.UNKNOWN: 0,
}


@dataclass(frozen=True)
class Verdict:
    cost: Cost
    source: Source
    detail: str = ""

    @property
    def is_free(self) -> bool:
        """Only a provably free model is free. UNKNOWN is not free."""
        return self.cost is Cost.FREE

    def __str__(self) -> str:
        d = f" ({self.detail})" if self.detail else ""
        return f"{self.cost.value}/{self.source.value}{d}"


UNKNOWN_VERDICT = Verdict(Cost.UNKNOWN, Source.UNKNOWN, "no price data")


def _num(value: Any) -> float | None:
    """Prices arrive as strings ('0', '0.0000006') and as numbers."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def parse_prices(model: dict[str, Any]) -> dict[str, float]:
    """All recognized prices of a model from a /models response.

    We look both in flat fields and in nested pricing/prices — the shape
    differs from host to host, the meaning is the same.
    """
    out: dict[str, float] = {}
    sources: list[dict[str, Any]] = [model]
    for nest in ("pricing", "prices", "cost", "price"):
        if isinstance(model.get(nest), dict):
            sources.append(model[nest])
    for src in sources:
        for key in _PRICE_KEYS:
            if key in src and (v := _num(src[key])) is not None:
                out[key] = v
    return out


def from_api(model: dict[str, Any]) -> Verdict:
    """Verdict from a GET /models response. UNKNOWN if it has no prices."""
    flag = model.get("free")
    if isinstance(flag, bool):
        return Verdict(
            Cost.FREE if flag else Cost.PAID, Source.API, "free field in /models"
        )

    prices = parse_prices(model)
    if not prices:
        return UNKNOWN_VERDICT

    positive = {k: v for k, v in prices.items() if v > 0}
    if positive:
        top = max(positive.items(), key=lambda kv: kv[1])
        return Verdict(Cost.PAID, Source.API, f"{top[0]}={top[1]:g}")
    # A negative price (OpenRouter sets -1 for dynamic rates) means
    # "price unknown", not "free".
    if any(v < 0 for v in prices.values()):
        return Verdict(Cost.UNKNOWN, Source.API, "dynamic rate")
    return Verdict(Cost.FREE, Source.API, "all prices = 0")


def from_model_id(model_id: str) -> Verdict:
    """A ':free' suffix is a convention, but a verifiable one: such models
    on OpenRouter always have a 0 price. No converse: the absence of the
    suffix proves nothing."""
    low = (model_id or "").lower()
    for mark in FREE_MARKERS:
        if low.endswith(mark) or f"{mark}/" in low:
            return Verdict(Cost.FREE, Source.ID, f"suffix {mark}")
    return UNKNOWN_VERDICT


def from_manifest(declared: bool | None) -> Verdict:
    """Source A. None means "not declared" — that is UNKNOWN, not free:
    this is exactly where paid models used to leak through."""
    if declared is None:
        return Verdict(Cost.UNKNOWN, Source.UNKNOWN, "free not declared in manifest")
    return Verdict(Cost.FREE if declared else Cost.PAID, Source.MANIFEST, "manifest")


def billed_cost(body: dict[str, Any]) -> float | None:
    """How much the provider charged for this answer. None if not reported.

    The last line of defense: if a bill was actually issued, the model is
    paid, whatever the manifest and /models say.
    """
    usage = body.get("usage")
    if not isinstance(usage, dict):
        return None
    total = 0.0
    found = False
    for key in _COST_KEYS:
        if (v := _num(usage.get(key))) is not None:
            total = max(total, v)
            found = True
    details = usage.get("cost_details")
    if isinstance(details, dict):
        for v in details.values():
            if (n := _num(v)) is not None:
                total = max(total, n)
                found = True
    return total if found else None


def merge(*verdicts: Verdict) -> Verdict:
    """Merge evidence. PAID takes priority over FREE from any source — a
    fail-closed policy; on a tie, the strongest source wins.
    """
    real = [v for v in verdicts if v.cost is not Cost.UNKNOWN]
    if not real:
        strongest = max(
            verdicts, key=lambda v: _STRENGTH[v.source], default=UNKNOWN_VERDICT
        )
        return strongest if verdicts else UNKNOWN_VERDICT
    paid = [v for v in real if v.cost is Cost.PAID]
    pool = paid or real
    return max(pool, key=lambda v: _STRENGTH[v.source])


def classify(
    model_id: str,
    *,
    api_model: dict[str, Any] | None = None,
    declared: bool | None = None,
    billed: float | None = None,
) -> Verdict:
    """Single entry point: gather all available evidence into one verdict."""
    verdicts = [from_manifest(declared), from_model_id(model_id)]
    if api_model is not None:
        verdicts.append(from_api(api_model))
    if billed is not None:
        verdicts.append(
            Verdict(
                Cost.PAID if billed > 0 else Cost.FREE,
                Source.BILLED,
                f"billed {billed:g}",
            )
        )
    return merge(*verdicts)


# Commercial market baseline prices in USD per 1M tokens (input, output).
# Used to calculate real money saved when routing to free tiers and local models.
# - S: GPT-4o-mini / Claude 3.5 Haiku baseline
# - M: Claude 3.5 Sonnet / GPT-4o baseline
# - L: OpenAI o1 / Claude 3.7 Thinking baseline
BASELINE_PRICING_PER_1M: dict[str, tuple[float, float]] = {
    "S": (0.15, 0.60),
    "M": (3.00, 15.00),
    "L": (15.00, 60.00),
}
BASELINE_NAME = "Waypost commercial baseline v1"


def baseline_cost(prompt_tokens: int, completion_tokens: int, tier: str = "M") -> float:
    """Estimated commercial cost in USD for the given token volume and tier."""
    t = str(tier).upper()
    rates = BASELINE_PRICING_PER_1M.get(t, BASELINE_PRICING_PER_1M["M"])
    in_cost = (prompt_tokens / 1_000_000.0) * rates[0]
    out_cost = (completion_tokens / 1_000_000.0) * rates[1]
    return round(in_cost + out_cost, 6)


def calculate_savings(
    prompt_tokens: int,
    completion_tokens: int,
    tier: str = "M",
    actual_cost: float = 0.0,
) -> float:
    """Calculates saved USD compared to commercial standard APIs."""
    base = baseline_cost(prompt_tokens, completion_tokens, tier)
    return max(0.0, round(base - actual_cost, 6))
