"""Policy layer.

Between ingress and the router stands one question: what is this
request actually allowed to do. The answer is not a score but a set of
hard flags:

  privacy_class  — where the request must not go (data-training providers)
  latency_class  — what we optimize: TTFT or quota savings
  quality_floor  — below which quality a model must not be picked
  guard          — whether there are injections in untrusted blocks

Policy is deliberately separated from scoring. A score can be outweighed:
a low-quality model still wins if the others are busy. A rule cannot be
outweighed — a request with PII physically does not go to the cloud.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .guard import Guard, GuardResult
from .pii import detect as detect_pii
from .schemas import ChatRequest, RequestProfile, Tier
from .stakes import StakesProfile, classify_stakes

# Minimum quality per tier. Prevents a router out of quota from
# downgrading an L task to a model that obviously cannot carry it: a
# honest local model is better than fast garbage from a free tier.
QUALITY_FLOOR = {Tier.S: 0.0, Tier.M: 0.35, Tier.L: 0.55}

TIER_ORDER = {Tier.S: 0, Tier.M: 1, Tier.L: 2}


@dataclass
class PolicyDecision:
    privacy: str = "default"
    privacy_reasons: list[str] = field(default_factory=list)
    latency_class: str = "interactive"
    quality_floor: float = 0.0
    stakes: str = "normal"
    stakes_reason: str = ""
    deadline_factor: float = 1.0
    escalation_factor: float = 1.0
    min_tier: Tier | None = None
    routing_profile: str = "auto"
    guard: GuardResult = field(default_factory=GuardResult)
    neutralized: int = 0
    blocked: bool = False
    block_reason: str = ""

    def as_meta(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.stakes != "normal":
            out["stakes"] = self.stakes
            out["stakes_reason"] = self.stakes_reason
            out["deadline_factor"] = self.deadline_factor
            out["escalation_factor"] = self.escalation_factor
        if self.privacy_reasons:
            out["pii"] = self.privacy_reasons
        if self.routing_profile and self.routing_profile != "auto":
            out["profile"] = self.routing_profile
        if self.guard.tripped:
            out["guard"] = self.guard.kinds
        if self.neutralized:
            out["neutralized_blocks"] = self.neutralized
        return out


class Policy:
    def __init__(
        self,
        *,
        enable_pii: bool = True,
        guard: Guard | None = None,
        quality_floor: dict[Tier, float] | None = None,
    ):
        self.enable_pii = enable_pii
        self.guard = guard or Guard(enabled=False)
        self.quality_floor = quality_floor or QUALITY_FLOOR

    def apply(self, req: ChatRequest, profile: RequestProfile) -> PolicyDecision:
        """Mutates req (privacy, untrusted-block wrappers) and returns the
        decision. The mutation is intentional: nothing downstream should
        have to remember that privacy can be derived rather than set."""
        # Routing profiles:
        # - privacy_only: strictly on-device (privacy=strict)
        # - code_completion: interactive latency class, low-TTFT
        # - reasoning: target Tier.L reasoning models with high quality floor
        if req.profile == "privacy_only":
            req.privacy = "strict"
        elif req.profile == "code_completion":
            req.latency_class = "interactive"
        elif req.profile == "reasoning":
            profile.tier = Tier.L

        # What a wrong answer costs here. Declared by the caller where it
        # can be; inferred only upward, never down — a router that quietly
        # decides a request is unimportant is worse than one that treats
        # everything alike.
        stakes: StakesProfile = classify_stakes(req, req.stakes)
        if stakes.min_tier is not None and TIER_ORDER[stakes.min_tier] > TIER_ORDER[
            profile.tier
        ]:
            profile.tier = stakes.min_tier

        d = PolicyDecision(
            privacy=req.privacy,
            latency_class=req.latency_class,
            quality_floor=max(
                0.0,
                self.quality_floor.get(profile.tier, 0.0) + stakes.quality_floor_bonus,
            ),
            stakes=stakes.level,
            stakes_reason=stakes.reason,
            deadline_factor=stakes.deadline_factor,
            escalation_factor=stakes.escalation_factor,
            min_tier=stakes.min_tier,
            routing_profile=req.profile,
        )

        if req.profile == "privacy_only":
            d.privacy_reasons.append("profile:privacy_only")

        # PII is a rule, not a score: free tiers are paid for with data,
        # and the decision must not depend on attentiveness at prompt time.
        if self.enable_pii and req.privacy != "strict":
            kinds = detect_pii(_scannable_text(req))
            if kinds:
                req.privacy = "strict"
                d.privacy = "strict"
                d.privacy_reasons = sorted(kinds)

        d.guard = self.guard.scan(req)
        if d.guard.tripped:
            if self.guard.action == "block":
                d.blocked = True
                d.block_reason = "injection in untrusted block: " + ", ".join(
                    d.guard.kinds
                )
            else:
                d.neutralized = self.guard.neutralize_request(req)
        return d


def _scannable_text(req: ChatRequest) -> str:
    parts: list[str] = []
    for m in req.messages:
        if isinstance(m.content, str):
            parts.append(m.content)
        elif isinstance(m.content, list):
            parts.extend(
                b.get("text", "")
                for b in m.content
                if isinstance(b, dict) and b.get("type") == "text"
            )
    return "\n".join(parts)
