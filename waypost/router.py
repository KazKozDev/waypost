"""Router: hard filter, then scoring.

The stages must not be mixed. If capability is weighed as a score, a
model without tool calling gets a high speed score and breaks the
request. The privacy policy is also a filter, not a weight: a score can
be outweighed, a rule cannot.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from .bandit import Bandit
from .breaker import CircuitBreaker
from .latency import LatencyTracker
from .ledger import Ledger
from .registry import Offering, Registry
from .schemas import Capability, ChatRequest, RequestProfile, Tier

TIER_ORDER = {Tier.S: 0, Tier.M: 1, Tier.L: 2}

FIM_MARKERS = (
    "<fim_prefix>",
    "<fim_suffix>",
    "<fim_middle>",
    "<PRE>",
    "<SUF>",
    "<MID>",
    "[PREFIX]",
    "[SUFFIX]",
    "[INSERT]",
    "<｜fim begin｜>",
    "<｜fim hole｜>",
    "<｜fim end｜>",
)


def has_fim_markers(req: ChatRequest) -> bool:
    """Detect if request contains Fill-In-The-Middle code completion markers."""
    for m in req.messages:
        content = m.content
        if isinstance(content, str):
            if any(marker in content for marker in FIM_MARKERS):
                return True
        elif isinstance(content, list):
            for b in content:
                if isinstance(b, dict) and any(
                    marker in b.get("text", "") for marker in FIM_MARKERS
                ):
                    return True
    return False


# Scoring weights by latency class. For background work, speed barely
# matters, but not burning the quota needed by interactive traffic does.
WEIGHTS = {
    "interactive": dict(
        quality=1.0, latency=0.6, burn=0.3, affinity=0.4, reliability=0.5,
        pressure=0.5, inflight=0.4,
    ),
    "batch": dict(
        quality=1.2, latency=0.05, burn=1.0, affinity=0.2, reliability=0.5,
        pressure=0.9, inflight=0.1,
    ),
    "code_completion": dict(
        quality=0.8, latency=1.5, burn=0.2, affinity=0.5, reliability=0.8,
        pressure=0.4, inflight=0.5,
    ),
    "reasoning": dict(
        quality=1.6, latency=0.1, burn=0.2, affinity=0.3, reliability=0.8,
        pressure=0.5, inflight=0.3,
    ),
}

# Above this many concurrent calls a provider is treated as saturated. The
# term exists to break the herd: without it the scoring is a deterministic
# argmax, every request in a burst picks the same top provider, and they
# all collect a 429 together.
SOFT_CONCURRENCY = 8


@dataclass
class Candidate:
    offering: Offering
    score: float
    reasons: dict[str, float]


def _diversify(cands: list["Candidate"], per_provider: int = 2) -> list["Candidate"]:
    """Cap how many candidates one provider may occupy in a plan.

    A plan of four models that all live behind the same host is not a
    fallback ladder — when that host rate limits or goes down, every rung
    fails at once. Ordering within the plan is preserved; only the excess
    is pushed to the tail, so nothing is lost if the pool is thin.
    """
    kept: list[Candidate] = []
    spill: list[Candidate] = []
    seen: dict[str, int] = {}
    for c in cands:
        n = seen.get(c.offering.provider, 0)
        if n < per_provider:
            seen[c.offering.provider] = n + 1
            kept.append(c)
        else:
            spill.append(c)
    return kept + spill


class Router:
    def __init__(
        self,
        registry: Registry,
        ledger: Ledger,
        breaker: CircuitBreaker,
        bandit: Bandit | None = None,
        predictor: Any | None = None,
        free_only: bool = True,
        sticky_limit: int = 2048,
        latency: LatencyTracker | None = None,
        inflight: dict[str, int] | None = None,
        stochastic: bool = True,
    ):
        self.registry = registry
        self.ledger = ledger
        self.breaker = breaker
        self.bandit = bandit
        self.predictor = predictor
        self.free_only = free_only
        self.sticky_limit = sticky_limit
        self.latency = latency or LatencyTracker()
        for o in registry.all():
            self.latency.seed(o.key, o.ttft_p50_ms)
        # Live per-offering concurrency, owned by the executor and read
        # here. A dict rather than a callback: the hot path must not pay
        # for an indirection it does not need.
        self.inflight = inflight if inflight is not None else {}
        # Thompson draw instead of the posterior mean. Deterministic
        # scoring is what makes a burst stampede onto one provider.
        self.stochastic = stochastic
        # Sticky routing: switching providers zeroes a warmed-up prefix,
        # and that is more expensive than a small scoring loss. Two
        # binding keys: the session (a dialogue) and the hash of the
        # static zone (the same system prompt + tools from different
        # sessions).
        self._sticky: dict[str, str] = {}
        self._sticky_prefix: dict[str, str] = {}

    def local_vs_cloud_threshold(self) -> float:
        """Calculates adaptive quota threshold (Spec K.2).

        When quota is abundant (near 100%), threshold is lower (~0.35) to utilize cloud.
        As quota is consumed (near 0%), threshold rises to 1.0 to enforce local routing.
        """
        with self.ledger._lock:
            if not self.ledger._buckets:
                return 0.50
            now = time.time()
            used_ratios: list[float] = []
            for buckets in self.ledger._buckets.values():
                for b in buckets.values():
                    if b.limit > 0:
                        b._roll(now)
                        used_ratios.append(min(1.0, b.used / b.limit))
            avg_used = sum(used_ratios) / len(used_ratios) if used_ratios else 0.0
            return round(min(1.0, max(0.35, 0.35 + 0.65 * avg_used)), 2)

    # ------------------------------------------------------------ filter
    def _passes(
        self,
        o: Offering,
        req: ChatRequest,
        p: RequestProfile,
        quality_floor: float = 0.0,
    ) -> bool:
        if not o.usable:
            return False
        if self.free_only and not o.free:
            return False
        # Pure admission check. Taking the half-open probe token here — as
        # allows() used to — spent it during *planning*, on candidates that
        # were often never called, and nothing ever gave it back.
        if not self.breaker.can_admit(o.provider):
            return False
        if p.est_input_tokens + p.est_output_tokens > o.ctx_window:
            return False
        if not p.required_caps.issubset(o.caps):
            return False

        # Spec I: Hard modality filtering
        has_image = any(
            isinstance(m.content, list)
            and any(
                isinstance(b, dict) and b.get("type") in ("image_url", "image")
                for b in m.content
            )
            for m in req.messages
        )
        if has_image and Capability.VISION not in o.caps:
            return False

        has_audio = any(
            isinstance(m.content, list)
            and any(
                isinstance(b, dict) and b.get("type") in ("input_audio", "audio")
                for b in m.content
            )
            for m in req.messages
        )
        if has_audio:
            # Gemma 4 26B-A4B and 31B do not support audio despite vision cap
            if "26b-a4b" in o.model_id.lower() or "31b" in o.model_id.lower():
                return False
            if "audio" not in [
                c.value if hasattr(c, "value") else str(c) for c in o.caps
            ]:
                return False

        if (req.tools or req.functions) and Capability.TOOLS not in o.caps:
            return False

        if (
            req.response_format
            and req.response_format.get("type") == "json_object"
            and Capability.JSON not in o.caps
        ):
            return False

        # Profile: privacy_only requires strictly local providers
        if req.profile == "privacy_only" and not o.is_local:
            return False
        # A free tier is paid for with data. strict — only where data is
        # not trained on (in practice: a local model).
        if req.privacy == "strict" and o.trains_on_data:
            return False
        if not self.ledger.can_afford(o, p.est_total_tokens):
            return False
        # Profile: reasoning filter to models capable of reasoning or Tier.L
        if req.profile == "reasoning":
            is_reasoning_model = any(
                k in o.model_id.lower()
                for k in ("r1", "qwq", "thinking", "reason", "o1", "o3", "deepseek")
            )
            if o.tier != Tier.L and not is_reasoning_model:
                return False
        # The quality floor from policy: a local model is not bound by
        # it, it is the end of the ladder and must always stay available.
        if (
            quality_floor
            and not o.is_local
            and o.quality_for(p.task_class) < quality_floor
        ):
            return False
        return True

    # ------------------------------------------------------------- score
    def _score(
        self,
        o: Offering,
        req: ChatRequest,
        p: RequestProfile,
        prefix: str | None = None,
    ) -> Candidate:
        # Determine weight scheme based on profile or latency class
        if req.profile in WEIGHTS:
            w = WEIGHTS[req.profile]
        elif req.latency_class in WEIGHTS:
            w = WEIGHTS[req.latency_class]
        else:
            w = WEIGHTS["interactive"]

        # Tier fit: a model below the required tier is heavily penalized
        # (it would cause an escalation), one above is lightly penalized
        # (just overspend).
        gap = TIER_ORDER[o.tier] - TIER_ORDER[p.tier]
        tier_fit = 1.0 if gap == 0 else (0.75 if gap > 0 else 0.35)

        # Measured latency, not the manifest constant: this is the only
        # thing that notices a provider degrading while still returning
        # 200 OK.
        latency = min(1.0, self.latency.get(o.key, o.ttft_p50_ms) / 5000.0)
        burn = self.ledger.burn_ratio(o, p.est_total_tokens)
        pressure = self.ledger.pressure(o)
        load = min(1.0, self.inflight.get(o.key, 0) / SOFT_CONCURRENCY)
        # Session affinity is stronger than prefix affinity: in a
        # dialogue the whole history tail is cached, not just the system
        # part.
        affinity = 0.0
        if self._sticky.get(req.session_id or "") == o.key:
            affinity = 1.0
        elif prefix and self._sticky_prefix.get(prefix) == o.key:
            affinity = 0.6
        # A warmed prefix is only worth staying for while the provider is
        # still fast. Once it drifts, stickiness just pins traffic to the
        # slow host.
        if affinity and self.latency.drifted(o.key):
            affinity = 0.0

        # Quality: predicted P(pass) or bandit / task-specific quality
        if self.predictor:
            base_quality = self.predictor.predict_p_pass(
                o, p, embedding=getattr(p, "embedding", None)
            )
        else:
            base_quality = o.quality_for(p.task_class)
        if self.bandit is None:
            quality = base_quality
        elif self.stochastic:
            quality = self.bandit.sample(p.task_class, o.key, base_quality)
        else:
            quality = self.bandit.quality(p.task_class, o.key, base_quality)

        # Code model specialization bonus for code completion profile
        code_bonus = 0.0
        if req.profile == "code_completion" or has_fim_markers(req):
            if any(
                k in o.model_id.lower()
                for k in ("code", "coder", "codestral", "starcoder", "qwen2.5-coder")
            ):
                code_bonus = 0.3

        reasons = {
            "quality": w["quality"] * quality * tier_fit + code_bonus,
            "latency": -w["latency"] * latency,
            "burn": -w["burn"] * burn,
            "pressure": -w.get("pressure", 0.5) * pressure,
            "inflight": -w.get("inflight", 0.4) * load,
            "affinity": w["affinity"] * affinity,
            "reliability": w["reliability"] * o.success_rate,
        }
        return Candidate(o, sum(reasons.values()) * o.weight, reasons)

    # -------------------------------------------------------------- plan
    def plan(
        self,
        req: ChatRequest,
        p: RequestProfile,
        limit: int = 4,
        *,
        quality_floor: float = 0.0,
        prefix: str | None = None,
    ) -> list[Candidate]:
        """Returns an ordered list of attempts, not a single model. The
        executor descends it on failures."""
        if req.model != "auto":
            low_model = req.model.lower()
            # Support targeting a tier directly (e.g. 'Tier S', 'Tier M', 'Tier L', 'tier:s')
            if (
                low_model in ("tier s", "tier m", "tier l", "tier:s", "tier:m", "tier:l", "tier-s", "tier-m", "tier-l")
                or (low_model.startswith("tier") and len(low_model.strip()) <= 8)
            ):
                t_str = low_model.replace("tier", "").replace(":", "").replace("-", "").strip().upper()
                if t_str in Tier.__members__:
                    target_tier = Tier[t_str]
                    tier_offerings = [o for o in self.registry.all() if o.tier == target_tier]
                    valid = [o for o in tier_offerings if self._passes(o, req, p)]
                    if valid:
                        scored = [self._score(o, req, p, prefix) for o in valid]
                        scored.sort(key=lambda c: c.score, reverse=True)
                        fallbacks = [
                            c
                            for c in self._auto_plan(req, p, limit=2, prefix=prefix)
                            if c.offering.key not in {c_s.offering.key for c_s in scored}
                        ]
                        return scored + fallbacks

            prov_key = low_model.split(":")[0]
            if prov_key in ("local", "ollama", "lmstudio", "llamacpp", "mlx") or (
                self.registry.find_by_provider(prov_key)
                and not self.registry.get(req.model)
            ):
                target_tier = None
                if ":" in low_model:
                    t_str = low_model.split(":")[1].upper()
                    if t_str in Tier.__members__:
                        target_tier = Tier[t_str]

                prov_offerings = self.registry.find_by_provider(prov_key)
                if target_tier:
                    prov_offerings = [
                        o for o in prov_offerings if o.tier == target_tier
                    ]

                valid = [o for o in prov_offerings if self._passes(o, req, p)]
                if valid:
                    scored = [self._score(o, req, p, prefix) for o in valid]
                    scored.sort(key=lambda c: c.score, reverse=True)
                    fallbacks = [
                        c
                        for c in self._auto_plan(req, p, limit=2, prefix=prefix)
                        if c.offering.key not in {c_s.offering.key for c_s in scored}
                    ]
                    return scored + fallbacks

            o = self.registry.find_by_model_id(req.model)
            if o is None:
                return []
            # An explicit model name must not resurrect a stale local
            # manifest entry.  Fall back through the normal plan when its
            # runtime did not pass the live availability check.
            if not o.usable:
                return self._auto_plan(req, p, limit, prefix=prefix)
            # The user's will — but not above the free_only policy: a
            # paid model cannot be called even explicitly, otherwise the
            # rule stops being a rule. Fall through to the auto-plan
            # (free only).
            if self.free_only and not o.free:
                return self._auto_plan(req, p, limit, prefix=prefix)
            # An explicitly requested model is the user's will. But the
            # local fallback is still appended next.
            plan = [self._score(o, req, p, prefix)]
            plan += [
                c
                for c in self._auto_plan(req, p, limit=2, prefix=prefix)
                if c.offering.is_local and c.offering.key != o.key
            ]
            return plan
        return self._auto_plan(
            req, p, limit, quality_floor=quality_floor, prefix=prefix
        )

    def _auto_plan(
        self,
        req: ChatRequest,
        p: RequestProfile,
        limit: int,
        min_tier: Tier | None = None,
        *,
        quality_floor: float = 0.0,
        prefix: str | None = None,
    ) -> list[Candidate]:
        def survivors(floor: float) -> list[Candidate]:
            return [
                self._score(o, req, p, prefix)
                for o in self.registry.all()
                if self._passes(o, req, p, floor)
                and (min_tier is None or TIER_ORDER[o.tier] >= TIER_ORDER[min_tier])
            ]

        cands = survivors(quality_floor)
        if not cands and quality_floor:
            # The quality floor is a preference, not a ban: if nobody
            # fits it, a weak candidate is better than a refusal.
            cands = survivors(0.0)
        cands.sort(key=lambda c: c.score, reverse=True)
        cloud_cands = _diversify([c for c in cands if not c.offering.is_local])
        local_cands = [c for c in cands if c.offering.is_local]

        # The local model is purely a fallback at the end of the ladder,
        # unless privacy_only was requested.
        if req.profile == "privacy_only":
            head = (local_cands + cloud_cands)[:limit]
        elif cloud_cands:
            head = cloud_cands[: max(1, limit - (1 if local_cands else 0))]
            if local_cands and not any(c.offering.is_local for c in head):
                head.append(local_cands[0])
        else:
            head = local_cands[:limit]
        return head

    def plan_escalated(
        self, req: ChatRequest, p: RequestProfile, min_tier: Tier, limit: int = 4
    ) -> list[Candidate]:
        """Plan for a cascade escalation: only models not below min_tier.
        Called by the verifier when a lower-tier answer failed its
        check."""
        return self._auto_plan(req, p, limit, min_tier=min_tier)

    def remember(
        self, session_id: str | None, key: str, prefix: str | None = None
    ) -> None:
        """Remember where a prefix is warmed. The dicts are size-bounded:
        the router lives for weeks and its memory is not infinite."""
        if session_id:
            if len(self._sticky) >= self.sticky_limit:
                self._sticky.clear()
            self._sticky[session_id] = key
        if prefix:
            if len(self._sticky_prefix) >= self.sticky_limit:
                self._sticky_prefix.clear()
            self._sticky_prefix[prefix] = key
