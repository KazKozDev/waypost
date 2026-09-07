"""A/B for routing policy.

Every knob in this router was set by an argument. Some of those arguments
were good; none of them were measured. Is stochastic selection actually
better than argmax here? Does the neighbourhood earn its place? Is a
half-life of fourteen days right? The honest answer to all three has been
"it sounded right", and the way to replace that with a number is to run
both and compare.

Three properties this needs and a naive coin flip does not have.

*Stable assignment.* A session must stay in one arm. Flipping a
conversation between policies mid-way measures neither of them, and the
router's own stickiness — warmed prefixes, session affinity — would be
destroyed by the switching rather than by either policy.

*A hash, not a counter.* Assignment is a hash of the unit, so it survives
restarts, splits identically across replicas, and can be recomputed after
the fact from the log rather than having to be stored.

*Outcomes joined afterwards.* Nothing is recorded at assignment time
beyond the arm name. What the arm cost is read from the attempt log and
the feedback table, which are written anyway — so an experiment adds a
field, not a pipeline.

The comparison is deliberately blunt: success rate, p95 latency, attempts
per request, escalation rate. No significance testing, because with a few
hundred requests a p-value would be theatre; the report says how many
observations each arm has and leaves the judgement where it belongs.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("waypost.experiment")


@dataclass
class Arm:
    """One policy variant: a name and the settings it overrides."""

    name: str
    overrides: dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0


@dataclass
class Experiment:
    name: str
    arms: list[Arm]
    # What assignment is keyed on. "session" keeps a conversation whole,
    # which is what you want unless the thing under test is per-request.
    unit: str = "session"
    enabled: bool = True

    def assign(self, key: str | None) -> Arm:
        """Deterministic arm for this unit.

        Without a key — no session id, a one-off request — the control arm
        is used rather than a random one: an unassignable request should
        not quietly become data for an arm it was never in.
        """
        if not self.enabled or not self.arms:
            return Arm("control")
        if not key:
            return self.arms[0]
        total = sum(max(0.0, a.weight) for a in self.arms) or 1.0
        digest = hashlib.sha256(f"{self.name}:{key}".encode()).digest()
        # 4 bytes is plenty of resolution for a handful of arms.
        point = int.from_bytes(digest[:4], "big") / 0xFFFFFFFF * total
        upto = 0.0
        for arm in self.arms:
            upto += max(0.0, arm.weight)
            if point <= upto:
                return arm
        return self.arms[-1]


class ExperimentRegistry:
    """The experiments in force, and how the router should read them."""

    def __init__(self, experiments: list[Experiment] | None = None):
        self.experiments = {e.name: e for e in (experiments or [])}

    def add(self, exp: Experiment) -> None:
        self.experiments[exp.name] = exp

    def assignments(self, key: str | None) -> dict[str, str]:
        return {
            name: exp.assign(key).name
            for name, exp in self.experiments.items()
            if exp.enabled
        }

    def overrides(self, key: str | None) -> dict[str, Any]:
        """Merged settings for this unit.

        Two experiments touching the same knob is a mistake, not a
        feature: the second would silently win and both reports would be
        wrong. It is logged loudly rather than resolved.
        """
        merged: dict[str, Any] = {}
        for name, exp in self.experiments.items():
            if not exp.enabled:
                continue
            for field_name, value in exp.assign(key).overrides.items():
                if field_name in merged and merged[field_name] != value:
                    log.error(
                        "experiments collide on %s: %s wants %r, another wants %r",
                        field_name,
                        name,
                        value,
                        merged[field_name],
                    )
                merged[field_name] = value
        return merged

    def snapshot(self) -> dict[str, Any]:
        return {
            name: {
                "enabled": exp.enabled,
                "unit": exp.unit,
                "arms": [
                    {"name": a.name, "weight": a.weight, "overrides": a.overrides}
                    for a in exp.arms
                ],
            }
            for name, exp in self.experiments.items()
        }


def compare(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate outcomes per arm.

    Each row is one final attempt, carrying the arm it belonged to. What
    comes back is a small table, plus the sample size — which is the
    number to read first. Two arms differing by three points over ninety
    requests differ by nothing.
    """
    by_arm: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        arm = r.get("arm") or "control"
        by_arm.setdefault(arm, []).append(r)

    out: dict[str, Any] = {}
    for arm, items in sorted(by_arm.items()):
        n = len(items)
        latencies = sorted(float(i.get("latency_ms") or 0) for i in items)
        ok = sum(1 for i in items if i.get("outcome") == "pass")
        attempts = [int(i.get("attempt_no") or 1) for i in items]
        out[arm] = {
            "n": n,
            "success_rate": round(ok / n, 3) if n else 0.0,
            "p50_latency_ms": int(latencies[n // 2]) if n else 0,
            "p95_latency_ms": int(latencies[min(n - 1, int(n * 0.95))]) if n else 0,
            "mean_attempts": round(sum(attempts) / n, 2) if n else 0.0,
            "escalation_rate": round(
                sum(1 for a in attempts if a > 1) / n, 3
            ) if n else 0.0,
        }
    if len(out) >= 2:
        arms = list(out)
        base, *rest = arms
        out["_verdict"] = {
            "baseline": base,
            "note": "sample sizes first: a few hundred requests decide very little",
            "deltas": {
                arm: {
                    "success_rate": round(
                        out[arm]["success_rate"] - out[base]["success_rate"], 3
                    ),
                    "p95_latency_ms": out[arm]["p95_latency_ms"]
                    - out[base]["p95_latency_ms"],
                }
                for arm in rest
            },
        }
    return out
