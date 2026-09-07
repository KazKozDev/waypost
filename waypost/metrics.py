"""Prometheus-format metrics.

A custom exposition implementation instead of prometheus_client — for one
reason: the router has a dozen metrics, while the dependency drags in
collectors, registries and its own error format. The text exposition
format is stable and takes thirty lines.

What actually needs to be visible (spec §5):
  TTFT per provider          — to catch degradation before a failure
  hit rate per cache level   — to see what L0 gives vs L2
  quota burn-down            — to see what time we hit the wall tonight
  cascade escalation share   — to know whether it pays off at all
"""
from __future__ import annotations

import threading
import time
from typing import Any, Callable, Iterable

# Latency histogram bounds: from a local model (hundreds of ms) to an
# overloaded free tier (tens of seconds).
LATENCY_BUCKETS_MS = (50, 100, 250, 500, 1000, 2000, 5000, 10_000, 30_000)


def _labels(pairs: dict[str, Any]) -> str:
    if not pairs:
        return ""
    inner = ",".join(f'{k}="{_escape(str(v))}"' for k, v in sorted(pairs.items()))
    return "{" + inner + "}"


def _escape(v: str) -> str:
    return v.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


class Metrics:
    def __init__(self):
        self._lock = threading.Lock()
        self._counters: dict[tuple[str, tuple], float] = {}
        self._hist: dict[tuple[str, tuple], list[float]] = {}
        self._gauge_sources: list[Callable[[], Iterable[tuple]]] = []
        self.started_at = time.time()

    # ------------------------------------------------------------- write
    def inc(self, name: str, value: float = 1.0, **labels) -> None:
        key = (name, tuple(sorted(labels.items())))
        with self._lock:
            self._counters[key] = self._counters.get(key, 0.0) + value

    def observe(self, name: str, value_ms: float, **labels) -> None:
        key = (name, tuple(sorted(labels.items())))
        with self._lock:
            self._hist.setdefault(key, []).append(value_ms)

    def register_gauges(self, source: Callable[[], Iterable[tuple]]) -> None:
        """A gauge source: a function returning (name, labels, value).

        The ledger and breaker know their own state; mirroring it in
        counters is a guaranteed way to drift from reality.
        """
        self._gauge_sources.append(source)

    # ------------------------------------------------------------- output
    def render(self) -> str:
        lines: list[str] = []
        with self._lock:
            counters = dict(self._counters)
            hist = {k: list(v) for k, v in self._hist.items()}

        seen: set[str] = set()
        for (name, labels), value in sorted(counters.items()):
            if name not in seen:
                lines.append(f"# TYPE {name} counter")
                seen.add(name)
            lines.append(f"{name}{_labels(dict(labels))} {value:g}")

        for (name, labels), values in sorted(hist.items()):
            if name not in seen:
                lines.append(f"# TYPE {name} histogram")
                seen.add(name)
            base = dict(labels)
            cumulative = 0
            ordered = sorted(values)
            for bound in LATENCY_BUCKETS_MS:
                cumulative = sum(1 for v in ordered if v <= bound)
                lines.append(
                    f"{name}_bucket{_labels({**base, 'le': bound})} {cumulative}"
                )
            lines.append(
                f"{name}_bucket{_labels({**base, 'le': '+Inf'})} {len(ordered)}"
            )
            lines.append(f"{name}_sum{_labels(base)} {sum(ordered):g}")
            lines.append(f"{name}_count{_labels(base)} {len(ordered)}")

        for source in self._gauge_sources:
            try:
                for name, labels, value in source():
                    if name not in seen:
                        lines.append(f"# TYPE {name} gauge")
                        seen.add(name)
                    lines.append(f"{name}{_labels(labels)} {value:g}")
            except Exception:  # noqa: BLE001
                continue

        lines.append("# TYPE waypost_uptime_seconds gauge")
        lines.append(f"waypost_uptime_seconds {time.time() - self.started_at:g}")
        return "\n".join(lines) + "\n"


BREAKER_STATES = {"closed": 0, "half_open": 1, "open": 2}


LIFECYCLE_STATES = {"active": 0, "shadow": 1, "candidate": 2, "quarantine": 3}


def gauges_from(
    ledger,
    breaker,
    caches: dict[str, Any],
    *,
    latency=None,
    rate_governor=None,
    registry=None,
    inflight: dict[str, int] | None = None,
) -> Callable:
    """Wires metrics to the router's live state."""

    def source():
        if latency is not None:
            for key, st in latency.snapshot().items():
                # The ratio, not the absolute: 6 s is awful for a 1B model
                # and fine for a 70B one. >2.5 means silent degradation.
                yield ("waypost_latency_ema_ms", {"offering": key}, st["ema_ms"])
                yield ("waypost_latency_drift_ratio", {"offering": key}, st["ratio"])
        if rate_governor is not None:
            for slot, st in rate_governor.snapshot().items():
                yield ("waypost_learned_rpm", {"slot": slot}, st["effective_rpm"])
        if inflight:
            for key, n in inflight.items():
                yield ("waypost_inflight", {"offering": key}, n)
        if registry is not None:
            counts: dict[str, int] = {}
            for o in registry.all():
                counts[o.lifecycle] = counts.get(o.lifecycle, 0) + 1
            for state in LIFECYCLE_STATES:
                yield ("waypost_pool_size", {"state": state}, counts.get(state, 0))
        yield from _core()

    def _core():
        for offering, buckets in ledger.snapshot().items():
            for bucket, remaining in buckets.items():
                yield (
                    "waypost_quota_remaining",
                    {"offering": offering, "bucket": bucket},
                    remaining,
                )
        for provider, state in breaker.snapshot().items():
            yield (
                "waypost_breaker_state",
                {"provider": provider},
                BREAKER_STATES.get(state, 0),
            )
        for level, cache in caches.items():
            if cache is None:
                continue
            yield ("waypost_cache_hit_rate", {"level": level}, cache.hit_rate())

    return source
