"""Online latency EMA — the missing feedback loop.

Latency was measured on every answer, written to telemetry and then
ignored: the router scored candidates from ``ttft_p50_ms``, a manifest
constant refreshed only by the probe job (which is off by default). A
provider degrading from 8 s to 25 s while still returning 200 OK was
therefore invisible — its score only went *up*, because the bandit
counted every 200 as a success.

Two decisions worth stating:

*Asymmetric smoothing.* Degradation is taken seriously at once
(alpha_up = 0.3), recovery is trusted slowly (alpha_down = 0.1). One
lucky fast answer must not put a flapping provider back at the top of
the plan.

*A baseline.* The absolute number says little — 6 s is awful for a 1B
model and fine for a 70B one. What matters is the ratio to what this
offering normally does, so ``drifted()`` compares against a slow-moving
baseline instead of a global threshold.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass


@dataclass
class _Stat:
    ema_ms: float
    baseline_ms: float
    samples: int = 0
    last_ts: float = 0.0


class LatencyTracker:
    def __init__(
        self,
        *,
        alpha_up: float = 0.3,
        alpha_down: float = 0.1,
        baseline_alpha: float = 0.02,
        drift_ratio: float = 2.5,
        min_samples: int = 5,
    ):
        self.alpha_up = alpha_up
        self.alpha_down = alpha_down
        self.baseline_alpha = baseline_alpha
        self.drift_ratio = drift_ratio
        self.min_samples = min_samples
        self._stats: dict[str, _Stat] = {}
        self._lock = threading.Lock()

    def seed(self, key: str, ms: float) -> None:
        """Prime from the manifest / last probe, so a cold offering is not
        scored as if it had zero latency."""
        if ms <= 0:
            return
        with self._lock:
            if key not in self._stats:
                self._stats[key] = _Stat(ema_ms=ms, baseline_ms=ms)

    def observe(self, key: str, ms: float) -> float:
        """Record one measured latency. A timeout counts: the budget that
        was burned is exactly the signal the score needs."""
        if ms <= 0:
            return self.get(key)
        with self._lock:
            st = self._stats.get(key)
            if st is None:
                st = _Stat(ema_ms=ms, baseline_ms=ms)
                self._stats[key] = st
            else:
                a = self.alpha_up if ms > st.ema_ms else self.alpha_down
                st.ema_ms = a * ms + (1.0 - a) * st.ema_ms
                st.baseline_ms = (
                    self.baseline_alpha * ms
                    + (1.0 - self.baseline_alpha) * st.baseline_ms
                )
            st.samples += 1
            st.last_ts = time.time()
            return st.ema_ms

    def get(self, key: str, default_ms: float = 0.0) -> float:
        st = self._stats.get(key)
        return st.ema_ms if st else default_ms

    def drifted(self, key: str) -> bool:
        """Is this offering markedly slower than its own normal?

        Used to drop prefix stickiness: staying pinned to a warmed prefix
        is only worth it while the provider is still fast.
        """
        st = self._stats.get(key)
        if st is None or st.samples < self.min_samples or st.baseline_ms <= 0:
            return False
        return st.ema_ms / st.baseline_ms >= self.drift_ratio

    def snapshot(self) -> dict[str, dict[str, float]]:
        with self._lock:
            return {
                key: {
                    "ema_ms": round(st.ema_ms, 1),
                    "baseline_ms": round(st.baseline_ms, 1),
                    "ratio": round(st.ema_ms / st.baseline_ms, 2)
                    if st.baseline_ms
                    else 1.0,
                    "samples": st.samples,
                    "drifted": self.drifted(key),
                }
                for key, st in self._stats.items()
            }
