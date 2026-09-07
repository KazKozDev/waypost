"""Rate-limit governor: learning the real limit instead of trusting the docs.

A 429 is not a health signal. The endpoint is alive and willing — we are
knocking faster than the free tier allows. Feeding it into the circuit
breaker opens a perfectly healthy provider; feeding it nowhere means the
router keeps rediscovering the same limit on every burst.

So 429 flows here. The governor does one thing: it multiplies the
declared rpm limit by a factor it learns from refusals, and writes the
result straight into the ledger's bucket. Reusing the ledger's accounting
rather than keeping a second counter matters — two admission counters
that disagree are worse than one that is slightly wrong.

  ≥2 refusals within burst_window_s  →  factor ×= 0.75  (floor 0.2)
  no refusal for calm_s              →  factor ×= 1.10  (cap 1.0)

The declared limit is the ceiling: a provider that documents 30 rpm is
never given 40, however calm it has been.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from .ledger import Ledger
from .registry import Offering

log = logging.getLogger("waypost.ratelimit")


@dataclass
class _Learned:
    declared_rpm: int
    factor: float = 1.0
    events: list[float] = field(default_factory=list)
    last_429: float = 0.0
    last_grow: float = 0.0


class RateGovernor:
    def __init__(
        self,
        ledger: Ledger,
        *,
        shrink: float = 0.75,
        grow: float = 1.10,
        floor: float = 0.2,
        burst_window_s: float = 60.0,
        calm_s: float = 300.0,
        burst_threshold: int = 2,
        shared: Any | None = None,
    ):
        self.ledger = ledger
        # The learned limit is a property of the provider's account, not
        # of one replica's traffic. Shared, every replica benefits from
        # the refusal any one of them collected.
        self.shared = shared
        self.shrink = shrink
        self.grow = grow
        self.floor = floor
        self.burst_window_s = burst_window_s
        self.calm_s = calm_s
        self.burst_threshold = burst_threshold
        self._learned: dict[str, _Learned] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _slot(o: Offering, key_index: int) -> str:
        return f"{o.key}#{key_index}"

    def _apply(self, o: Offering, key_index: int, st: _Learned) -> None:
        limit = max(1, int(round(st.declared_rpm * st.factor)))
        self.ledger.set_limit(o, key_index, "rpm", limit)
        if self.shared is not None:
            self.shared.set_factor(self._slot(o, key_index), st.factor)

    def sync_from_shared(self, offerings: list[Offering]) -> int:
        """Adopt limits other replicas learned. Called by the control
        plane, not on the hot path."""
        if self.shared is None:
            return 0
        updated = 0
        for o in offerings:
            if not o.limit_rpm:
                continue
            for idx in range(max(1, o.key_count)):
                slot = self._slot(o, idx)
                factor = self.shared.get_factor(slot)
                if factor is None or factor >= 1.0:
                    continue
                with self._lock:
                    st = self._learned.setdefault(
                        slot, _Learned(declared_rpm=o.limit_rpm)
                    )
                    if abs(st.factor - factor) < 1e-6:
                        continue
                    st.factor = factor
                    self._apply(o, idx, st)
                updated += 1
        return updated

    # --------------------------------------------------------------- api
    def on_rate_limit(self, o: Offering, key_index: int = 0) -> None:
        """A 429 arrived. Two of them in a minute mean the declared limit
        is a fiction — shrink what we admit."""
        declared = o.limit_rpm
        if not declared:
            return
        slot = self._slot(o, key_index)
        with self._lock:
            st = self._learned.setdefault(slot, _Learned(declared_rpm=declared))
            now = time.time()
            st.last_429 = now
            st.events = [t for t in st.events if now - t < self.burst_window_s]
            st.events.append(now)
            if len(st.events) < self.burst_threshold:
                return
            before = st.factor
            st.factor = max(self.floor, st.factor * self.shrink)
            st.events.clear()
            if st.factor != before:
                log.info(
                    "rate: %s learned limit %d → %d rpm (factor %.2f)",
                    slot,
                    int(declared * before),
                    int(declared * st.factor),
                    st.factor,
                )
            self._apply(o, key_index, st)

    def on_success(self, o: Offering, key_index: int = 0) -> None:
        """Calm for long enough — try to give the quota back, slowly."""
        if not o.limit_rpm:
            return
        slot = self._slot(o, key_index)
        with self._lock:
            st = self._learned.get(slot)
            if st is None or st.factor >= 1.0:
                return
            now = time.time()
            if now - st.last_429 < self.calm_s:
                return
            if now - st.last_grow < self.calm_s:
                return
            st.last_grow = now
            st.factor = min(1.0, st.factor * self.grow)
            self._apply(o, key_index, st)
            log.info("rate: %s recovering → factor %.2f", slot, st.factor)

    def factor(self, o: Offering, key_index: int = 0) -> float:
        st = self._learned.get(self._slot(o, key_index))
        return st.factor if st else 1.0

    def snapshot(self) -> dict[str, dict[str, float]]:
        with self._lock:
            return {
                slot: {
                    "declared_rpm": st.declared_rpm,
                    "effective_rpm": max(1, int(round(st.declared_rpm * st.factor))),
                    "factor": round(st.factor, 3),
                    "last_429": int(st.last_429) if st.last_429 else 0,
                }
                for slot, st in self._learned.items()
                if st.factor < 1.0
            }
