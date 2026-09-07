"""Per-provider circuit breaker.

Without it the router would stubbornly hammer a down endpoint, spending
latency budget on every request.

closed → open  : N failures within a window
open → half_open: cooldown expired
half_open      : a single probe request is allowed; success → closed,
                 failure → open with doubled cooldown (up to a cap)

Admission and probing are deliberately split into two methods:

    can_admit()     — pure, side-effect free. The router calls it for every
                      candidate of every request while it is only *planning*.
    acquire_probe() — takes the single half-open token, and is called at the
                      moment of the actual provider call.

Merging the two loses the token whenever a planned candidate is never
called (it ranked below the plan limit, an earlier candidate answered,
the request was cancelled) — and since only a real call produces
on_success/on_failure, nothing ever gives it back: the provider stays
excluded for the lifetime of the process.  release_probe() in a finally
block and a TTL on the token are the two belts that keep that from
happening.

Rate limiting does NOT belong here.  A 429 means "we are knocking too
fast", not "the endpoint is sick"; it is handled by the ledger (a hard
block until Retry-After) and by RateGovernor (learning the real limit).
Feeding it into the failure counter opens the breaker on a provider that
is perfectly healthy.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass
class _State:
    failures: list[float] = field(default_factory=list)
    opened_at: float = 0.0
    cooldown_s: float = 30.0
    probe_started_at: float = 0.0  # 0.0 — no probe in flight


@dataclass(frozen=True)
class ProbeToken:
    """Handle for a half-open probe. Returned by acquire_probe(), and must
    reach release_probe() on every path that does not resolve it."""

    provider: str
    started_at: float


class CircuitBreaker:
    def __init__(
        self,
        threshold: int = 4,
        window_s: float = 60.0,
        base_cooldown_s: float = 30.0,
        max_cooldown_s: float = 600.0,
        probe_ttl_s: float = 300.0,
    ):
        self.threshold = threshold
        self.window_s = window_s
        self.base_cooldown_s = base_cooldown_s
        self.max_cooldown_s = max_cooldown_s
        # A probe that neither succeeded nor failed within this window is
        # assumed lost (a crashed task, a cancelled request). The last
        # line of defence against the token leak described above.
        self.probe_ttl_s = probe_ttl_s
        self._states: dict[str, _State] = {}
        self._lock = threading.Lock()

    def _get(self, provider: str) -> _State:
        return self._states.setdefault(
            provider, _State(cooldown_s=self.base_cooldown_s)
        )

    def _probe_in_flight(self, s: _State, now: float) -> bool:
        if s.probe_started_at == 0.0:
            return False
        if now - s.probe_started_at >= self.probe_ttl_s:
            s.probe_started_at = 0.0  # stale token — reclaim it
            return False
        return True

    def state(self, provider: str) -> str:
        with self._lock:
            s = self._get(provider)
            if s.opened_at == 0.0:
                return "closed"
            if time.time() - s.opened_at >= s.cooldown_s:
                return "half_open"
            return "open"

    # ---------------------------------------------------------- admission
    def can_admit(self, provider: str) -> bool:
        """Pure check: may this provider be considered at all?

        Called during planning, possibly many times per request. It must
        not mutate anything the executor depends on.
        """
        with self._lock:
            s = self._get(provider)
            now = time.time()
            if s.opened_at == 0.0:
                return True
            if now - s.opened_at < s.cooldown_s:
                return False
            return not self._probe_in_flight(s, now)

    def allows(self, provider: str) -> bool:
        """Backwards-compatible alias for the pure check."""
        return self.can_admit(provider)

    def acquire_probe(self, provider: str) -> ProbeToken | None:
        """Take the half-open token, at the moment of the real call.

        Returns None when the breaker is closed (no token needed) or when
        another probe already holds it — in the latter case the caller has
        lost the race and must skip the candidate, which can_admit() could
        not know at planning time.
        """
        with self._lock:
            s = self._get(provider)
            now = time.time()
            if s.opened_at == 0.0:
                return None  # closed: ordinary traffic, nothing to hold
            if now - s.opened_at < s.cooldown_s:
                return ProbeToken(provider, -1.0)  # sentinel: caller must skip
            if self._probe_in_flight(s, now):
                return ProbeToken(provider, -1.0)
            s.probe_started_at = now
            return ProbeToken(provider, now)

    @staticmethod
    def probe_lost_race(token: ProbeToken | None) -> bool:
        """True when acquire_probe() refused: the candidate must be skipped."""
        return token is not None and token.started_at < 0.0

    def release_probe(self, token: ProbeToken | None) -> None:
        """Give the token back without a verdict.

        Called from a finally block: if the call resolved, on_success /
        on_failure already cleared the slot and this is a no-op.
        """
        if token is None or token.started_at < 0.0:
            return
        with self._lock:
            s = self._get(token.provider)
            if s.probe_started_at == token.started_at:
                s.probe_started_at = 0.0

    # ------------------------------------------------------------ verdicts
    def on_success(self, provider: str) -> None:
        with self._lock:
            s = self._get(provider)
            s.failures.clear()
            s.opened_at = 0.0
            s.cooldown_s = self.base_cooldown_s
            s.probe_started_at = 0.0

    def on_failure(self, provider: str) -> None:
        """Health failures only: 5xx, timeouts, connection errors.

        A 429 must never reach this method — see the module docstring.
        """
        with self._lock:
            s = self._get(provider)
            now = time.time()
            if self._probe_in_flight(s, now):
                # The half-open probe failed: straight back to open with a
                # doubled cooldown.
                s.probe_started_at = 0.0
                s.opened_at = now
                s.cooldown_s = min(self.max_cooldown_s, s.cooldown_s * 2)
                return
            s.failures = [t for t in s.failures if now - t < self.window_s]
            s.failures.append(now)
            if len(s.failures) >= self.threshold:
                s.opened_at = now
                s.failures.clear()

    def snapshot(self) -> dict[str, str]:
        return {p: self.state(p) for p in list(self._states)}
