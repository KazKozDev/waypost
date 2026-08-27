"""Per-provider circuit breaker.

Without it the router would stubbornly hammer a down endpoint, spending
latency budget on every request.

closed → open  : N failures within a window
open → half_open: cooldown expired
half_open      : a single probe request is allowed; success → closed,
                 failure → open with doubled cooldown (up to a cap)
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
    half_open_inflight: bool = False


class CircuitBreaker:
    def __init__(
        self,
        threshold: int = 4,
        window_s: float = 60.0,
        base_cooldown_s: float = 30.0,
        max_cooldown_s: float = 600.0,
    ):
        self.threshold = threshold
        self.window_s = window_s
        self.base_cooldown_s = base_cooldown_s
        self.max_cooldown_s = max_cooldown_s
        self._states: dict[str, _State] = {}
        self._lock = threading.Lock()

    def _get(self, provider: str) -> _State:
        return self._states.setdefault(
            provider, _State(cooldown_s=self.base_cooldown_s)
        )

    def state(self, provider: str) -> str:
        with self._lock:
            s = self._get(provider)
            if s.opened_at == 0.0:
                return "closed"
            if time.time() - s.opened_at >= s.cooldown_s:
                return "half_open"
            return "open"

    def allows(self, provider: str) -> bool:
        with self._lock:
            s = self._get(provider)
            if s.opened_at == 0.0:
                return True
            if time.time() - s.opened_at < s.cooldown_s:
                return False
            if s.half_open_inflight:
                return False
            s.half_open_inflight = True  # allow exactly one probe
            return True

    def on_success(self, provider: str) -> None:
        with self._lock:
            s = self._get(provider)
            s.failures.clear()
            s.opened_at = 0.0
            s.cooldown_s = self.base_cooldown_s
            s.half_open_inflight = False

    def on_failure(self, provider: str) -> None:
        with self._lock:
            s = self._get(provider)
            now = time.time()
            if s.half_open_inflight:
                s.half_open_inflight = False
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
