"""Self-update: reloading the pool without dropping traffic.

Discovery finds new free models, probing measures them, models get
retired, limits change. Applying any of that used to mean either editing
the manifest and restarting — which throws away the ledger, the bandit
and every breaker state — or mutating the live registry entry by entry,
where a half-applied change is a pool that briefly makes no sense.

The cycle here is the usual one for configuration that must not be
trusted blindly:

    detect → validate → stage → swap → observe → roll back if worse

Three parts of it are worth stating outright.

*Invariants before the swap, not after.* A rebuilt pool is refused if it
would leave the router without a local fallback, without a tier, or
suddenly half the size — the classic shape of a provider returning a
truncated /models page. Refusing to reload is always safe; reloading
into an unusable pool is not.

*The swap is one reference rebind.* Requests in flight hold their
Offering objects, so they finish against the pool they were planned on.

*Observation, because invariants only catch what we thought of.* A pool
that passes every check can still route worse. So the reloader watches
the error rate for a window afterwards and puts the previous pool back
if it regressed, keeping the last few versions for exactly that.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .registry import Offering, Registry

log = logging.getLogger("waypost.selfupdate")

# How far the pool may shrink in one reload before it is treated as a bad
# read rather than as news.
MIN_POOL_RATIO = 0.5
# Watch this long after a swap before deciding it was fine.
OBSERVE_WINDOW_S = 600.0
# Fewer attempts than this in the window and there is nothing to judge.
OBSERVE_MIN_ATTEMPTS = 20
# Roll back when the success rate falls to this fraction of the baseline.
REGRESSION_RATIO = 0.7


@dataclass
class SwapResult:
    ok: bool
    version: int
    reason: str = ""
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)


def check_invariants(
    fresh: list[Offering], current: list[Offering], *, free_only: bool = True
) -> tuple[bool, str]:
    """Would this pool still be able to serve traffic?

    Checked against the *usable* subset on both sides: a pool of two
    hundred offerings none of which has a key is an empty pool.
    """
    usable_next = [o for o in fresh if o.usable]
    usable_now = [o for o in current if o.usable]

    if not usable_next:
        return False, "no usable offerings"

    # The local model is the rung that always answers. A pool without it
    # can only fail upward, into a 502.
    if usable_now and any(o.is_local for o in usable_now):
        if not any(o.is_local for o in usable_next):
            return False, "lost the local fallback"

    # A provider returning a truncated /models page must not be able to
    # halve the pool in one cycle.
    if usable_now and len(usable_next) < len(usable_now) * MIN_POOL_RATIO:
        return (
            False,
            f"pool shrank {len(usable_now)} → {len(usable_next)} "
            f"(below {MIN_POOL_RATIO:.0%})",
        )

    # Every tier that could be served before must still be servable, or
    # classification will route requests at a tier nothing can answer.
    tiers_now = {o.tier for o in usable_now}
    tiers_next = {o.tier for o in usable_next}
    if missing := tiers_now - tiers_next:
        return False, f"lost tier(s) {','.join(sorted(t.value for t in missing))}"

    # Capabilities are the same argument: losing every tool-calling model
    # turns a working request into a hard filter miss.
    for cap in ("tools", "vision"):
        had = any(cap in {c.value for c in o.caps} for o in usable_now)
        has = any(cap in {c.value for c in o.caps} for o in usable_next)
        if had and not has:
            return False, f"lost every {cap}-capable offering"

    if free_only and (paid := [o for o in usable_next if not o.free]):
        return False, f"free_only violated by {paid[0].key}"

    return True, ""


class RegistryReloader:
    """Owns the swap, the version history and the rollback watch."""

    def __init__(
        self,
        registry: Registry,
        *,
        build: Callable[[], list[Offering]],
        success_rate: Callable[[], tuple[int, float]],
        free_only: bool = True,
        observe_window_s: float = OBSERVE_WINDOW_S,
        history_depth: int = 3,
        on_swap: Callable[[list[Offering]], None] | None = None,
    ):
        self.registry = registry
        self.build = build
        self.success_rate = success_rate
        self.free_only = free_only
        self.observe_window_s = observe_window_s
        self.history_depth = history_depth
        # Called with the new pool right after a swap: the ledger has to
        # learn about buckets for keys that did not exist a moment ago.
        self.on_swap = on_swap
        self._history: list[tuple[int, list[Offering]]] = []
        self._events: list[dict[str, Any]] = []
        self._watch: asyncio.Task | None = None

    # ------------------------------------------------------------ events
    def _emit(self, kind: str, **fields: Any) -> None:
        event = {"kind": kind, "ts": time.time(), **fields}
        self._events.append(event)
        del self._events[:-100]
        log.info("registry.%s %s", kind, fields)

    def events(self, limit: int = 20) -> list[dict[str, Any]]:
        return self._events[-limit:]

    # -------------------------------------------------------------- swap
    def reload(self, *, observe: bool = True) -> SwapResult:
        current = self.registry.all()
        fresh = self.registry.carry_over(self.build())

        ok, reason = check_invariants(fresh, current, free_only=self.free_only)
        if not ok:
            self._emit("rejected", reason=reason, version=self.registry.version)
            return SwapResult(False, self.registry.version, reason)

        before = {o.key for o in current}
        after = {o.key for o in fresh}
        if before == after and len(current) == len(fresh):
            # Nothing changed. Bumping the version anyway would make the
            # rollback history meaningless.
            return SwapResult(True, self.registry.version, "no change")

        self._history.append((self.registry.version, current))
        del self._history[: -self.history_depth]

        if self.on_swap is not None:
            self.on_swap(fresh)  # buckets first, then traffic
        version = self.registry.swap(fresh)

        result = SwapResult(
            True,
            version,
            added=sorted(after - before),
            removed=sorted(before - after),
        )
        self._emit(
            "swap",
            version=version,
            added=len(result.added),
            removed=len(result.removed),
            size=len(fresh),
        )
        if observe:
            self._start_watch(version)
        return result

    def rollback(self, reason: str = "manual") -> SwapResult:
        if not self._history:
            return SwapResult(False, self.registry.version, "no previous version")
        prev_version, prev = self._history.pop()
        if self.on_swap is not None:
            self.on_swap(prev)
        version = self.registry.swap(prev)
        self._emit(
            "rollback", version=version, restored=prev_version, reason=reason
        )
        return SwapResult(True, version, reason)

    # ----------------------------------------------------------- observe
    def _start_watch(self, version: int) -> None:
        if self._watch is not None and not self._watch.done():
            self._watch.cancel()
        try:
            self._watch = asyncio.create_task(self._observe(version))
        except RuntimeError:
            self._watch = None  # no running loop (CLI, tests)

    async def _observe(self, version: int) -> None:
        """Invariants only catch what we thought to check. This catches
        the rest: if the new pool routes measurably worse, put the old one
        back and say which metric said so."""
        baseline_n, baseline_rate = self.success_rate()
        await asyncio.sleep(self.observe_window_s)
        if self.registry.version != version:
            return  # something else already swapped; not ours to judge
        n, rate = self.success_rate()
        observed = max(0, n - baseline_n)
        if observed < OBSERVE_MIN_ATTEMPTS:
            self._emit("observed", version=version, verdict="too few attempts")
            return
        if baseline_rate and rate < baseline_rate * REGRESSION_RATIO:
            self.rollback(
                f"success rate {rate:.0%} vs baseline {baseline_rate:.0%} "
                f"over {observed} attempts"
            )
            return
        self._emit(
            "observed", version=version, verdict="ok", rate=round(rate, 3), n=observed
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "version": self.registry.version,
            "history": [v for v, _ in self._history],
            "watching": bool(self._watch and not self._watch.done()),
            "events": self.events(10),
        }


class ProbeTriggers:
    """Out-of-band probe requests.

    The scheduled probe runs once a day, which is right for keeping
    measurements fresh and useless for reacting: a provider that just
    started failing, or one sitting in half-open that no plan happens to
    reach, waits a full cycle for anyone to check on it. Producers push a
    key here, the health job drains it.
    """

    def __init__(self, max_pending: int = 32):
        self.max_pending = max_pending
        self._pending: dict[str, str] = {}  # key → reason

    def request(self, key: str, reason: str) -> None:
        if key in self._pending or len(self._pending) >= self.max_pending:
            return
        self._pending[key] = reason
        log.info("probe requested for %s (%s)", key, reason)

    def drain(self) -> dict[str, str]:
        pending, self._pending = self._pending, {}
        return pending

    def pending(self) -> dict[str, str]:
        return dict(self._pending)


def collect_triggers(
    triggers: ProbeTriggers,
    registry: Registry,
    breaker: Any,
    latency: Any,
) -> None:
    """Scan live state for offerings worth an immediate probe."""
    states = breaker.snapshot()
    for o in registry.all():
        if o.is_local:
            continue
        # A half-open provider is waiting for one call to decide its fate.
        # If no plan reaches it, that call has to come from the probe.
        if states.get(o.provider) == "half_open":
            triggers.request(o.key, "breaker half_open")
        # Degrading while still answering 200: the probe re-measures TTFT
        # and re-checks the capabilities that may have silently changed.
        elif latency.drifted(o.key):
            triggers.request(o.key, "latency drift")
