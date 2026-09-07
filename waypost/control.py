"""Control plane: background jobs.

The split into data plane and control plane is not architectural
decoration but a latency requirement. The hot path must fit in single
milliseconds, so everything that goes over the network to refresh
knowledge about providers is moved to the background:

    discovery → probe → health → purge
        ↓         ↓        ↓        ↓
     registry  registry  registry  disk

Jobs are independent: a failed discovery must not take down probe, and
neither must take down the server. So each is wrapped in its own try and
logs the error instead of raising it upward.
"""
from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

log = logging.getLogger("waypost.control")


@dataclass
class Job:
    name: str
    fn: Callable[[], Awaitable[Any]]
    interval_s: float
    initial_delay_s: float = 0.0
    enabled: bool = True
    runs: int = 0
    failures: int = 0
    last_run: float = 0.0
    last_error: str | None = None
    last_result: Any = field(default=None, repr=False)


class ControlPlane:
    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._tasks: list[asyncio.Task] = []

    def add(
        self,
        name: str,
        fn: Callable[[], Awaitable[Any]],
        interval_s: float,
        *,
        initial_delay_s: float = 0.0,
        enabled: bool = True,
    ) -> None:
        self._jobs[name] = Job(name, fn, interval_s, initial_delay_s, enabled)

    def start(self) -> None:
        for job in self._jobs.values():
            if job.enabled:
                self._tasks.append(asyncio.create_task(self._loop(job)))

    async def stop(self) -> None:
        for t in self._tasks:
            t.cancel()
        for t in self._tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        self._tasks.clear()

    async def run_once(self, name: str) -> Any:
        """Manual run of a job — from the CLI or from an endpoint."""
        job = self._jobs.get(name)
        if job is None:
            raise KeyError(name)
        return await self._run(job)

    async def _run(self, job: Job) -> Any:
        started = time.time()
        try:
            result = await job.fn()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            job.failures += 1
            job.last_error = f"{type(exc).__name__}: {exc}"[:300]
            log.warning("job %s failed: %s", job.name, job.last_error)
            return None
        else:
            job.runs += 1
            job.last_error = None
            job.last_result = result
            log.debug("job %s done in %.1fs", job.name, time.time() - started)
            return result
        finally:
            job.last_run = started

    def _next_delay(self, job: Job) -> float:
        """+/-20% jitter. Without it every instance (and every job that
        shares an interval) wakes at the same instant and probes the same
        providers together — a self-inflicted burst on exactly the free
        tiers the router is trying to conserve."""
        return job.interval_s * (0.8 + 0.4 * random.random())

    async def _loop(self, job: Job) -> None:
        if job.initial_delay_s:
            await asyncio.sleep(job.initial_delay_s * (0.8 + 0.4 * random.random()))
        while True:
            await self._run(job)
            await asyncio.sleep(self._next_delay(job))

    def snapshot(self) -> dict[str, dict[str, Any]]:
        return {
            name: {
                "enabled": j.enabled,
                "interval_s": j.interval_s,
                "runs": j.runs,
                "failures": j.failures,
                "last_run": int(j.last_run) if j.last_run else None,
                "last_error": j.last_error,
            }
            for name, j in self._jobs.items()
        }
