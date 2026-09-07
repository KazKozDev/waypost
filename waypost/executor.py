"""Executor: the degradation ladder.

  1. Retry on the same provider and key — only 5xx/timeout, exponential
     backoff with jitter, at most two attempts
  2. Another key of the same provider — the model is the same, the prefix
     is not lost
  3. Another provider (the same model at another host, or a neighbor)
  4. Another model of the same tier
  5. A local model — the end of the cascade, always available
  6. An approximate answer from the semantic cache (see server.py)
  7. A structured error

Hedging: if the first candidate stays silent longer than its typical
TTFT, a second one starts in parallel, the first to answer wins. This
costs quota, so it is bounded by a share of requests (hedge_budget) —
otherwise the spend doubles for nothing.

Idempotency: reserve is done on a specific key and returned if the
attempt did not happen (an error or a canceled hedge). Without this a
canceled hedge would charge the quota that nobody spent.
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import time
import uuid
from collections import deque
from typing import Any, AsyncIterator

from . import pricing
from .bandit import Bandit
from .breaker import CircuitBreaker, ProbeToken
from .latency import LatencyTracker
from .ledger import Ledger
from .ratelimit import RateGovernor
from .providers.openai_compat import (
    OpenAICompatAdapter,
    ProviderError,
    Verdict,
)
from .registry import Offering
from .router import Candidate
from .schemas import (
    ChatRequest,
    ChatResponse,
    RequestProfile,
    RouterError,
    RouterMeta,
    Usage,
)
from .telemetry import AttemptLogEntry, Telemetry
from .verify import Verifier, map_outcome

log = logging.getLogger("waypost.executor")

# How many times the typical TTFT we wait before hedging.
HEDGE_TTFT_FACTOR = 3.0
HEDGE_MIN_DELAY_S = 0.5

# Total wall-clock budget for one request, by latency class. Without it the
# ladder is unbounded: max_attempts candidates x retries_per_provider x
# timeout_s is ~16 minutes of a client holding a socket open, and every
# rung of the ladder is spent on a request nobody is waiting for any more.
DEADLINE_S = {
    "interactive": 20.0,
    "code_completion": 12.0,
    "batch": 300.0,
    "reasoning": 90.0,
}
DEFAULT_DEADLINE_S = 30.0
# Below this there is no point starting another attempt.
MIN_SLICE_S = 0.75
# Share of the remaining budget a single attempt may spend while lower rungs
# of the ladder are still ahead. Letting the first candidate use the whole
# budget means the local fallback — the rung that always answers — never
# gets its turn, and the client gets a 502 instead of a slightly worse
# answer. The last candidate gets whatever is left.
ATTEMPT_SHARE = 0.6


class _StreamUsageTracker:
    """Incrementally read usage from arbitrarily chunked SSE bytes."""

    def __init__(self) -> None:
        self._buffer = b""
        self.body: dict[str, Any] | None = None

    def feed(self, chunk: bytes) -> None:
        self._buffer += chunk
        lines = self._buffer.split(b"\n")
        self._buffer = lines.pop()
        for line in lines:
            self._parse_line(line.rstrip(b"\r"))

    def finish(self) -> None:
        if self._buffer:
            self._parse_line(self._buffer.rstrip(b"\r"))
            self._buffer = b""

    def _parse_line(self, line: bytes) -> None:
        if not line.startswith(b"data:"):
            return
        raw = line[5:].strip()
        if not raw or raw == b"[DONE]":
            return
        try:
            body = json.loads(raw)
        except (TypeError, ValueError):
            return
        if isinstance(body, dict) and isinstance(body.get("usage"), dict):
            self.body = body


def _detect_modality(req: ChatRequest) -> tuple[str, int, float, int | None]:
    """Detect modality from ChatRequest messages."""
    image_count = 0
    audio_duration_s = 0.0
    has_audio = False
    has_video = False
    for m in req.messages:
        if isinstance(m.content, list):
            for part in m.content:
                if isinstance(part, dict):
                    t = part.get("type")
                    if t in ("image_url", "image"):
                        image_count += 1
                    elif t in ("audio", "input_audio"):
                        has_audio = True
                    elif t == "video":
                        has_video = True
    if image_count and (has_audio or has_video):
        return "mixed", image_count, audio_duration_s, None
    if image_count:
        return "image", image_count, audio_duration_s, None
    if has_audio:
        return "audio", image_count, audio_duration_s, None
    if has_video:
        return "video", image_count, audio_duration_s, None
    return "text", 0, 0.0, None


class Attempt:
    """The result of an attempt: either an answer or an error. Only FATAL
    escapes upward — the ladder must digest the rest."""

    __slots__ = ("response", "error", "offering", "key_index")

    def __init__(
        self,
        response: ChatResponse | None = None,
        error: ProviderError | None = None,
        offering: Offering | None = None,
        key_index: int = 0,
    ):
        self.response = response
        self.error = error
        self.offering = offering
        self.key_index = key_index

    @property
    def ok(self) -> bool:
        return self.response is not None


class Executor:
    def __init__(
        self,
        adapter: OpenAICompatAdapter,
        ledger: Ledger,
        breaker: CircuitBreaker,
        telemetry: Telemetry,
        *,
        verifier: Verifier | None = None,
        max_attempts: int = 4,
        timeout_s: float = 120.0,
        retries_per_provider: int = 2,
        bandit: Bandit | None = None,
        hedge_budget: float = 0.05,
        enable_hedging: bool = True,
        enable_exploration: bool = False,
        explore_rate: float = 0.10,
        latency: LatencyTracker | None = None,
        rate_governor: RateGovernor | None = None,
        inflight: dict[str, int] | None = None,
        deadlines: dict[str, float] | None = None,
        hedge_window_s: float = 300.0,
        triggers: Any | None = None,
    ):
        self.adapter = adapter
        self.ledger = ledger
        self.breaker = breaker
        self.telemetry = telemetry
        self.bandit = bandit
        self.latency = latency or LatencyTracker()
        self.rate_governor = rate_governor or RateGovernor(ledger)
        self.inflight = inflight if inflight is not None else {}
        self.deadlines = deadlines or dict(DEADLINE_S)
        # ProbeTriggers, if the control plane is running. A run of server
        # errors is worth re-measuring now rather than at the next daily
        # probe, by which time the model may be long gone.
        self.triggers = triggers
        self._error_runs: dict[str, int] = {}
        self.verifier = verifier or Verifier()
        self.max_attempts = max_attempts
        self.timeout_s = timeout_s
        self.retries_per_provider = retries_per_provider
        self.enable_hedging = enable_hedging
        self.hedge_budget = hedge_budget
        self.enable_exploration = enable_exploration
        self.explore_rate = explore_rate
        self._slots: dict[str, asyncio.Semaphore] = {}
        self._counters = {
            "requests": 0,
            "hedges": 0,
            "billed": 0,
            "deadline_exceeded": 0,
            "explorations": 0,
        }
        # Fire-and-forget exploration tasks: asyncio only holds a weak
        # reference, so without this set they can be collected mid-flight.
        self._background: set[asyncio.Task] = set()
        # Sliding window for the hedge budget — see _hedge_allowed().
        self.hedge_window_s = hedge_window_s
        self._req_times: deque[float] = deque()
        self._hedge_times: deque[float] = deque()

    # ------------------------------------------------------------ deadline
    def budget_for(self, req: ChatRequest) -> float:
        return self.deadlines.get(
            req.profile or "", self.deadlines.get(req.latency_class, DEFAULT_DEADLINE_S)
        )

    # --------------------------------------------------------------- util
    def _bandit_update(self, o: Offering, profile: RequestProfile, ok: bool) -> None:
        """Quality evidence only.

        The bandit estimates how good this model is at this task class.
        A 429, a connection reset or a 5xx says nothing about that — those
        belong to the rate governor and the circuit breaker. Feeding them
        here made the router avoid whichever provider was merely busy, and
        rewarding every HTTP 200 (as this used to, before the verifier
        ran) taught it that a syntactically broken answer was a success.
        """
        if self.bandit is not None:
            self.bandit.update(profile.task_class, o.key, 1.0 if ok else 0.0)

    ERROR_RUN_TRIGGER = 3

    def _note_error_run(self, o: Offering) -> None:
        """Three server errors in a row from one offering: ask for a probe.

        Not a breaker decision — that one is per provider and about
        admission. This is per model and about knowledge: the model may
        have been retired, or lost a capability, and only a canary can
        tell the difference from a bad afternoon.
        """
        n = self._error_runs.get(o.key, 0) + 1
        self._error_runs[o.key] = n
        if n >= self.ERROR_RUN_TRIGGER and self.triggers is not None:
            self.triggers.request(o.key, f"{n} consecutive errors")
            self._error_runs[o.key] = 0

    def _enter(self, o: Offering) -> None:
        self.inflight[o.key] = self.inflight.get(o.key, 0) + 1

    def _leave(self, o: Offering) -> None:
        n = self.inflight.get(o.key, 0) - 1
        if n > 0:
            self.inflight[o.key] = n
        else:
            self.inflight.pop(o.key, None)

    def _account(
        self, o: Offering, est: int, body: dict[str, Any], key_index: int
    ) -> Usage:
        pt, ct, total, _, measured = self._usage_details(o, body)
        self.ledger.commit(o, est, total if measured else est, key_index)
        self._check_billed(o, body)
        return Usage(
            prompt_tokens=pt or 0,
            completion_tokens=ct or 0,
            total_tokens=total,
        )

    def _check_billed(self, o: Offering, body: dict[str, Any]) -> None:
        """The last line of wallet defense."""
        cost = pricing.billed_cost(body)
        if cost is None or cost <= 0:
            return
        verdict = pricing.Verdict(
            pricing.Cost.PAID, pricing.Source.BILLED, f"billed {cost:g}"
        )
        self._counters["billed"] = self._counters.get("billed", 0) + 1
        if o.set_pricing(verdict):
            log.error(
                "PAID: %s billed %g — model marked paid and excluded from selection",
                o.key,
                cost,
            )

    @staticmethod
    def _usage_details(
        o: Offering, body: dict[str, Any]
    ) -> tuple[int | None, int | None, int, float | None, bool]:
        usage = body.get("usage")
        measured = isinstance(usage, dict) and any(
            key in usage
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        )
        prompt_tokens = (
            int(usage["prompt_tokens"])
            if measured and usage.get("prompt_tokens") is not None
            else None
        )
        completion_tokens = (
            int(usage["completion_tokens"])
            if measured and usage.get("completion_tokens") is not None
            else None
        )
        split_total = (prompt_tokens or 0) + (completion_tokens or 0)
        total_tokens = (
            int(usage["total_tokens"])
            if measured and usage.get("total_tokens") is not None
            else split_total
        )
        billed = pricing.billed_cost(body)
        actual_cost = (
            billed
            if billed is not None
            else (0.0 if o.free or o.is_local else None)
        )
        return prompt_tokens, completion_tokens, total_tokens, actual_cost, measured

    def _slice(self, deadline: float | None, share: float) -> float:
        """How long this attempt may run: a share of what is left, so the
        rungs below it still have a budget to run in."""
        left = self._left(deadline)
        if left == float("inf"):
            return self.timeout_s
        return min(self.timeout_s, max(0.0, left * share))

    @staticmethod
    def _left(deadline: float | None) -> float:
        """Seconds of budget left. No deadline means the old unbounded
        behaviour, kept for embedded callers and the batch worker."""
        if deadline is None:
            return float("inf")
        return deadline - time.monotonic()

    async def _backoff(self, attempt: int, deadline: float | None = None) -> None:
        delay = min(8.0, 0.4 * (2**attempt)) * (0.5 + random.random())
        # Sleeping past the deadline burns the budget the retry needs.
        delay = min(delay, max(0.0, self._left(deadline) - MIN_SLICE_S))
        if delay > 0:
            await asyncio.sleep(delay)

    def _slot(self, o: Offering) -> asyncio.Semaphore | None:
        """A parallelism limiter. For mlx-lm this is not tuning: two
        heavy generations at once contend for Mac Studio memory."""
        if not o.concurrency:
            return None
        sem = self._slots.get(o.key)
        if sem is None:
            sem = asyncio.Semaphore(o.concurrency)
            self._slots[o.key] = sem
        return sem

    # ------------------------------------------------------------- attempt
    async def _call(
        self,
        o: Offering,
        req: ChatRequest,
        profile: RequestProfile,
        key_index: int,
        *,
        meta: RouterMeta | None = None,
        routing_source: str | None = None,
        is_exploration: bool = False,
        attempt_no: int = 1,
        timeout: float | None = None,
    ) -> Attempt:
        """One provider call with reservation, accounting and attempt logging."""
        est = profile.est_total_tokens
        call_timeout = self.timeout_s if timeout is None else max(0.1, timeout)
        # The half-open probe token is taken HERE, at the real call — not
        # while planning. It is released on every path below.
        probe = self.breaker.acquire_probe(o.provider)
        if CircuitBreaker.probe_lost_race(probe):
            return Attempt(
                error=ProviderError(
                    Verdict.SWITCH, 503, "circuit open (probe in flight)", 5.0
                ),
                offering=o,
                key_index=key_index,
            )
        if not self.ledger.reserve(o, est, key_index):
            # A shared ledger refused: another replica took the last slot
            # between planning and calling. Skip without spending the
            # request — that refusal is exactly what it exists to prevent.
            self.breaker.release_probe(probe)
            return Attempt(
                error=ProviderError(
                    Verdict.SWITCH, 429, "quota taken by another replica", 5.0
                ),
                offering=o,
                key_index=key_index,
            )
        t0 = time.perf_counter()
        sem = self._slot(o)
        modality, img_cnt, aud_dur, vis_bud = _detect_modality(req)
        req_id = (
            (meta.request_id if meta and meta.request_id else None)
            or req.session_id
            or f"req_{uuid.uuid4().hex[:8]}"
        )
        if meta and not meta.request_id:
            meta.request_id = req_id
        src = routing_source or (
            meta.routing_source
            if meta and meta.routing_source
            else profile.classifier_source
        )
        backend = (
            "ollama"
            if "ollama" in o.provider.lower()
            else (
                "mlx"
                if "mlx" in o.provider.lower()
                else ("cloud" if not o.is_local else "local")
            )
        )

        async def _do() -> dict[str, Any]:
            if sem is not None:
                async with sem:
                    return await self.adapter.complete(
                        o,
                        req,
                        call_timeout,
                        api_key=o.api_key_at(key_index),
                        idempotency_key=req.idempotency_key,
                    )
            return await self.adapter.complete(
                o,
                req,
                call_timeout,
                api_key=o.api_key_at(key_index),
                idempotency_key=req.idempotency_key,
            )

        self._enter(o)
        try:
            # The transport timeout is not enough to bound an attempt: it
            # applies per socket operation, so a provider dripping bytes
            # slowly can outlive any read timeout. The budget is enforced
            # here, where it is actually a deadline.
            body = await asyncio.wait_for(_do(), call_timeout)
        except (asyncio.TimeoutError, TimeoutError) as exc:
            self.ledger.commit(o, est, 0, key_index)
            self._leave(o)
            self.breaker.release_probe(probe)
            latency = (time.perf_counter() - t0) * 1000
            self.latency.observe(o.key, latency)
            self.telemetry.log_attempt(
                o.key,
                profile,
                ok=False,
                verdict=Verdict.RETRY.value,
                latency_ms=latency,
                key_index=key_index,
            )
            log.warning(
                "attempt %s[key%d] exceeded its %.1fs slice", o.key, key_index,
                call_timeout,
            )
            return Attempt(
                error=ProviderError(
                    Verdict.RETRY, 408, f"attempt exceeded {call_timeout:.1f}s", 1.0
                ),
                offering=o,
                key_index=key_index,
            )
        except asyncio.CancelledError:
            # A canceled hedge: return the quota, otherwise the losing
            # branch eats the limit while spending nothing.
            self.ledger.commit(o, est, 0, key_index)
            self.breaker.release_probe(probe)
            self._leave(o)
            raise
        except ProviderError as exc:
            self.ledger.commit(o, est, 0, key_index)
            self._leave(o)
            latency = (time.perf_counter() - t0) * 1000
            # A refusal is not a quality signal, so the bandit is not
            # touched here. A timeout IS a latency signal: the budget it
            # burned is exactly what the router needs to see.
            if exc.status in (408, 504):
                self.latency.observe(o.key, latency)
            if exc.status == 429:
                self.rate_governor.on_rate_limit(o, key_index)
            self.breaker.release_probe(probe)
            self.telemetry.log_attempt(
                o.key,
                profile,
                ok=False,
                verdict=exc.verdict.value,
                latency_ms=latency,
                key_index=key_index,
            )
            stat = (
                "rate_limited"
                if exc.status == 429
                else (
                    "timeout"
                    if exc.status == 408
                    else ("refused" if exc.status == 403 else "error")
                )
            )
            outcome, outcome_source, outcome_detail = map_outcome(
                False, status=stat, error_detail=exc.message
            )
            entry = AttemptLogEntry(
                request_id=req_id,
                attempt_no=attempt_no,
                ts=time.time(),
                embedding=profile.embedding,
                embedder_version="minishlab/potion-multilingual-128M",
                l0_labels={
                    "task_type": profile.task_class,
                    "tier": profile.tier.value,
                    "complexity": profile.complexity,
                    "lang": profile.language,
                    "requires": [c.value for c in profile.required_caps],
                },
                l1_prediction=getattr(meta, "l1_prediction", None) if meta else None,
                input_tokens=est,
                modality=modality,
                image_count=img_cnt,
                audio_duration_s=aud_dur,
                vision_token_budget=vis_bud,
                provider=o.provider,
                backend=backend,
                model=o.model_id,
                model_version=getattr(o, "model_version", "") or "",
                tier=profile.tier.value,
                thinking_mode=bool(getattr(req, "thinking_mode", False)),
                routing_source=src,
                is_exploration=is_exploration,
                quota_remaining_pct=100.0,
                quota_window_reset_in_s=0,
                quota_binding_limit="requests",
                status=stat,
                error_class=type(exc).__name__,
                latency_ms=int(latency),
                ttft_ms=0,
                output_tokens=0,
                reasoning_tokens=None,
                peak_memory_mb=None,
                is_final=False,
                outcome=outcome,
                outcome_source=outcome_source,
                outcome_detail=outcome_detail,
            )
            self.telemetry.log_attempt_row(entry)
            log.warning("attempt %s[key%d] failed: %s", o.key, key_index, exc)
            return Attempt(error=exc, offering=o, key_index=key_index)

        self._leave(o)
        usage = self._account(o, est, body, key_index)
        prompt_tokens, completion_tokens, _, actual_cost, usage_measured = (
            self._usage_details(o, body)
        )
        self.breaker.on_success(o.provider)  # also clears the probe token
        self.rate_governor.on_success(o, key_index)
        self._error_runs.pop(o.key, None)
        latency = (time.perf_counter() - t0) * 1000
        self.latency.observe(o.key, latency)
        self.telemetry.log_attempt(
            o.key,
            profile,
            ok=True,
            verdict="ok",
            latency_ms=latency,
            tokens=usage.total_tokens,
            key_index=key_index,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            actual_cost_usd=actual_cost,
            usage_measured=usage_measured,
        )

        v_ok, v_reason = (
            self.verifier.verify(req, profile, body) if self.verifier else (True, "")
        )
        # The reward is the verifier's verdict, not the HTTP status. A 200
        # carrying malformed JSON or truncated code is not a success, and
        # rewarding it teaches the bandit to prefer exactly that model.
        self._bandit_update(o, profile, ok=v_ok)
        outcome, outcome_source, outcome_detail = map_outcome(
            v_ok, v_reason, status="ok"
        )
        stat = "truncated" if v_reason == "truncated" else ("ok" if v_ok else "error")
        usage_body = body.get("usage") or {}
        completion_details = usage_body.get("completion_tokens_details") or {}
        reasoning_tokens = completion_details.get("reasoning_tokens")
        entry = AttemptLogEntry(
            request_id=req_id,
            attempt_no=attempt_no,
            ts=time.time(),
            embedding=profile.embedding,
            embedder_version="minishlab/potion-multilingual-128M",
            l0_labels={
                "task_type": profile.task_class,
                "tier": profile.tier.value,
                "complexity": profile.complexity,
                "lang": profile.language,
                "requires": [c.value for c in profile.required_caps],
            },
            l1_prediction=getattr(meta, "l1_prediction", None) if meta else None,
            input_tokens=usage.prompt_tokens,
            modality=modality,
            image_count=img_cnt,
            audio_duration_s=aud_dur,
            vision_token_budget=vis_bud,
            provider=o.provider,
            backend=backend,
            model=o.model_id,
            model_version=getattr(o, "model_version", "") or "",
            tier=profile.tier.value,
            thinking_mode=bool(getattr(req, "thinking_mode", False)),
            routing_source=src,
            is_exploration=is_exploration,
            quota_remaining_pct=100.0,
            quota_window_reset_in_s=0,
            quota_binding_limit="requests",
            status=stat,
            error_class="",
            latency_ms=int(latency),
            ttft_ms=int(getattr(o, "ttft_p50_ms", 0) or 0),
            output_tokens=usage.completion_tokens,
            reasoning_tokens=reasoning_tokens,
            peak_memory_mb=None,
            is_final=True,
            outcome=outcome,
            outcome_source=outcome_source,
            outcome_detail=outcome_detail,
        )
        self.telemetry.log_attempt_row(entry)

        log.debug(
            "attempt %s[key%d] ok tokens=%d %.0fms",
            o.key,
            key_index,
            usage.total_tokens,
            latency,
        )
        resp = ChatResponse(
            id=body.get("id") or f"chatcmpl-{uuid.uuid4().hex[:20]}",
            model=o.key,
            choices=body.get("choices", []),
            usage=usage,
            router=RouterMeta(),
        )
        return Attempt(response=resp, offering=o, key_index=key_index)

    async def _try_candidate(
        self,
        cand: Candidate,
        req: ChatRequest,
        profile: RequestProfile,
        backup: Candidate | None = None,
        meta: RouterMeta | None = None,
        deadline: float | None = None,
        share: float = 1.0,
    ) -> Attempt:
        """A whole candidate: retries on 5xx and key rotation on 429."""
        o = cand.offering
        est = profile.est_total_tokens
        last: Attempt | None = None
        tried: set[int] = set()
        first_call = True

        for _ in range(max(1, o.key_count)):
            if self._left(deadline) < MIN_SLICE_S:
                break
            key_index = self.ledger.pick_key(o, est)
            if key_index is None or key_index in tried:
                # No free keys left: down the ladder.
                break
            tried.add(key_index)

            for attempt in range(self.retries_per_provider):
                slice_s = self._slice(deadline, share)
                if slice_s < MIN_SLICE_S:
                    self._counters["deadline_exceeded"] += 1
                    return last or Attempt(
                        error=ProviderError(
                            Verdict.SWITCH, 504, "request deadline exceeded", 1.0
                        ),
                        offering=o,
                    )
                att_no = (meta.attempts if meta else 1) or 1
                if first_call and backup is not None and self._hedge_allowed(req):
                    res = await self._call_with_hedge(
                        o,
                        req,
                        profile,
                        key_index,
                        backup,
                        meta,
                        attempt_no=att_no,
                        timeout=slice_s,
                    )
                else:
                    res = await self._call(
                        o,
                        req,
                        profile,
                        key_index,
                        meta=meta,
                        attempt_no=att_no,
                        timeout=slice_s,
                    )
                first_call = False
                if res.ok:
                    return res
                last = res
                exc = res.error
                if exc is None:
                    break

                if exc.verdict is Verdict.FATAL:
                    return res
                if exc.verdict is Verdict.SWITCH:
                    # 429 / auth / retired model. The ledger blocks this key
                    # for as long as the provider asked; the breaker is NOT
                    # touched — being rate limited is not being unhealthy,
                    # and counting it here used to open a live provider
                    # after four refusals.
                    self.ledger.penalize(o, exc.retry_after_s, key_index)
                    break  # this key is out — try the next
                # RETRY: 5xx, timeouts, connection errors. Health signal.
                self.breaker.on_failure(o.provider)
                self._note_error_run(o)
                if attempt + 1 < self.retries_per_provider:
                    await self._backoff(attempt, deadline)
        return last or Attempt(
            error=ProviderError(Verdict.SWITCH, 429, "quota exhausted", 60.0),
            offering=o,
        )

    # -------------------------------------------------------------- hedge
    @staticmethod
    async def _cancel(tasks) -> None:
        """Cancel the losing branches and WAIT for the cancellation."""
        tasks = [t for t in tasks if not t.done()]
        for t in tasks:
            t.cancel()
        if tasks:
            await asyncio.wait(tasks, timeout=2.0)

    def _hedge_delay(self, o: Offering) -> float:
        ttft = self.latency.get(o.key, o.ttft_p50_ms) or o.ttft_p50_ms
        return max(HEDGE_MIN_DELAY_S, ttft * HEDGE_TTFT_FACTOR / 1000.0)

    def _hedge_allowed(self, req: ChatRequest) -> bool:
        """A hedge is a second quota spend. Allowed for interactive and
        only within budget: 5% of requests, no more.

        Measured over a sliding window, not over the life of the process.
        A router that has been up for a week accumulates so many requests
        in the denominator that the ratio stops moving: a burst of hedges
        right now barely dents it, and the budget silently stops being a
        budget.
        """
        if not self.enable_hedging or req.latency_class != "interactive":
            return False
        now = time.monotonic()
        cutoff = now - self.hedge_window_s
        while self._req_times and self._req_times[0] < cutoff:
            self._req_times.popleft()
        while self._hedge_times and self._hedge_times[0] < cutoff:
            self._hedge_times.popleft()
        done = len(self._req_times)
        if done < 20:  # too early to judge the share
            return not self._hedge_times
        return len(self._hedge_times) / done < self.hedge_budget

    async def _call_with_hedge(
        self,
        o: Offering,
        req: ChatRequest,
        profile: RequestProfile,
        key_index: int,
        backup: Candidate,
        meta: RouterMeta | None,
        attempt_no: int = 1,
        timeout: float | None = None,
    ) -> Attempt:
        """The first call of a candidate against a backup: the first to
        answer wins, the loser is canceled and returns the quota."""
        primary = asyncio.create_task(
            self._call(
                o,
                req,
                profile,
                key_index,
                meta=meta,
                routing_source="l0",
                attempt_no=attempt_no,
                timeout=timeout,
            )
        )
        delay = self._hedge_delay(o)
        done, _ = await asyncio.wait({primary}, timeout=delay)
        if primary in done:
            return primary.result()

        self._counters["hedges"] += 1
        self._hedge_times.append(time.monotonic())
        log.info(
            "HEDGE %s silent > %.1fs → in parallel %s",
            o.key,
            delay,
            backup.offering.key,
        )
        hedge = asyncio.create_task(
            self._call(
                backup.offering,
                req,
                profile,
                0,
                meta=meta,
                routing_source="hedge",
                attempt_no=attempt_no + 1,
                timeout=timeout,
            )
        )
        pending = {primary, hedge}
        last: Attempt | None = None
        while pending:
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )
            for finished in done:
                res = finished.result()
                if res.ok:
                    await self._cancel(pending)
                    if finished is hedge and meta is not None:
                        meta.hedged = True
                        meta.fallback_path.append(backup.offering.key)
                    return res
                last = res
        return last or Attempt(offering=o, key_index=key_index)

    async def _run_exploration(
        self, cand: Candidate, req: ChatRequest, profile: RequestProfile, req_id: str
    ) -> None:
        """Background exploration call to record stats without affecting user response."""
        try:
            o = cand.offering
            est = profile.est_total_tokens
            key_index = self.ledger.pick_key(o, est)
            if key_index is None:
                return
            await self._call(
                o,
                req,
                profile,
                key_index,
                routing_source="explore",
                is_exploration=True,
                attempt_no=99,
            )
        except Exception as exc:  # noqa: BLE001
            log.debug("exploration call failed: %s", exc)

    # ------------------------------------------------------------ execute
    async def execute(
        self,
        req: ChatRequest,
        profile: RequestProfile,
        plan: list[Candidate],
        meta: RouterMeta,
    ) -> ChatResponse:
        if not plan:
            raise RouterError(
                "no available candidates: check keys, " "quotas and the privacy filter",
                503,
                meta,
            )

        started = time.perf_counter()
        self._counters["requests"] += 1
        self._req_times.append(time.monotonic())
        candidates = plan[: self.max_attempts]
        last: ProviderError | None = None
        # One budget for the whole ladder, not per attempt. Descending the
        # ladder must never take longer than the answer is worth.
        deadline = time.monotonic() + self.budget_for(req)
        meta.deadline_s = round(self.budget_for(req), 1)

        for i, cand in enumerate(candidates):
            if self._left(deadline) < MIN_SLICE_S:
                self._counters["deadline_exceeded"] += 1
                log.warning(
                    "DEADLINE %.1fs exhausted after %d attempts (path %s)",
                    self.budget_for(req),
                    meta.attempts,
                    ">".join(meta.fallback_path),
                )
                break
            meta.attempts += 1
            meta.fallback_path.append(cand.offering.key)
            backup = candidates[i + 1] if i + 1 < len(candidates) else None
            # The last candidate may use everything that is left; the ones
            # before it must leave room for it.
            is_last = i == len(candidates) - 1
            res = await self._try_candidate(
                cand,
                req,
                profile,
                backup,
                meta,
                deadline=deadline,
                share=1.0 if is_last else ATTEMPT_SHARE,
            )

            if res.ok and res.response is not None:
                o = res.offering or cand.offering
                meta.provider = o.provider
                meta.model = o.model_id
                meta.key_index = res.key_index
                meta.latency_ms = int((time.perf_counter() - started) * 1000)
                res.response.router = meta
                res.response.model = o.key

                # Exploration by duplicate call is off by default now: the
                # router samples the bandit posterior when it selects, so
                # exploration happens inside the choice instead of costing a
                # second call. When it is switched back on it must at least
                # not run while the alternative's quota is under pressure.
                if (
                    self.enable_exploration
                    and random.random() < self.explore_rate
                    and len(plan) > 1
                ):
                    alt_candidates = [
                        c
                        for c in plan
                        if c.offering.key != o.key
                        and self.ledger.pressure(c.offering) < 0.5
                    ]
                    if alt_candidates:
                        chosen_alt = random.choice(alt_candidates)
                        self._counters["explorations"] += 1
                        task = asyncio.create_task(
                            self._run_exploration(
                                chosen_alt, req, profile, meta.request_id or ""
                            )
                        )
                        # asyncio keeps only a weak reference to a bare task.
                        self._background.add(task)
                        task.add_done_callback(self._background.discard)

                return res.response

            last = res.error
            if last is not None and last.verdict is Verdict.FATAL:
                # An invalid request: the ladder will not save it, a body
                # fix will.
                raise RouterError(last.message, last.status, meta)

        raise RouterError(
            f"all candidates refused ({meta.attempts} attempts); "
            f"last error: {last}",
            502,
            meta,
        )

    # ------------------------------------------------------------- stream
    async def stream(
        self,
        req: ChatRequest,
        profile: RequestProfile,
        plan: list[Candidate],
        meta: RouterMeta,
    ) -> AsyncIterator[bytes]:
        """Fallback is possible only before the first byte sent to the
        client. After it, the stream belongs to the chosen provider.

        So the switch decision is made by a TTFT timeout: a silent
        provider is dropped before it delivered the first chunk.
        """
        if not plan:
            raise RouterError("no available candidates", 503, meta)

        last: ProviderError | None = None
        for cand in plan[: self.max_attempts]:
            o = cand.offering
            est = profile.est_total_tokens
            key_index = self.ledger.pick_key(o, est)
            if key_index is None:
                continue
            meta.attempts += 1
            meta.fallback_path.append(o.key)
            if not self.ledger.reserve(o, est, key_index):
                continue  # another replica took the last slot
            first_byte_sent = False
            ttft_budget = self.timeout_s if o.is_local else max(25.0, self._hedge_delay(o) * 5)
            t0 = time.perf_counter()
            try:
                usage_tracker = _StreamUsageTracker()
                stream = self.adapter.stream(
                    o,
                    req,
                    self.timeout_s,
                    api_key=o.api_key_at(key_index),
                    idempotency_key=req.idempotency_key,
                )
                agen = stream.__aiter__()
                while True:
                    # Before the first byte, wait no longer than the
                    # TTFT budget: after it switching is impossible, so
                    # decide now.
                    timeout = ttft_budget if not first_byte_sent else None
                    try:
                        chunk = await asyncio.wait_for(agen.__anext__(), timeout)
                    except StopAsyncIteration:
                        break
                    except (asyncio.TimeoutError, TimeoutError) as exc:
                        raise ProviderError(
                            Verdict.RETRY,
                            408,
                            f"no first chunk within {ttft_budget:.1f}s",
                            1.0,
                        ) from exc
                    if not first_byte_sent:
                        first_byte_sent = True
                        meta.provider, meta.model = o.provider, o.model_id
                        meta.key_index = key_index
                        self.breaker.on_success(o.provider)
                        self.rate_governor.on_success(o, key_index)
                        # TTFT measured for real — the streaming path is the
                        # only place where it can be.
                        self.latency.observe(o.key, (time.perf_counter() - t0) * 1000)
                        log.debug(
                            "stream %s ttft=%.0fms",
                            o.key,
                            (time.perf_counter() - t0) * 1000,
                        )
                    usage_tracker.feed(chunk)
                    yield chunk
                usage_tracker.finish()
                usage_body = usage_tracker.body or {}
                (
                    prompt_tokens,
                    completion_tokens,
                    total_tokens,
                    actual_cost,
                    usage_measured,
                ) = self._usage_details(o, usage_body)
                if usage_body:
                    self._check_billed(o, usage_body)
                self.ledger.commit(
                    o, est, total_tokens if usage_measured else est, key_index
                )
                self.telemetry.log_attempt(
                    o.key,
                    profile,
                    ok=True,
                    verdict="ok_stream",
                    tokens=total_tokens,
                    key_index=key_index,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    actual_cost_usd=actual_cost,
                    usage_measured=usage_measured,
                )
                self._bandit_update(o, profile, ok=True)
                return
            except ProviderError as exc:
                last = exc
                self.ledger.commit(o, est, 0, key_index)
                if exc.verdict is Verdict.SWITCH:
                    self.ledger.penalize(o, exc.retry_after_s, key_index)
                    if exc.status == 429:
                        self.rate_governor.on_rate_limit(o, key_index)
                else:
                    self.breaker.on_failure(o.provider)
                if exc.status in (408, 504):
                    self.latency.observe(o.key, (time.perf_counter() - t0) * 1000)
                self.telemetry.log_attempt(
                    o.key,
                    profile,
                    ok=False,
                    verdict=exc.verdict.value,
                    key_index=key_index,
                )
                if first_byte_sent:
                    # Too late to switch — honestly break the stream.
                    yield b'data: {"error":"upstream interrupted"}\n\n'
                    return
                if exc.verdict is Verdict.FATAL:
                    raise RouterError(exc.message, exc.status, meta) from exc
                continue

        raise RouterError(f"streaming failed: {last}", 502, meta)

    def snapshot(self) -> dict:
        return {
            "requests": self._counters["requests"],
            "hedges": self._counters["hedges"],
            "billed": self._counters["billed"],
            "deadline_exceeded": self._counters["deadline_exceeded"],
            "explorations": self._counters["explorations"],
            "inflight": dict(self.inflight),
            "rate_limits": self.rate_governor.snapshot(),
            # Over the window the budget actually governs, not over uptime.
            "hedge_rate": round(len(self._hedge_times) / len(self._req_times), 3)
            if self._req_times
            else 0.0,
            "hedge_window_s": self.hedge_window_s,
        }
