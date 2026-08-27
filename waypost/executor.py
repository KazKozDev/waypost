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
from typing import Any, AsyncIterator

from . import pricing
from .bandit import Bandit
from .breaker import CircuitBreaker
from .ledger import Ledger
from .providers.openai_compat import (
    OpenAICompatAdapter,
    ProviderError,
    Verdict,
    parse_usage,
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
        enable_exploration: bool = True,
        explore_rate: float = 0.10,
    ):
        self.adapter = adapter
        self.ledger = ledger
        self.breaker = breaker
        self.telemetry = telemetry
        self.bandit = bandit
        self.verifier = verifier or Verifier()
        self.max_attempts = max_attempts
        self.timeout_s = timeout_s
        self.retries_per_provider = retries_per_provider
        self.enable_hedging = enable_hedging
        self.hedge_budget = hedge_budget
        self.enable_exploration = enable_exploration
        self.explore_rate = explore_rate
        self._slots: dict[str, asyncio.Semaphore] = {}
        self._counters = {"requests": 0, "hedges": 0, "billed": 0}

    # --------------------------------------------------------------- util
    def _bandit_update(self, o: Offering, profile: RequestProfile, ok: bool) -> None:
        if self.bandit is not None:
            self.bandit.update(profile.task_class, o.key, 1.0 if ok else 0.0)

    def _account(
        self, o: Offering, est: int, body: dict[str, Any], key_index: int
    ) -> Usage:
        pt, ct = parse_usage(body)
        self.ledger.commit(o, est, pt + ct or est, key_index)
        self._check_billed(o, body)
        return Usage(prompt_tokens=pt, completion_tokens=ct, total_tokens=pt + ct)

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

    def _sniff_stream_cost(self, o: Offering, chunk: bytes) -> None:
        """The same bill check, but for SSE."""
        if b'"cost"' not in chunk and b'"total_cost"' not in chunk:
            return
        for line in chunk.split(b"\n"):
            if not line.startswith(b"data: "):
                continue
            raw = line[6:].strip()
            if not raw or raw == b"[DONE]":
                continue
            try:
                body = json.loads(raw)
            except ValueError:
                continue
            if isinstance(body, dict):
                self._check_billed(o, body)

    async def _backoff(self, attempt: int) -> None:
        delay = min(8.0, 0.4 * (2**attempt)) * (0.5 + random.random())
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
    ) -> Attempt:
        """One provider call with reservation, accounting and attempt logging."""
        est = profile.est_total_tokens
        self.ledger.reserve(o, est, key_index)
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

        try:
            if sem is not None:
                async with sem:
                    body = await self.adapter.complete(
                        o,
                        req,
                        self.timeout_s,
                        api_key=o.api_key_at(key_index),
                        idempotency_key=req.idempotency_key,
                    )
            else:
                body = await self.adapter.complete(
                    o,
                    req,
                    self.timeout_s,
                    api_key=o.api_key_at(key_index),
                    idempotency_key=req.idempotency_key,
                )
        except asyncio.CancelledError:
            # A canceled hedge: return the quota, otherwise the losing
            # branch eats the limit while spending nothing.
            self.ledger.commit(o, est, 0, key_index)
            raise
        except ProviderError as exc:
            self.ledger.commit(o, est, 0, key_index)
            self._bandit_update(o, profile, ok=False)
            latency = (time.perf_counter() - t0) * 1000
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

        usage = self._account(o, est, body, key_index)
        self.breaker.on_success(o.provider)
        self._bandit_update(o, profile, ok=True)
        latency = (time.perf_counter() - t0) * 1000
        self.telemetry.log_attempt(
            o.key,
            profile,
            ok=True,
            verdict="ok",
            latency_ms=latency,
            tokens=usage.total_tokens,
            key_index=key_index,
        )

        v_ok, v_reason = (
            self.verifier.verify(req, profile, body) if self.verifier else (True, "")
        )
        outcome, outcome_source, outcome_detail = map_outcome(
            v_ok, v_reason, status="ok"
        )
        stat = "truncated" if v_reason == "truncated" else ("ok" if v_ok else "error")
        reasoning_tokens = (
            body.get("usage", {})
            .get("completion_tokens_details", {})
            .get("reasoning_tokens")
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
    ) -> Attempt:
        """A whole candidate: retries on 5xx and key rotation on 429."""
        o = cand.offering
        est = profile.est_total_tokens
        last: Attempt | None = None
        tried: set[int] = set()
        first_call = True

        for _ in range(max(1, o.key_count)):
            key_index = self.ledger.pick_key(o, est)
            if key_index is None or key_index in tried:
                # No free keys left: down the ladder.
                break
            tried.add(key_index)

            for attempt in range(self.retries_per_provider):
                att_no = (meta.attempts if meta else 1) or 1
                if first_call and backup is not None and self._hedge_allowed(req):
                    res = await self._call_with_hedge(
                        o, req, profile, key_index, backup, meta, attempt_no=att_no
                    )
                else:
                    res = await self._call(
                        o, req, profile, key_index, meta=meta, attempt_no=att_no
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
                    self.ledger.penalize(o, exc.retry_after_s, key_index)
                    self.breaker.on_failure(o.provider)
                    break  # this key is out — try the next
                self.breaker.on_failure(o.provider)
                if attempt + 1 < self.retries_per_provider:
                    await self._backoff(attempt)
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
        return max(HEDGE_MIN_DELAY_S, o.ttft_p50_ms * HEDGE_TTFT_FACTOR / 1000.0)

    def _hedge_allowed(self, req: ChatRequest) -> bool:
        """A hedge is a second quota spend. Allowed for interactive and
        only within budget: 5% of requests, no more."""
        if not self.enable_hedging or req.latency_class != "interactive":
            return False
        done = self._counters["requests"]
        if done < 20:  # too early to judge the share
            return self._counters["hedges"] == 0
        return self._counters["hedges"] / done < self.hedge_budget

    async def _call_with_hedge(
        self,
        o: Offering,
        req: ChatRequest,
        profile: RequestProfile,
        key_index: int,
        backup: Candidate,
        meta: RouterMeta | None,
        attempt_no: int = 1,
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
            )
        )
        delay = self._hedge_delay(o)
        done, _ = await asyncio.wait({primary}, timeout=delay)
        if primary in done:
            return primary.result()

        self._counters["hedges"] += 1
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
        candidates = plan[: self.max_attempts]
        last: ProviderError | None = None

        for i, cand in enumerate(candidates):
            meta.attempts += 1
            meta.fallback_path.append(cand.offering.key)
            backup = candidates[i + 1] if i + 1 < len(candidates) else None
            res = await self._try_candidate(cand, req, profile, backup, meta)

            if res.ok and res.response is not None:
                o = res.offering or cand.offering
                meta.provider = o.provider
                meta.model = o.model_id
                meta.key_index = res.key_index
                meta.latency_ms = int((time.perf_counter() - started) * 1000)
                res.response.router = meta
                res.response.model = o.key

                # Exploration: run a duplicate background request to an alternative candidate
                if (
                    self.enable_exploration
                    and random.random() < self.explore_rate
                    and len(plan) > 1
                ):
                    alt_candidates = [c for c in plan if c.offering.key != o.key]
                    if alt_candidates:
                        chosen_alt = random.choice(alt_candidates)
                        asyncio.create_task(
                            self._run_exploration(
                                chosen_alt, req, profile, meta.request_id or ""
                            )
                        )

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
            self.ledger.reserve(o, est, key_index)
            first_byte_sent = False
            ttft_budget = self.timeout_s if o.is_local else max(25.0, self._hedge_delay(o) * 5)
            t0 = time.perf_counter()
            try:
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
                        log.debug(
                            "stream %s ttft=%.0fms",
                            o.key,
                            (time.perf_counter() - t0) * 1000,
                        )
                    self._sniff_stream_cost(o, chunk)
                    yield chunk
                self.telemetry.log_attempt(
                    o.key, profile, ok=True, verdict="ok_stream", key_index=key_index
                )
                self._bandit_update(o, profile, ok=True)
                return
            except ProviderError as exc:
                last = exc
                self.ledger.commit(o, est, 0, key_index)
                self.breaker.on_failure(o.provider)
                self._bandit_update(o, profile, ok=False)
                if exc.verdict is Verdict.SWITCH:
                    self.ledger.penalize(o, exc.retry_after_s, key_index)
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
            "hedge_rate": round(
                self._counters["hedges"] / self._counters["requests"], 3
            )
            if self._counters["requests"]
            else 0.0,
        }
