"""OpenAI-compatible facade.

Listens on 127.0.0.1 by default. To access from other devices
bind to the Tailscale interface, not to 0.0.0.0: provider keys
live on this machine and must not be exposed.

Layer separation (the hot path top-down):

    ingress → policy → cache L0 → cache L2 → router → executor → adapter
                 ↓                                        ↓
            PII / guard                            degradation ladder

The control plane (discovery, probe, health, purge, batch) runs on the side and
writes to the registry. It does not affect the hot path — only the decisions
the router makes on already-prepared data.
"""
from __future__ import annotations

import asyncio
import base64 as _b64
import json
import logging
import os as _os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    StreamingResponse,
)
from starlette.types import ASGIApp, Receive, Scope, Send

import uuid

from . import pricing, sysmem
from .bandit import Bandit
from .batch import BatchQueue, BatchWorker, QuotaExhausted
from .breaker import CircuitBreaker
from .cache import ExactCache, SemanticCache, cacheable, canonical_key
from .classify import classify, classify_l0
from .compress import Compressor
from .config import Settings, load_env
from .control import ControlPlane
from .discovery import discover_local_backends, run_discovery
from .embeddings import EmbeddingService
from .executor import Executor
from .guard import Guard
from .idempotency import IdempotencyStore
from .ledger import Ledger
from .lock import InstanceLock, InstanceLockError
from .metrics import Metrics, gauges_from
from .policy import Policy
from .ensemble import fanout_ensemble, self_consistency_sample
from .predictor import ModelQualityPredictor
from .prefix import prefix_hash
from .latency import LatencyTracker
from .ratelimit import RateGovernor
from .probe import apply_to_registry, latest as latest_probes
from .probe import probe_all, store as store_probes, summarize_by_tier
from .providers.openai_compat import OpenAICompatAdapter
from .registry import Registry
from .rerank import Reranker
from .responses import ResponsesStreamTranslator, chat_to_response, responses_to_chat
from .router import Router
from .schemas import ChatRequest, RouterError, RouterMeta, Tier
from .telemetry import AttemptLogEntry, Telemetry
from .ui import render_chat_html, render_dashboard_html, render_setup_html
from .verify import Verifier

log = logging.getLogger("waypost.server")
settings = Settings()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# --------------------------------------------------------------- lifespan


@asynccontextmanager
async def lifespan(app: FastAPI):
    global settings
    settings = Settings()
    setup_logging(settings.log_level)
    load_env()  # keys from .env → os.environ (the registry reads them)

    # AR-06: a second instance would split quota accounting. The lock is taken before
    # any SQLite opens, so a competitor cannot write anything in between.
    lock = InstanceLock(settings.db_path.with_suffix(".lock"))
    try:
        lock.acquire()
    except InstanceLockError as exc:
        log.error(str(exc))
        raise SystemExit(3) from exc
    app.state.lock = lock

    registry = Registry.from_manifest(settings.manifest_path)
    # Measurement beats declaration: past probe results
    # override what is written in the manifest.
    if probes := latest_probes(settings.db_path):
        applied = apply_to_registry(registry, probes)
        log.info("probe: applied to %d offerings", applied)

    if settings.free_only:
        paid = [o for o in registry.usable() if not o.free]
        if paid:
            log.info(
                "free_only: excluded paid offerings — %d (%s)",
                len(paid),
                ", ".join(sorted(o.key for o in paid)[:5]),
            )
        unknown = [
            o
            for o in registry.usable()
            if o.free_source == pricing.Source.UNKNOWN.value
        ]
        if unknown:
            log.warning(
                "paidness is not declared for %d offerings — treating them "
                "as paid: %s",
                len(unknown),
                ", ".join(sorted(o.key for o in unknown)[:5]),
            )

    ledger = Ledger(settings.db_path)
    for o in registry.all():
        ledger.register(o)

    telemetry = Telemetry(settings.db_path)
    for key, rate in telemetry.success_rates().items():
        registry.apply_probe(key, success_rate=rate)

    breaker = CircuitBreaker()
    bandit = Bandit(settings.db_path) if settings.enable_bandit else None
    # Shared between router and executor: the executor measures, the router
    # scores. Two copies would mean the router reading numbers nobody writes.
    latency = LatencyTracker()
    for o in registry.all():
        latency.seed(o.key, o.ttft_p50_ms)
    rate_governor = RateGovernor(ledger)
    inflight: dict[str, int] = {}
    # A connection pool per process: TLS reuse noticeably cuts TTFT.
    client = httpx.AsyncClient(
        limits=httpx.Limits(max_connections=64, max_keepalive_connections=32)
    )
    adapter = OpenAICompatAdapter(client)

    # Do not accept traffic with manifest-only local availability.  A local
    # process must answer /models and expose the configured model first.
    startup_local_discovery = await discover_local_backends(adapter, registry)

    app.state.settings = settings
    app.state.registry = registry
    app.state.ledger = ledger
    app.state.breaker = breaker
    app.state.bandit = bandit
    app.state.telemetry = telemetry
    app.state.client = client
    app.state.cache = ExactCache(settings.db_path, settings.cache_ttl_s)
    app.state.semantic = SemanticCache(
        enabled=settings.enable_semantic_cache,
        threshold=settings.semantic_threshold,
        db_path=settings.db_path,
        ttl_s=settings.cache_ttl_s,
        rerank_threshold=settings.rerank_threshold,
    )
    app.state.verifier = Verifier()
    app.state.policy = Policy(
        enable_pii=settings.enable_pii,
        guard=Guard(
            enabled=settings.enable_guard,
            scope=settings.guard_scope,  # type: ignore[arg-type]
            action=settings.guard_action,  # type: ignore[arg-type]
            neutralize=settings.guard_neutralize,
        ),
    )
    app.state.compressor = Compressor(
        mode=settings.compress_mode,  # type: ignore[arg-type]
        min_tokens=settings.compress_min_tokens,
        rate=settings.compress_rate,
    )
    app.state.idempotency = IdempotencyStore(
        settings.db_path, settings.idempotency_ttl_s
    )
    app.state.metrics = Metrics()
    app.state.metrics.register_gauges(
        gauges_from(
            ledger,
            breaker,
            {"exact": app.state.cache, "semantic": app.state.semantic},
            latency=latency,
            rate_governor=rate_governor,
            registry=registry,
            inflight=inflight,
        )
    )
    predictor = ModelQualityPredictor()
    app.state.predictor = predictor
    app.state.latency = latency
    app.state.rate_governor = rate_governor
    app.state.inflight = inflight
    app.state.router = Router(
        registry,
        ledger,
        breaker,
        bandit,
        predictor=predictor,
        free_only=settings.free_only,
        latency=latency,
        inflight=inflight,
        stochastic=settings.stochastic_routing,
    )
    app.state.executor = Executor(
        adapter,
        ledger,
        breaker,
        telemetry,
        verifier=app.state.verifier,
        max_attempts=settings.max_attempts,
        timeout_s=settings.request_timeout_s,
        bandit=bandit,
        enable_hedging=settings.enable_hedging,
        hedge_budget=settings.hedge_budget,
        enable_exploration=settings.enable_exploration,
        explore_rate=settings.explore_rate,
        latency=latency,
        rate_governor=rate_governor,
        inflight=inflight,
        deadlines={
            "interactive": settings.deadline_interactive_s,
            "code_completion": settings.deadline_code_s,
            "batch": settings.deadline_batch_s,
            "reasoning": settings.deadline_reasoning_s,
        },
    )

    # Phase A: eager embedding loading on startup
    embeddings = EmbeddingService(
        model_name=settings.embed_model,
        window_ms=settings.embed_micro_batch_ms,
        max_batch=settings.embed_max_batch,
    )
    log.info("embeddings: backend %s (dim=%d)", embeddings.backend, embeddings.dim)
    app.state.embeddings = embeddings
    app.state.reranker = None
    app.state.discovery = startup_local_discovery

    # ------------------------------------------------------ control plane
    control = ControlPlane()
    app.state.control = control

    async def job_discovery():
        results = await run_discovery(adapter, registry)
        app.state.discovery = results
        for r in results:
            if r.get("added"):
                log.info("discovery %s: +%s", r["provider"], r["added"])
                for o in registry.all():
                    ledger.register(o)
        return results

    async def job_probe():
        # Three populations, not one:
        #   usable     — keep the measurements fresh
        #   candidate  — just discovered, must be measured before it may
        #                take a single user request
        #   quarantine — resurrection: a model removed on a bad day gets
        #                one call a day to prove it is back. Without this
        #                the pool only ever shrinks.
        targets = [o for o in registry.usable() if not o.is_local]
        targets += [o for o in registry.by_lifecycle("candidate") if not o.is_local]
        targets += [
            o
            for o in registry.quarantined(settings.resurrect_after_h * 3600)
            if not o.is_local
        ]
        if not targets:
            return []
        results = await probe_all(adapter, targets)
        store_probes(settings.db_path, results)
        applied = apply_to_registry(registry, {r["key"]: r for r in results})
        log.info("probe: measured %d, applied %d", len(results), applied)
        return results

    async def job_health():
        """Refresh reliability and live availability of local runtimes."""
        rates = telemetry.success_rates(window_s=86_400)
        for key, rate in rates.items():
            registry.apply_probe(key, success_rate=rate)
        # Persist the measured latency into the registry: it is what the
        # manifest constant was always meant to approximate.
        for key, st in latency.snapshot().items():
            if st["samples"] >= 5:
                registry.apply_probe(key, ttft_p50_ms=st["ema_ms"])
        # Graduation: a shadow offering that has carried real traffic well
        # earns full weight. Demotion the other way: an active one that
        # stopped answering goes back to shadow rather than out of the
        # pool, so it keeps a small share and can prove itself again.
        promoted, demoted = [], []
        counts = telemetry.attempt_counts(window_s=86_400)
        for o in registry.by_lifecycle("shadow", "active"):
            n, rate = counts.get(o.key, (0, 1.0))
            if o.lifecycle == "shadow" and n >= 50 and rate >= 0.90:
                if registry.transition(o.key, "active", f"{n} ok at {rate:.0%}"):
                    promoted.append(o.key)
            elif o.lifecycle == "active" and n >= 20 and rate < 0.80:
                if registry.transition(o.key, "shadow", f"success rate {rate:.0%}"):
                    demoted.append(o.key)
        local_results = await discover_local_backends(adapter, registry)
        return {
            "updated": len(rates),
            "promoted": promoted,
            "demoted": demoted,
            "breakers": breaker.snapshot(),
            "rate_limits": rate_governor.snapshot(),
            "latency": latency.snapshot(),
            "local_backends": local_results,
        }

    async def job_purge():
        removed = app.state.idempotency.purge()
        return {"idempotency_removed": removed}

    control.add(
        "discovery",
        job_discovery,
        settings.discovery_interval_h * 3600,
        initial_delay_s=1.0,
        enabled=settings.enable_discovery,
    )
    control.add(
        "probe",
        job_probe,
        settings.probe_interval_h * 3600,
        initial_delay_s=60.0,
        enabled=settings.enable_probe,
    )
    control.add(
        "health", job_health, settings.health_interval_min * 60, initial_delay_s=30.0
    )
    control.add(
        "purge", job_purge, settings.purge_interval_h * 3600, initial_delay_s=300.0
    )
    control.start()

    # ------------------------------------------------------------- batch
    app.state.batch = BatchQueue(settings.db_path) if settings.enable_batch else None
    worker = None
    if app.state.batch is not None:
        worker = BatchWorker(
            app.state.batch, _batch_runner(app), concurrency=settings.batch_concurrency
        )
        worker.start()
    app.state.batch_worker = worker

    try:
        yield
    finally:
        await control.stop()
        if worker is not None:
            await worker.stop()
        if app.state.embeddings is not None:
            await app.state.embeddings.aclose()
        app.state.lock.release()
        await client.aclose()


class PathNormalizationMiddleware:
    """Normalizes whitespace and duplicate slashes/prefixes in request paths.

    Forgives accidental client URLs like:
      - /v1 /chat/completions -> /v1/chat/completions
      - /v1%20/models -> /v1/models
      - /v1/v1/chat/completions -> /v1/chat/completions (duplicated /v1)
      - /chat/completions -> /v1/chat/completions (missing /v1 prefix)
      - /models -> /v1/models
    """

    def __init__(self, app_: ASGIApp):
        self.app = app_

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] in ("http", "websocket"):
            path = scope.get("path", "")
            parts = [p.strip() for p in path.split("/") if p.strip()]
            while len(parts) >= 2 and parts[0] == "v1" and parts[1] == "v1":
                parts.pop(0)
            if parts:
                normalized = "/" + "/".join(parts)
                if parts[0] in (
                    "chat",
                    "models",
                    "embeddings",
                    "rerank",
                    "responses",
                    "pricing",
                    "discovery",
                    "jobs",
                    "stats",
                    "batches",
                ):
                    normalized = "/v1" + normalized
                if normalized != path:
                    scope["path"] = normalized
                    scope["raw_path"] = normalized.encode("latin-1")
        await self.app(scope, receive, send)


app = FastAPI(title="waypost", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(PathNormalizationMiddleware)


# ------------------------------------------------------ lazy services


def get_embeddings(app_: FastAPI) -> EmbeddingService:
    """The encoder loads on first use: if the semantic cache and
    /v1/embeddings are not used, paying to load the model is pointless."""
    if app_.state.embeddings is None:
        app_.state.embeddings = EmbeddingService(
            model_name=settings.embed_model,
            window_ms=settings.embed_micro_batch_ms,
            max_batch=settings.embed_max_batch,
        )
        log.info("embeddings: backend %s", app_.state.embeddings.backend)
        if app_.state.embeddings.is_fallback:
            log.warning(
                "embeddings on the lexical fallback: install "
                "model2vec (pip install -e '.[ml]'), otherwise there is nothing "
                "to tune the semantic-cache threshold against"
            )
    return app_.state.embeddings


def get_reranker(app_: FastAPI) -> Reranker:
    if app_.state.reranker is None:
        app_.state.reranker = Reranker(model_name=settings.rerank_model, enabled=True)
        log.info("reranker: backend %s", app_.state.reranker.backend)
        if settings.enable_semantic_rerank:
            app_.state.semantic.reranker = app_.state.reranker.as_verifier()
    return app_.state.reranker


# ---------------------------------------------------------- hot path


def _query_text(req: ChatRequest) -> str:
    return "\n".join(m.content for m in req.messages if isinstance(m.content, str))


def _namespace(profile, req: ChatRequest) -> str:
    """The semantic-cache namespace: compare only the comparable.
    A different system prompt or a different tools set is a different universe."""
    return f"{profile.tier.value}:{profile.language}:{prefix_hash(req)}"


async def _embedding_for(req: ChatRequest, profile):
    """A single request vector, always one-dimensional.

    The embedding has two producers: the L1 classifier returns (D,), and
    the microbatch service — a matrix (N, D). A shape mismatch dropped the
    cosine in the semantic cache, so normalization lives here, in the single
    point where both branches meet.
    """
    if profile.embedding is not None:
        return profile.embedding
    svc = get_embeddings(app)
    try:
        vectors = await svc.aencode([_query_text(req)])
    except Exception as exc:  # noqa: BLE001
        log.warning("embedding not computed: %s", exc)
        return None
    return vectors[0] if len(vectors) else None


async def run_chat(req: ChatRequest, meta: RouterMeta | None = None) -> dict:
    """The full hot path without streaming. Used both by the endpoint and
    the batch worker — the same code, a different latency_class."""
    t0 = time.perf_counter()
    meta = meta or RouterMeta()
    metrics: Metrics = app.state.metrics

    profile = classify(
        req, enable_l1=settings.enable_l1_classifier, head_path=str(settings.head_path)
    )

    # Policy: PII → strict, injections in untrusted blocks → a fence
    # or a refusal. This is a filter, not a score: it cannot be outweighed.
    decision = app.state.policy.apply(req, profile)
    if decision.blocked:
        metrics.inc("waypost_requests_total", status="blocked")
        raise RouterError(decision.block_reason, 400, meta)
    meta.policy = decision.as_meta()
    if decision.guard.tripped:
        log.warning(
            "GUARD %s in untrusted block → %s",
            ",".join(decision.guard.kinds),
            "refusal" if decision.blocked else "fence",
        )
        for kind in decision.guard.kinds:
            metrics.inc("waypost_guard_total", kind=kind)

    # Dynamic-zone compression. After it the token estimate changes —
    # recompute, otherwise the quota reservation would lie.
    comp = app.state.compressor.compress(req)
    if comp.messages_touched:
        fresh = classify_l0(req)
        profile.est_input_tokens = fresh.est_input_tokens
        profile.est_output_tokens = fresh.est_output_tokens
        meta.policy["compression"] = comp.as_dict()
        log.info(
            "COMPRESS %s: %d → %d tokens",
            comp.mode,
            comp.tokens_before,
            comp.tokens_after,
        )

    meta.task_class = profile.task_class
    meta.complexity_tier = profile.tier.value
    meta.classifier_source = profile.classifier_source

    # L0: exact cache. Cheaper than any routing, so before it.
    key = canonical_key(req, model_class=profile.tier.value)
    if cacheable(req) and (hit := app.state.cache.get(key)) is not None:
        saved = pricing.calculate_savings(
            hit.get("usage", {}).get("prompt_tokens", 0),
            hit.get("usage", {}).get("completion_tokens", 0),
            profile.tier.value,
        )
        lat_ms = int((time.perf_counter() - t0) * 1000)
        req_id = req.session_id or f"req_{uuid.uuid4().hex[:8]}"
        hit["router"] = {
            **hit.get("router", {}),
            "cache": "exact",
            "routing_source": "cache",
            "request_id": req_id,
            "latency_ms": lat_ms,
            "routing_profile": req.profile,
            "saved_usd": saved,
        }
        log.info(
            "CACHE exact hit task=%s tier=%s", profile.task_class, profile.tier.value
        )
        metrics.inc("waypost_cache_hits_total", level="exact")
        metrics.inc("waypost_requests_total", status="cache")
        entry = AttemptLogEntry(
            request_id=req_id,
            attempt_no=1,
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
            l1_prediction=None,
            input_tokens=hit.get("usage", {}).get("prompt_tokens", 0),
            modality="text",
            image_count=0,
            audio_duration_s=0.0,
            vision_token_budget=None,
            provider="cache",
            backend="cache",
            model="cache",
            model_version="",
            tier=profile.tier.value,
            thinking_mode=False,
            routing_source="cache",
            is_exploration=False,
            quota_remaining_pct=100.0,
            quota_window_reset_in_s=0,
            quota_binding_limit="requests",
            status="ok",
            error_class="",
            latency_ms=lat_ms,
            ttft_ms=0,
            output_tokens=hit.get("usage", {}).get("completion_tokens", 0),
            reasoning_tokens=None,
            peak_memory_mb=None,
            is_final=True,
            outcome="pass",
            outcome_source="hard_check",
            outcome_detail={"cache": "exact"},
        )
        app.state.telemetry.log_attempt_row(entry)
        return hit

    # L2: semantic cache. A high threshold, entities checked exactly,
    # a cross-encoder confirms (if enabled).
    embedding = None
    namespace = _namespace(profile, req)
    if settings.enable_semantic_cache and cacheable(req):
        get_reranker(app)  # wires the verifier to the cache
        embedding = await _embedding_for(req, profile)
        if embedding is not None:
            hit = app.state.semantic.lookup(_query_text(req), embedding, namespace)
            if hit is not None:
                saved = pricing.calculate_savings(
                    hit.get("usage", {}).get("prompt_tokens", 0),
                    hit.get("usage", {}).get("completion_tokens", 0),
                    profile.tier.value,
                )
                lat_ms = int((time.perf_counter() - t0) * 1000)
                req_id = req.session_id or f"req_{uuid.uuid4().hex[:8]}"
                hit["router"] = {
                    **hit.get("router", {}),
                    "cache": "semantic",
                    "routing_source": "cache",
                    "request_id": req_id,
                    "latency_ms": lat_ms,
                    "routing_profile": req.profile,
                    "saved_usd": saved,
                }
                log.info(
                    "CACHE semantic hit task=%s tier=%s",
                    profile.task_class,
                    profile.tier.value,
                )
                metrics.inc("waypost_cache_hits_total", level="semantic")
                metrics.inc("waypost_requests_total", status="cache")
                entry = AttemptLogEntry(
                    request_id=req_id,
                    attempt_no=1,
                    ts=time.time(),
                    embedding=embedding,
                    embedder_version="minishlab/potion-multilingual-128M",
                    l0_labels={
                        "task_type": profile.task_class,
                        "tier": profile.tier.value,
                        "complexity": profile.complexity,
                        "lang": profile.language,
                        "requires": [c.value for c in profile.required_caps],
                    },
                    l1_prediction=None,
                    input_tokens=hit.get("usage", {}).get("prompt_tokens", 0),
                    modality="text",
                    image_count=0,
                    audio_duration_s=0.0,
                    vision_token_budget=None,
                    provider="cache",
                    backend="cache",
                    model="cache",
                    model_version="",
                    tier=profile.tier.value,
                    thinking_mode=False,
                    routing_source="cache",
                    is_exploration=False,
                    quota_remaining_pct=100.0,
                    quota_window_reset_in_s=0,
                    quota_binding_limit="requests",
                    status="ok",
                    error_class="",
                    latency_ms=lat_ms,
                    ttft_ms=0,
                    output_tokens=hit.get("usage", {}).get("completion_tokens", 0),
                    reasoning_tokens=None,
                    peak_memory_mb=None,
                    is_final=True,
                    outcome="pass",
                    outcome_source="hard_check",
                    outcome_detail={"cache": "semantic"},
                )
                app.state.telemetry.log_attempt_row(entry)
                return hit

    prefix = prefix_hash(req)
    plan = app.state.router.plan(
        req, profile, quality_floor=decision.quality_floor, prefix=prefix
    )
    log.debug("plan: %s", [c.offering.key for c in plan])

    try:
        resp = await app.state.executor.execute(req, profile, plan, meta)
    except RouterError as exc:
        # Step 6 of the ladder: an approximate answer from the cache with a lowered
        # threshold. Marked approximate — it must not be served as fresh.
        if (
            settings.enable_degraded_cache
            and settings.enable_semantic_cache
            and cacheable(req)
        ):
            if embedding is None:
                embedding = await _embedding_for(req, profile)
            if embedding is not None:
                stale = app.state.semantic.lookup_degraded(
                    _query_text(req), embedding, namespace
                )
                if stale is not None:
                    log.warning(
                        "all providers refused → an approximate "
                        "answer from the cache"
                    )
                    stale = dict(stale)
                    stale["router"] = {
                        **stale.get("router", {}),
                        "cache": "semantic_degraded",
                        "approximate": True,
                    }
                    metrics.inc("waypost_requests_total", status="approximate")
                    return stale
        metrics.inc("waypost_requests_total", status="error")
        raise exc

    # Cascade verifier: a lower-tier answer failed its check —
    # escalate first locally with thinking ON, then to a higher tier.
    if settings.enable_verifier:
        ok, reason = app.state.verifier.verify(req, profile, resp.model_dump())
        if not ok:
            log.warning("VERIFY fail reason=%s → escalate", reason)
            meta.escalated = True
            meta.verify_reason = reason
            metrics.inc("waypost_escalations_total", reason=reason)
            app.state.telemetry.log_attempt(
                f"{meta.provider}/{meta.model}",
                profile,
                ok=False,
                verdict="escalated",
                escalated=True,
                verify_reason=reason,
            )

            # Step 1: Thinking escalation on local model if thinking was off
            if not getattr(req, "thinking_mode", False) and plan:
                first_o = plan[0].offering
                if (
                    first_o.is_local
                    or "mlx" in first_o.provider.lower()
                    or "ollama" in first_o.provider.lower()
                ):
                    log.info(
                        "ESCALATE thinking: retrying %s with thinking_mode=True",
                        first_o.key,
                    )
                    req_thinking = req.model_copy(update={"thinking_mode": True})
                    try:
                        thinking_plan = [plan[0]]
                        resp_thinking = await app.state.executor.execute(
                            req_thinking, profile, thinking_plan, meta
                        )
                        t_ok, _ = app.state.verifier.verify(
                            req_thinking, profile, resp_thinking.model_dump()
                        )
                        if t_ok:
                            resp = resp_thinking
                            ok = True
                    except Exception as exc:  # noqa: BLE001
                        log.debug("thinking escalation failed: %s", exc)

            # Step 2: Self-Consistency for verifiable tasks (code, json, math)
            if not ok and settings.enable_ensemble and plan:
                if profile.task_class in ("code", "math", "extraction", "json"):
                    log.info(
                        "ESCALATE self-consistency: sampling %s (n=%d)",
                        plan[0].offering.key,
                        settings.self_consistency_samples,
                    )
                    try:
                        resp_sc = await self_consistency_sample(
                            app.state.executor,
                            req,
                            profile,
                            plan[0],
                            n_samples=settings.self_consistency_samples,
                            verifier=app.state.verifier,
                        )
                        if resp_sc:
                            sc_ok, _ = app.state.verifier.verify(
                                req, profile, resp_sc.model_dump()
                            )
                            if sc_ok:
                                resp = resp_sc
                                ok = True
                    except Exception as exc:  # noqa: BLE001
                        log.debug("self-consistency escalation failed: %s", exc)

            # Step 3: Tier escalation to higher tier if still not ok
            if not ok and profile.tier is not Tier.L:
                nxt = {Tier.S: Tier.M, Tier.M: Tier.L}[profile.tier]
                plan2 = app.state.router.plan_escalated(req, profile, nxt)
                if plan2:
                    resp = await app.state.executor.execute(req, profile, plan2, meta)
                    ok, _ = app.state.verifier.verify(req, profile, resp.model_dump())

            # Step 4: Fan-out ensemble across diverse families if still failing
            if not ok and settings.enable_ensemble and len(plan) > 1:
                log.info(
                    "ESCALATE fan-out: dispatching diverse ensemble across %d models",
                    len(plan),
                )
                try:
                    resp_fan = await fanout_ensemble(
                        app.state.executor,
                        req,
                        profile,
                        plan,
                        max_proposers=3,
                        fanout_timeout_ms=settings.fanout_timeout_ms,
                    )
                    if resp_fan:
                        resp = resp_fan
                except Exception as exc:  # noqa: BLE001
                    log.debug("fanout escalation failed: %s", exc)

    app.state.router.remember(req.session_id, f"{meta.provider}/{meta.model}", prefix)

    elapsed = (time.perf_counter() - t0) * 1000
    log.info(
        "ROUTE %s task=%s tier=%s src=%s attempts=%d%s path=%s → %s %.0fms",
        meta.model,
        profile.task_class,
        profile.tier.value,
        profile.classifier_source,
        meta.attempts,
        " hedged" if meta.hedged else "",
        ">".join(meta.fallback_path),
        meta.provider,
        elapsed,
    )
    metrics.inc("waypost_requests_total", status="ok")
    metrics.observe("waypost_latency_ms", elapsed, provider=meta.provider or "unknown")
    if meta.hedged:
        metrics.inc("waypost_hedges_total", provider=meta.provider or "?")
    metrics.inc(
        "waypost_tokens_total",
        resp.usage.prompt_tokens,
        provider=meta.provider or "?",
        direction="prompt",
    )
    metrics.inc(
        "waypost_tokens_total",
        resp.usage.completion_tokens,
        provider=meta.provider or "?",
        direction="completion",
    )

    if settings.enable_prompt_log:
        app.state.telemetry.log_prompt(
            _query_text(req),
            profile,
            offering=f"{meta.provider}/{meta.model}",
            tokens=resp.usage.total_tokens,
            ok=True,
            escalated=meta.escalated,
        )

    meta.routing_profile = req.profile
    meta.saved_usd = pricing.calculate_savings(
        resp.usage.prompt_tokens,
        resp.usage.completion_tokens,
        profile.tier.value,
    )

    body = resp.model_dump()
    if cacheable(req):
        app.state.cache.put(key, body)
        # Only verified answers go into the semantic cache: garbage
        # that got in would be returned over and over.
        if (
            settings.enable_semantic_cache
            and embedding is not None
            and not meta.escalated
        ):
            app.state.semantic.put(_query_text(req), embedding, namespace, body)
    return body


def _batch_runner(app_: FastAPI):
    """Executor of one batch item. The same hot path, but with
    latency_class=batch: speed does not matter, not burning quota does."""

    async def runner(body: dict[str, Any]) -> dict[str, Any]:
        payload = dict(body)
        payload.setdefault("model", "auto")
        payload["latency_class"] = "batch"
        payload["stream"] = False
        req = ChatRequest(**payload)
        try:
            return await run_chat(req)
        except RouterError as exc:
            if exc.status_code == 503:
                # No candidates — quota. The item returns to the queue.
                raise QuotaExhausted(exc.message) from exc
            raise

    return runner


# ------------------------------------------------------------- endpoints


@app.exception_handler(RouterError)
async def _router_error(_: Request, exc: RouterError):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {"message": exc.message, "type": "router_error"},
            "router": exc.meta.model_dump(),
        },
    )


@app.post("/v1/active-model")
async def set_active_model(req: Request):
    try:
        body = await req.json()
        model_val = body.get("model", "auto")
        mode_val = body.get("mode", "auto")
        app.state.active_model = model_val
        app.state.active_mode = mode_val
        return {
            "status": "ok",
            "model": app.state.active_model,
            "mode": app.state.active_mode,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.get("/health")
async def health():
    active_model = getattr(app.state, "active_model", "auto")
    active_mode = getattr(app.state, "active_mode", "auto")

    last_mode = active_mode
    last_model = active_model

    if active_model == "auto":
        last_model = "auto (Smart Router)"
    elif active_model.startswith("Tier"):
        last_model = active_model

    recent = []
    try:
        recent = app.state.telemetry.attempt_logs(limit=1)
    except Exception:
        pass

    if recent and active_model in (
        "auto",
        "auto (Smart Router)",
        "Tier S",
        "Tier M",
        "Tier L",
    ):
        r = recent[0]
        model_name = r.get("model") or r.get("backend") or ""
        prov = r.get("provider") or r.get("backend") or ""
        is_loc = (
            prov.lower() in ("mlx", "local", "lmstudio", "llamacpp")
            or "mlx" in model_name.lower()
            or (prov.lower() == "ollama" and not model_name.lower().startswith("cloud"))
        )
        last_mode = "local" if is_loc else "cloud"
        if model_name:
            last_model = f"auto ({model_name})"

    return {
        "status": "ok",
        "last_mode": last_mode,
        "last_model": last_model,
        "breakers": app.state.breaker.snapshot(),
    }


def _get_providers_data(registry: Registry) -> dict[str, Any]:
    from . import keys

    all_offerings = registry.all()
    prov_list = []
    configured_keys = 0

    for p in keys.PROVIDER_DEFS:
        p_id = p["id"]
        env_var = p["env_var"]
        alt_vars = p.get("alt_env_vars", [])
        active_key = keys.get_active_key(env_var, alt_vars)
        has_key = bool(active_key)
        if has_key:
            configured_keys += 1

        prov_offerings = [
            o
            for o in all_offerings
            if o.provider.lower() == p_id.lower()
            or (p_id == "gemini" and o.provider.lower() in ("gemini", "google"))
            or (p_id == "github" and o.provider.lower() in ("github", "azure"))
        ]
        usable_offerings = [o for o in prov_offerings if o.usable]

        prov_list.append(
            {
                "id": p_id,
                "name": p["name"],
                "env_var": env_var,
                "alt_env_vars": alt_vars,
                "base_url": p["base_url"],
                "description": p["description"],
                "doc_url": p["doc_url"],
                "has_key": has_key,
                "masked_key": keys.mask_key(active_key),
                "models_count": len(prov_offerings),
                "usable_models_count": len(usable_offerings),
            }
        )

    usable_cloud_models = sum(1 for o in all_offerings if not o.is_local and o.usable)

    return {
        "summary": {
            "total_providers": len(keys.PROVIDER_DEFS),
            "configured_keys": configured_keys,
            "usable_cloud_models": usable_cloud_models,
        },
        "providers": prov_list,
    }


@app.get("/providers", response_class=HTMLResponse)
async def providers_page():
    from .ui import render_providers_html

    data = _get_providers_data(app.state.registry)
    return HTMLResponse(render_providers_html(data))


@app.get("/v1/providers/keys")
async def list_provider_keys():
    data = _get_providers_data(app.state.registry)
    return data


@app.post("/v1/providers/keys")
async def save_provider_key(payload: dict = Body(...)):
    from . import keys

    env_var = payload.get("env_var", "").strip()
    api_key = payload.get("api_key", "").strip()
    prov_id = payload.get("provider", "").strip()

    if not env_var:
        raise HTTPException(status_code=400, detail="env_var is required")

    if not api_key:
        # An empty field means "leave it alone". It used to mean "delete",
        # and the UI clears the input after every save — so one extra click
        # on Save silently destroyed the key. Deletion has its own verb.
        raise HTTPException(
            status_code=400,
            detail=f"No key provided for {env_var}. "
            "To remove it, use the Remove action (DELETE /v1/providers/keys).",
        )

    in_keychain = keys.save_key(env_var, api_key)

    # Calculate new usable models count for response
    prov_offerings = [
        o
        for o in app.state.registry.all()
        if o.provider.lower() == prov_id.lower()
        or (prov_id == "gemini" and o.provider.lower() in ("gemini", "google"))
        or (prov_id == "github" and o.provider.lower() in ("github", "azure"))
    ]
    usable_count = sum(1 for o in prov_offerings if o.usable)

    return {
        "status": "ok",
        "message": f"API key for {env_var} saved to .env"
        + (" and Keychain" if in_keychain else " (Keychain unavailable)"),
        "masked_key": keys.mask_key(api_key),
        "usable_count": usable_count,
    }


@app.delete("/v1/providers/keys")
async def delete_provider_key(payload: dict = Body(...)):
    from . import keys

    env_var = payload.get("env_var", "").strip()
    if not env_var:
        raise HTTPException(status_code=400, detail="env_var is required")

    keys.delete_key(env_var)

    return {
        "status": "ok",
        "message": f"API key for {env_var} removed from .env, "
        "the Keychain and the running process",
    }


def _get_local_backends(registry: Registry) -> list[dict]:
    from .discovery import LOCAL_DEFAULTS

    local_offerings = [o for o in registry.all() if o.is_local]
    backends = []
    for name, url_getter in LOCAL_DEFAULTS:
        url = url_getter()
        prov_models = [o for o in local_offerings if o.provider.lower() == name.lower()]
        active = any(o.usable for o in prov_models)
        backends.append(
            {
                "name": name,
                "url": url,
                "models_count": len(prov_models),
                "status": "online" if active else "offline",
                "models": [o.model_id for o in prov_models],
            }
        )

    known_names = {b["name"].lower() for b in backends}
    for o in local_offerings:
        if o.provider.lower() not in known_names:
            known_names.add(o.provider.lower())
            prov_models = [
                x for x in local_offerings if x.provider.lower() == o.provider.lower()
            ]
            backends.append(
                {
                    "name": o.provider,
                    "url": o.base_url,
                    "models_count": len(prov_models),
                    "status": "online"
                    if any(x.usable for x in prov_models)
                    else "offline",
                    "models": [x.model_id for x in prov_models],
                }
            )
    return backends


@app.get("/v1/models")
async def list_models(request: Request):
    """One virtual model for programmatic clients.
    A browser gets a visual catalog of cloud & local models and providers.
    """
    offerings = app.state.registry.usable()
    if settings.free_only:
        offerings = [o for o in offerings if o.free]
    # ponytail: expose a single virtual model so every client targets one id;
    # the per-offering list lives at /v1/pricing and /v1/stats.
    data = [
        {
            "id": "auto",
            "object": "model",
            "owned_by": "waypost",
            "tier": "auto",
            "ctx": max((o.ctx_window for o in offerings), default=0),
            "local": any(o.is_local for o in offerings),
            "trains_on_data": False,
            "free": settings.free_only,
            "free_source": pricing.Source.UNKNOWN.value,
            "free_detail": "router picks the offering",
            "keys": sum(o.key_count for o in offerings),
            "caps": ["tools", "json", "vision", "stream", "chat"],
            "capabilities": {
                "tools": True,
                "function_calling": True,
                "vision": True,
                "streaming": True,
            },
            "supports_tools": True,
        }
    ]
    if "text/html" in request.headers.get("accept", ""):
        registry = app.state.registry
        all_offerings = sorted(
            registry.all(), key=lambda o: (not o.usable, o.provider, o.model_id)
        )
        backends = _get_local_backends(registry)

        models_data = [
            {
                "id": "auto (Virtual Router)",
                "key": "auto",
                "provider": "waypost",
                "tier": "auto",
                "ctx": max((o.ctx_window for o in offerings), default=0),
                "free": True,
                "is_local": False,
                "usable": True,
                "free_source": "builtin",
                "free_detail": "Dynamic routing across all cloud & local providers",
                "selectable": True,
                "caps": ["chat", "stream"],
                "base_url": "http://127.0.0.1:8080/v1",
            }
        ] + [
            {
                "id": o.model_id,
                "key": o.key,
                "provider": o.provider,
                "tier": o.tier.value,
                "ctx": o.ctx_window,
                "free": o.free,
                "is_local": o.is_local,
                "usable": o.usable,
                "free_source": o.free_source,
                "free_detail": o.free_detail or "",
                "selectable": o.usable and (o.free or not settings.free_only),
                "caps": [c.value for c in o.caps],
                "base_url": o.base_url,
            }
            for o in all_offerings
        ]

        payload = {
            "counts": {
                "total": len(models_data) - 1,
                "cloud": sum(1 for m in models_data[1:] if not m["is_local"]),
                "local": sum(1 for m in models_data[1:] if m["is_local"]),
                "free": sum(1 for m in models_data[1:] if m["free"]),
                "paid": sum(1 for m in models_data[1:] if not m["free"]),
                "active_engines": sum(1 for b in backends if b["status"] == "online"),
                "total_engines": len(backends),
            },
            "backends": backends,
            "models": models_data,
        }
        return HTMLResponse(_models_html(payload))
    return {"object": "list", "data": data}


# Logo/favicon for HTML pages. Load the PNG (macos/icon-96.png)
# at startup; if the file is missing — fall back to an inline-SVG signpost.
_LOGO_PNG = "macos/icon-96.png"
if _os.path.exists(_LOGO_PNG):
    with open(_LOGO_PNG, "rb") as _f:
        _LOGO_URI = "data:image/png;base64," + _b64.b64encode(_f.read()).decode()
    _LOGO = f'<img src="{_LOGO_URI}" width=22 height=22 alt="waypost" style="border-radius:4px;vertical-align:-4px">'
    _FAVICON = _LOGO_URI
else:
    _LOGO = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" '
        'viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
        'stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-4px">'
        '<circle cx="12" cy="12" r="10"/>'
        '<polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76"/>'
        "</svg>"
    )
    _FAVICON = "data:image/svg+xml," + _LOGO.replace('"', "'").replace(
        "<", "%3C"
    ).replace(">", "%3E").replace("#", "%23").replace(" ", "%20")


def _claude_theme_css() -> str:
    """Pure minimalist monochrome grayscale design system."""
    return """
:root {
  --bg: #FAFAFA;
  --bg-subtle: #F4F4F5;
  --card: #FFFFFF;
  --card-hover: #F4F4F5;
  --text: #09090B;
  --text-secondary: #52525B;
  --text-muted: #A1A1AA;
  --border: #E4E4E7;
  --border-subtle: #F4F4F5;
  --border-focus: #09090B;
  --accent: #09090B;
  --accent-hover: #27272A;
  --accent-fg: #FAFAFA;
  --accent-bg: rgba(9, 9, 11, 0.05);
  --font-sans: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", Roboto, sans-serif;
  --font-mono: ui-monospace, "SF Mono", Menlo, Monaco, Consolas, monospace;
  --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.04);
  --shadow-md: 0 4px 16px rgba(0, 0, 0, 0.05);
  --radius-sm: 6px;
  --radius-md: 10px;
  --radius-lg: 14px;
}

@media (prefers-color-scheme: dark) {
  :root {
    --bg: #09090B;
    --bg-subtle: #141416;
    --card: #18181B;
    --card-hover: #202024;
    --text: #FAFAFA;
    --text-secondary: #A1A1AA;
    --text-muted: #71717A;
    --border: #27272A;
    --border-subtle: #1E1E22;
    --border-focus: #FAFAFA;
    --accent: #FAFAFA;
    --accent-hover: #E4E4E7;
    --accent-fg: #09090B;
    --accent-bg: rgba(250, 250, 250, 0.08);
    --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.3);
    --shadow-md: 0 4px 16px rgba(0, 0, 0, 0.4);
  }
}

* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font: 14px/1.55 var(--font-sans);
  -webkit-font-smoothing: antialiased;
}
a { color: var(--text); text-decoration: none; }
a:hover { opacity: 0.8; }

.claude-header {
  display: grid;
  grid-template-columns: 1fr auto 1fr;
  align-items: center;
  padding: 8px 24px;
  background: var(--card);
  border-bottom: 1px solid var(--border);
  position: sticky;
  top: 0;
  z-index: 100;
}
.macos-app .claude-header,
html.macos-app .claude-header,
body.macos-app .claude-header {
  padding-left: 20px;
  -webkit-app-region: drag;
  user-select: none;
}
.macos-app .claude-header .header-left {
  width: 72px;
}
.claude-header .header-left {
  display: flex;
  align-items: center;
  grid-column: 1;
}
.claude-header .brand-center,
.claude-header .brand {
  display: flex;
  align-items: center;
  justify-content: center;
  text-decoration: none;
  color: var(--text);
  grid-column: 2;
}
.brand-name, .brand-title {
  font-family: var(--font-sans);
  font-size: 15px;
  font-weight: 700;
  letter-spacing: -0.01em;
  color: var(--text);
}
.claude-header .nav-tabs {
  display: flex;
  gap: 4px;
  background: var(--bg-subtle);
  padding: 3px;
  border-radius: var(--radius-sm);
  justify-self: end;
  grid-column: 3;
}
.macos-app .claude-header .brand,
.macos-app .claude-header .brand-center,
.macos-app .claude-header .nav-tabs,
.macos-app .claude-header a,
.macos-app .claude-header button,
.macos-app .claude-header input,
.macos-app .claude-header select {
  -webkit-app-region: no-drag;
}
.brand-badge {
  font-family: var(--font-mono);
  font-size: 10px;
  font-weight: 500;
  padding: 2px 6px;
  border-radius: var(--radius-sm);
  background: var(--bg-subtle);
  color: var(--text-secondary);
  border: 1px solid var(--border);
  text-transform: uppercase;
  letter-spacing: 0.04em;
}
.nav-tabs {
  display: flex;
  gap: 3px;
  align-items: center;
  background: var(--bg-subtle);
  padding: 3px;
  border-radius: var(--radius-sm);
  border: 1px solid var(--border);
}
.nav-tab {
  padding: 5px 12px;
  border-radius: 4px;
  color: var(--text-secondary);
  text-decoration: none;
  font-size: 12.5px;
  font-weight: 500;
  transition: all 0.15s ease;
}
.nav-tab:hover {
  color: var(--text);
  text-decoration: none;
}
.nav-tab.active {
  background: var(--card);
  color: var(--text);
  font-weight: 600;
  box-shadow: var(--shadow-sm);
}
.page-container {
  max-width: 1080px;
  margin: 0 auto;
  padding: 32px 24px;
}
h1.page-title {
  font-family: var(--font-sans);
  font-size: 20px;
  font-weight: 600;
  letter-spacing: -0.02em;
  margin: 0 0 6px;
  color: var(--text);
}
p.page-sub {
  color: var(--text-secondary);
  margin: 0 0 24px;
  font-size: 13px;
  line-height: 1.5;
}
p.page-sub code {
  background: var(--bg-subtle);
  border: 1px solid var(--border);
  padding: 2px 6px;
  border-radius: var(--radius-sm);
  font-family: var(--font-mono);
  font-size: 11.5px;
  color: var(--text);
}
.search-input {
  width: 100%;
  max-width: 320px;
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 7px 12px;
  font-size: 13px;
  color: var(--text);
  outline: none;
  font-family: var(--font-sans);
  transition: border-color 0.15s ease;
}
.search-input:focus {
  border-color: var(--border-focus);
}
.summary-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 12px;
  margin-bottom: 24px;
}
.stat-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  padding: 16px 18px;
  box-shadow: var(--shadow-sm);
}
.stat-card b {
  font-size: 22px;
  font-family: var(--font-sans);
  font-weight: 600;
  letter-spacing: -0.02em;
  display: block;
  color: var(--text);
  margin-bottom: 2px;
}
.stat-card span {
  color: var(--text-secondary);
  font-size: 12px;
  font-weight: 500;
}
.claude-table-wrap {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  overflow: hidden;
  box-shadow: var(--shadow-sm);
}
table.claude-table {
  width: 100%;
  border-collapse: collapse;
  text-align: left;
  font-size: 13px;
}
table.claude-table th, table.claude-table td {
  padding: 11px 16px;
  border-bottom: 1px solid var(--border-subtle);
  vertical-align: middle;
}
table.claude-table th {
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: .05em;
  color: var(--text-muted);
  font-weight: 600;
  background: var(--bg-subtle);
  border-bottom: 1px solid var(--border);
}
table.claude-table tr:last-child td { border-bottom: none; }
table.claude-table tr:hover td { background: var(--card-hover); }

.id { font-family: var(--font-mono); font-size: 12.5px; font-weight: 500; }
.detail { color: var(--text-secondary); font-size: 12px; max-width: 340px; }
.badge {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 2px 7px;
  border-radius: var(--radius-sm);
  font-size: 11px;
  font-weight: 500;
  border: 1px solid var(--border);
  background: var(--bg-subtle);
  color: var(--text-secondary);
}
.badge.free { color: var(--text); font-weight: 600; }
.badge.paid { color: var(--text-muted); }
.badge.local { color: var(--text); }

.tier-tag {
  display: inline-flex;
  align-items: center;
  padding: 2px 6px;
  border-radius: var(--radius-sm);
  font-size: 10.5px;
  font-weight: 600;
  border: 1px solid var(--border);
  background: var(--bg-subtle);
  color: var(--text-secondary);
  text-transform: uppercase;
}
.tier-s { background: var(--bg-subtle); color: var(--text-secondary); }
.tier-m { background: var(--bg-subtle); color: var(--text); font-weight: 600; }
.tier-l { background: var(--accent); color: var(--accent-fg); border-color: var(--accent); }
"""


def _nav_header(active: str = "chat") -> str:
    return f"""<header class="claude-header">
  <div class="header-left"></div>
  <a href="/chat" class="brand-center brand">
    <span class="brand-name">Waypost</span>
  </a>
  <nav class="nav-tabs">
    <a href="/chat" class="nav-tab {'active' if active == 'chat' else ''}">Chat</a>
    <a href="/dashboard" class="nav-tab {'active' if active == 'dashboard' else ''}">Dashboard</a>
    <a href="/providers" class="nav-tab {'active' if active == 'providers' else ''}">Providers</a>
    <a href="/setup" class="nav-tab {'active' if active == 'setup' else ''}">Setup</a>
  </nav>
</header>"""


def _models_html(data: dict) -> str:
    counts = data.get("counts", {})
    backends = data.get("backends", [])
    models = data.get("models", [])

    engine_cards = []
    for b in backends:
        st = b.get("status", "offline")
        st_badge = (
            "<span class='badge free'>online</span>"
            if st == "online"
            else (
                "<span class='badge'>configured</span>"
                if st == "configured"
                else "<span class='badge paid'>offline</span>"
            )
        )
        count = b.get("models_count", 0)
        url = b.get("url", "")
        engine_cards.append(
            f"""
            <div class="stat-card" style="padding:14px 16px">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
                <b style="font-size:14px;margin:0;text-transform:capitalize">{b.get('name')}</b>
                {st_badge}
              </div>
              <div style="font-size:12px;color:var(--text-secondary);margin-bottom:4px">
                <b>{count}</b> models loaded
              </div>
              <div style="font-family:var(--font-mono);font-size:11px;color:var(--text-muted);word-break:break-all">
                {url}
              </div>
            </div>
            """
        )
    engine_cards_html = "".join(engine_cards)

    rows = []
    for m in models:
        free_badge = (
            "<span class='badge free'>free</span>"
            if m.get("free")
            else "<span class='badge paid'>paid</span>"
        )
        local_badge = (
            "<span class='badge local'>local</span>"
            if m.get("is_local")
            else "<span class='badge'>cloud</span>"
        )
        selectable_badge = (
            "<span class='badge free'>active</span>"
            if m.get("selectable")
            else "<span class='badge paid'>standby</span>"
        )
        ctx = f"{m['ctx']//1000}K" if m.get("ctx") else "—"
        detail = m.get("free_detail") or ""
        src = m.get("free_source") or ""
        if src and src not in ("unknown", "builtin", "—"):
            detail = f"<span class='badge' style='font-size:10px;margin-right:4px'>{src}</span> {detail}"
        t = str(m.get("tier", "auto")).lower()
        kind = "local" if m.get("is_local") else "cloud"
        is_free = "true" if m.get("free") else "false"

        rows.append(
            f"<tr data-kind='{kind}' data-free='{is_free}' data-tier='{t}' data-selectable='{str(m.get('selectable')).lower()}'>"
            f"<td class=id>{m['id']}</td>"
            f"<td><span style='font-weight:600'>{m.get('provider', '—')}</span></td>"
            f"<td><span class='tier-tag tier-{t}'>{m.get('tier', '—')}</span></td>"
            f"<td>{ctx}</td>"
            f"<td>{free_badge} {local_badge}</td>"
            f"<td>{selectable_badge}</td>"
            f"<td class=detail>{detail}</td></tr>"
        )

    return f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Waypost — Models & Providers</title>
<link rel=icon href="{_FAVICON}">
<style>
{_claude_theme_css()}
.section-card {{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  padding: 18px 22px;
  margin-bottom: 20px;
  box-shadow: var(--shadow-sm);
}}
.section-card h2 {{
  font-family: var(--font-sans);
  font-size: 11.5px;
  text-transform: uppercase;
  letter-spacing: .05em;
  color: var(--text-muted);
  margin: 0 0 12px;
  font-weight: 600;
}}
.filter-pills {{
  display: flex;
  gap: 6px;
  align-items: center;
  flex-wrap: wrap;
}}
.filter-pill {{
  background: var(--bg-subtle);
  border: 1px solid var(--border);
  color: var(--text-secondary);
  padding: 5px 12px;
  border-radius: var(--radius-sm);
  font-size: 12px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.15s ease;
}}
.filter-pill:hover {{
  color: var(--text);
  border-color: var(--border-focus);
}}
.filter-pill.active {{
  background: var(--card);
  color: var(--text);
  font-weight: 600;
  border-color: var(--text);
  box-shadow: var(--shadow-sm);
}}
</style></head><body>
{_nav_header("models")}
<div class="page-container">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:12px;margin-bottom:20px">
    <div>
      <h1 class="page-title">Models & Providers</h1>
      <p class="page-sub" style="margin-bottom:0">Unified catalog of cloud & local inference engines · <code>curl http://127.0.0.1:8080/v1/models</code></p>
    </div>
    <div style="display:flex;gap:8px;align-items:center">
      <input type="text" id="model-search" placeholder="Search models, providers, notes..." oninput="applyFilters()" class="search-input">
      <a href="/v1/local?refresh=true" style="background:var(--accent);color:var(--accent-fg);padding:7px 13px;border-radius:var(--radius-sm);font-size:12px;font-weight:600;text-decoration:none;white-space:nowrap">↻ Scan Engines</a>
    </div>
  </div>

  <div class="summary-grid">
    <div class="stat-card"><b>{counts.get('total', len(models))}</b><span>total models</span></div>
    <div class="stat-card"><b>{counts.get('cloud', 0)}</b><span>cloud models</span></div>
    <div class="stat-card"><b>{counts.get('active_engines', 0)}/{counts.get('total_engines', len(backends))}</b><span>local engines active</span></div>
    <div class="stat-card"><b>{counts.get('free', 0)}</b><span>free & zero-cost</span></div>
  </div>

  <div class="section-card">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">
      <h2 style="margin:0">Local Inference Engines</h2>
      <span style="font-size:11.5px;color:var(--text-muted)">On-device zero-egress backends</span>
    </div>
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px">
      {engine_cards_html or '<div style="color:var(--text-muted);font-size:12px">No local backends configured</div>'}
    </div>
  </div>

  <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;margin-bottom:14px">
    <div class="filter-pills" id="filter-container">
      <button class="filter-pill active" onclick="setFilter('all', this)">All Models ({counts.get('total', len(models))})</button>
      <button class="filter-pill" onclick="setFilter('cloud', this)">Cloud ({counts.get('cloud', 0)})</button>
      <button class="filter-pill" onclick="setFilter('local', this)">Local ({counts.get('local', 0)})</button>
      <button class="filter-pill" onclick="setFilter('free', this)">Free Tier ({counts.get('free', 0)})</button>
      <button class="filter-pill" onclick="setFilter('tier-s', this)">Tier S</button>
      <button class="filter-pill" onclick="setFilter('tier-m', this)">Tier M</button>
      <button class="filter-pill" onclick="setFilter('tier-l', this)">Tier L</button>
    </div>
  </div>

  <div class="claude-table-wrap">
    <table class="claude-table" id="models-table">
      <thead><tr><th>Model ID</th><th>Provider</th><th>Tier</th><th>Context</th><th>Type / Cost</th><th>Selection</th><th>Verification Notes</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
  </div>
</div>
<script>
let currentFilter = 'all';

function setFilter(filter, el) {{
  currentFilter = filter;
  document.querySelectorAll('#filter-container .filter-pill').forEach(btn => btn.classList.remove('active'));
  if (el) el.classList.add('active');
  applyFilters();
}}

function applyFilters() {{
  const query = (document.getElementById('model-search').value || '').toLowerCase();
  const rows = document.querySelectorAll('#models-table tbody tr');
  
  rows.forEach(r => {{
    const text = r.innerText.toLowerCase();
    const kind = r.getAttribute('data-kind');
    const free = r.getAttribute('data-free');
    const tier = r.getAttribute('data-tier');
    
    let matchFilter = true;
    if (currentFilter === 'cloud') matchFilter = (kind === 'cloud');
    else if (currentFilter === 'local') matchFilter = (kind === 'local');
    else if (currentFilter === 'free') matchFilter = (free === 'true');
    else if (currentFilter === 'tier-s') matchFilter = (tier === 's');
    else if (currentFilter === 'tier-m') matchFilter = (tier === 'm');
    else if (currentFilter === 'tier-l') matchFilter = (tier === 'l');
    
    const matchQuery = text.includes(query);
    r.style.display = (matchFilter && matchQuery) ? '' : 'none';
  }});
}}
</script>
</body></html>"""


def _probe_html(data: dict) -> str:
    summary = data.get("summary", {})
    by_tier = summary.get("by_tier", {})
    probes = data.get("probes", [])
    breakers = data.get("breakers", {})
    active_tier = data.get("filters", {}).get("tier") or "ALL"

    tier_cards = []
    for t_name, t_title in (
        ("S", "Tier S · Small / Fast"),
        ("M", "Tier M · Medium / General"),
        ("L", "Tier L · Large / Reasoning"),
    ):
        ts = by_tier.get(t_name, {})
        tot = ts.get("total", 0)
        hlth = ts.get("healthy", 0)
        deg = ts.get("degraded", 0)
        dead = ts.get("dead", 0)
        nokey = ts.get("no_key", 0)
        avg = f"{ts['avg_ttft_ms']}ms" if ts.get("avg_ttft_ms") else "—"
        tier_cards.append(
            f"""
        <div class="stat-card">
          <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">
            <span class="tier-tag tier-{t_name.lower()}">{t_name}</span>
            <span style="font-weight:600;font-size:13px;color:var(--text)">{t_title}</span>
          </div>
          <div style="margin:6px 0">
            <span style="font-size:22px;font-family:var(--font-sans);font-weight:700;color:var(--text);margin-right:8px">{hlth}/{tot}</span>
            <span style="color:var(--text-secondary);font-size:12px">healthy · avg TTFT {avg}</span>
          </div>
          <div style="color:var(--text-muted);font-size:11px;margin-top:6px">
            degraded: {deg} · dead: {dead} · no key: {nokey}
          </div>
        </div>
        """
        )
    tier_cards_html = "".join(tier_cards)

    brk_badges = []
    for p, st in sorted(breakers.items()):
        is_ok = st in ("closed", "ok", "healthy")
        b_class = "free" if is_ok else "paid"
        brk_badges.append(
            f"<span class='badge {b_class}' style='font-family:var(--font-mono);margin:0 6px 6px 0;padding:4px 9px'>{p}: {st}</span>"
        )
    brk_html = (
        "".join(brk_badges)
        or '<span style="color:var(--text-muted);font-size:12px">all provider circuits nominal</span>'
    )

    rows = []
    for r in probes:
        t = str(r.get("tier", "M")).upper()
        st = str(r.get("status", "unknown")).lower()
        st_badge_class = (
            "free"
            if st in ("healthy", "ok", "online")
            else ("paid" if st in ("dead", "failed", "error", "no_key") else "")
        )
        st_badge = f"<span class='badge {st_badge_class}'>{st}</span>"
        ttft = r.get("ttft_p50_ms")
        mark = "~" if r.get("ttft_estimated") else ""
        lat_display = f"{mark}{round(ttft)}ms" if ttft else "—"

        caps = r.get("caps", [])
        caps_html = " ".join(
            f"<span style='background:var(--bg-subtle);border:1px solid var(--border);color:var(--text-secondary);padding:1px 6px;border-radius:4px;font-size:11px;margin-right:4px'>{c}</span>"
            for c in caps
            if c != "stream"
        )

        err = r.get("error", "")
        sample = r.get("sample", "")
        detail_txt = (
            f"<span style='color:var(--text-muted)'>{err}</span>"
            if err
            else (
                f"<span style='color:var(--text-secondary);font-family:var(--font-mono);font-size:11px'>{sample}</span>"
                if sample
                else "<span style='color:var(--text-secondary)'>OK</span>"
            )
        )

        rows.append(
            f"""
        <tr>
          <td><span class="tier-tag tier-{t.lower()}">{t}</span></td>
          <td class="id">{r.get('model_id')}</td>
          <td><span style="font-weight:600">{r.get('provider')}</span></td>
          <td>{st_badge}</td>
          <td>{lat_display}</td>
          <td>{caps_html or '—'}</td>
          <td class="detail">{detail_txt}</td>
        </tr>
        """
        )
    rows_html = (
        "".join(rows)
        or "<tr><td colspan=7 style='color:var(--text-muted);padding:24px;text-align:center'>no models match filter</td></tr>"
    )

    filters_html = f"""
    <div style="display:flex;gap:6px;align-items:center;margin-bottom:16px;flex-wrap:wrap">
      <div class="nav-tabs" style="padding:2px">
        <a class="nav-tab {'active' if active_tier == 'ALL' else ''}" href="/v1/probe">All ({summary.get('total', 0)})</a>
        <a class="nav-tab {'active' if active_tier == 'S' else ''}" href="/v1/probe?tier=S">Tier S ({by_tier.get('S', {}).get('total', 0)})</a>
        <a class="nav-tab {'active' if active_tier == 'M' else ''}" href="/v1/probe?tier=M">Tier M ({by_tier.get('M', {}).get('total', 0)})</a>
        <a class="nav-tab {'active' if active_tier == 'L' else ''}" href="/v1/probe?tier=L">Tier L ({by_tier.get('L', {}).get('total', 0)})</a>
      </div>
      <a style="margin-left:auto;background:var(--accent);color:var(--accent-fg);padding:6px 13px;border-radius:var(--radius-sm);font-size:12px;font-weight:600;text-decoration:none" href="/v1/probe?refresh=true{'&tier=' + active_tier if active_tier != 'ALL' else ''}">↻ Run Live Probe</a>
    </div>
    """

    return f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Waypost — Health & Probes</title>
<link rel=icon href="{_FAVICON}">
<style>
{_claude_theme_css()}
.section-card {{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  padding: 18px 22px;
  margin-bottom: 20px;
  box-shadow: var(--shadow-sm);
}}
.section-card h2 {{
  font-family: var(--font-sans);
  font-size: 11.5px;
  text-transform: uppercase;
  letter-spacing: .05em;
  color: var(--text-muted);
  margin: 0 0 12px;
  font-weight: 600;
}}
</style></head><body>
{_nav_header("probe")}
<div class="page-container">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:12px;margin-bottom:20px">
    <div>
      <h1 class="page-title">Health & Probes</h1>
      <p class="page-sub" style="margin-bottom:0">Active availability probes, TTFT latency & circuit breakers · <code>curl http://127.0.0.1:8080/v1/probe</code></p>
    </div>
    <a style="background:var(--accent);color:var(--accent-fg);padding:7px 14px;border-radius:var(--radius-sm);font-size:12px;font-weight:600;text-decoration:none;display:inline-block" href="/v1/probe?refresh=true{'&tier=' + active_tier if active_tier != 'ALL' else ''}">Run Live Probe</a>
  </div>

  <div class="summary-grid" style="grid-template-columns:repeat(auto-fit,minmax(260px,1fr))">{tier_cards_html}</div>

  <div class="section-card">
    <h2>Provider Circuit Breakers</h2>
    <div style="display:flex;flex-wrap:wrap;gap:6px;align-items:center">
      {brk_html}
    </div>
  </div>

  {filters_html}
  <div class="claude-table-wrap">
    <table class="claude-table"><thead><tr><th>Tier</th><th>Model</th><th>Provider</th><th>Status</th><th>TTFT / Latency</th><th>Caps</th><th>Details / Reason</th></tr></thead>
    <tbody>{rows_html}</tbody></table>
  </div>
</div>
</body></html>"""


def _stats_html(s: dict) -> str:
    ex = s.get("executor", {})
    cascade = s.get("cascade", {})
    hedging = s.get("hedging", {})
    last = s.get("last_24h", [])
    savings = s.get("savings", {})
    saved_usd = savings.get("total_saved_usd", 0.0)

    def pct(v):
        return f"{round(v*100, 1)}%"

    cards = [
        (f"${saved_usd:,.2f}", "estimated saved"),
        (ex.get("requests", 0), "requests"),
        (pct(s.get("cache_hit_rate", 0)), "cache hits"),
        (pct(s.get("semantic_hit_rate", 0)), "semantic cache"),
        (pct(cascade.get("escalation_rate", 0)), "cascade escalations"),
        (ex.get("hedges", 0), "hedge attempts"),
        (ex.get("billed", 0), "billed responses"),
    ]
    cards_html = "".join(
        f"<div class='stat-card'><b>{v}</b><span>{lbl}</span></div>" for v, lbl in cards
    )

    cascade_reasons = cascade.get("reasons", {})
    reasons_html = ", ".join(f"{k}: {v}" for k, v in cascade_reasons.items()) or "—"

    idem = s.get("idempotency", {})
    embed = s.get("embeddings", {})
    comp = s.get("compression", {})
    prof_txt = "yes" if cascade.get("profitable") else "no"

    rows = []
    for r in sorted(last, key=lambda x: x.get("attempts", 0), reverse=True):
        sr = r.get("success_rate", 0)
        rows.append(
            f"<tr><td class=id>{r['offering']}</td>"
            f"<td>{r.get('attempts', 0)}</td>"
            f"<td><span class='badge free'>{pct(sr)}</span></td>"
            f"<td>{round(r.get('avg_latency_ms', 0))}</td>"
            f"<td>{r.get('tokens', 0):,}</td></tr>"
        )
    rows_html = (
        "".join(rows)
        or "<tr><td colspan=5 style='color:var(--text-muted);padding:20px;text-align:center'>no data</td></tr>"
    )

    return f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Waypost — Analytics & Stats</title>
<link rel=icon href="{_FAVICON}">
<style>
{_claude_theme_css()}
.section-card {{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  padding: 18px 22px;
  margin-bottom: 18px;
  box-shadow: var(--shadow-sm);
}}
.section-card h2 {{
  font-family: var(--font-sans);
  font-size: 11.5px;
  text-transform: uppercase;
  letter-spacing: .05em;
  color: var(--text-muted);
  margin: 0 0 12px;
  font-weight: 600;
}}
.kv {{ color: var(--text-secondary); font-size: 13px; margin: 6px 0; }}
.kv b {{ color: var(--text); font-weight: 600; }}
</style></head><body>
{_nav_header("stats")}
<div class="page-container">
  <h1 class="page-title">Analytics & Telemetry</h1>
  <p class="page-sub">Execution performance, cost savings & optimization metrics · <code>curl http://127.0.0.1:8080/v1/stats</code></p>
  <div class="summary-grid">{cards_html}</div>
  <div class="section-card">
    <h2>Cost Savings vs Paid Commercial APIs</h2>
    <div class="kv">Total saved: <b>${saved_usd:,.2f}</b> · routed free tokens: <b>{savings.get('total_tokens',0):,}</b> across <b>{savings.get('total_requests',0)}</b> requests</div>
    <div class="kv">Baseline comparison: <b>GPT-4o-mini (Tier S)</b> · <b>Claude 3.5 Sonnet (Tier M)</b> · <b>OpenAI o1 (Tier L)</b></div>
  </div>
  <div class="section-card">
    <h2>Cascade & Optimization Mechanics</h2>
    <div class="kv">Cascade answers: <b>{cascade.get('answers',0)}</b> · escalations: <b>{cascade.get('escalations',0)}</b> ({pct(cascade.get('escalation_rate',0))}) · profitable: <b>{prof_txt}</b></div>
    <div class="kv">Escalation reasons: <b>{reasons_html}</b></div>
    <div class="kv">Hedge attempts: <b>{hedging.get('attempts',0)}</b> · hedged: <b>{hedging.get('hedged',0)}</b> ({pct(hedging.get('rate',0))})</div>
    <div class="kv">Idempotency: replays <b>{idem.get('replays',0)}</b> · buffer <b>{idem.get('stored',0)}</b></div>
    <div class="kv">Compression: <b>{comp.get('mode','—')}</b> · embeddings: <b>{embed.get('backend','—')}</b></div>
  </div>
  <div class="claude-table-wrap">
    <div style="padding:16px 20px 0"><h2 style="font-size:11.5px;text-transform:uppercase;letter-spacing:.05em;color:var(--text-muted);margin:0;font-weight:600">Last 24h Activity</h2></div>
    <table class="claude-table" style="margin-top:12px">
      <thead><tr><th>Model / Offering</th><th>Attempts</th><th>Success Rate</th><th>Avg Latency, ms</th><th>Tokens</th></tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
  </div>
</div>
</body></html>"""


def _pricing_html(data: dict) -> str:
    return _models_html(
        {
            "counts": data.get("counts", {}),
            "backends": [],
            "models": data.get("models", []),
        }
    )


def _local_html(data: dict) -> str:
    return _models_html(
        {
            "counts": {
                "total": data.get("total_models", len(data.get("models", []))),
                "cloud": 0,
                "local": data.get("total_models", len(data.get("models", []))),
                "free": data.get("total_models", len(data.get("models", []))),
                "paid": 0,
                "active_engines": data.get("active_engines", 0),
                "total_engines": data.get("total_engines", 0),
            },
            "backends": data.get("backends", []),
            "models": data.get("models", []),
        }
    )


@app.get("/v1/probe")
@app.get("/v1/probes")
async def probe_endpoint(
    request: Request,
    tier: str | None = None,
    provider: str | None = None,
    status: str | None = None,
    refresh: bool = False,
    all: bool = True,
):
    """Monitor model and provider availability with tier breakdowns."""
    registry: Registry = app.state.registry
    adapter = OpenAICompatAdapter(app.state.client)

    if refresh:
        candidates = registry.all() if all else registry.usable()
        if tier:
            candidates = [o for o in candidates if o.tier.value.upper() == tier.upper()]
        if provider:
            candidates = [
                o for o in candidates if o.provider.lower() == provider.lower()
            ]

        fresh = await probe_all(adapter, candidates)
        store_probes(settings.db_path, fresh)
        apply_to_registry(registry, {r["key"]: r for r in fresh})
        latest_map = latest_probes(settings.db_path)
    else:
        latest_map = latest_probes(settings.db_path)

    all_offerings = registry.all()
    results = []
    for o in all_offerings:
        if o.key in latest_map:
            item = dict(latest_map[o.key])
            item.setdefault("tier", o.tier.value)
            item.setdefault("provider", o.provider)
            item.setdefault("model_id", o.model_id)
            item.setdefault("free", o.free)
            item.setdefault("is_local", o.is_local)
            item.setdefault("caps", [c.value for c in o.caps])
            results.append(item)
        else:
            st = "healthy" if o.usable else "no_key"
            err = "" if o.usable else f"API key '{o.api_key_env}' is not configured"
            results.append(
                {
                    "key": o.key,
                    "provider": o.provider,
                    "model_id": o.model_id,
                    "tier": o.tier.value,
                    "free": o.free,
                    "is_local": o.is_local,
                    "caps": [c.value for c in o.caps],
                    "ts": 0,
                    "status": st,
                    "error": err,
                    "ttft_p50_ms": o.ttft_p50_ms if o.usable else None,
                }
            )

    summary = summarize_by_tier(results)

    filtered = results
    if tier:
        filtered = [
            r for r in filtered if str(r.get("tier", "")).upper() == tier.upper()
        ]
    if provider:
        filtered = [
            r
            for r in filtered
            if str(r.get("provider", "")).lower() == provider.lower()
        ]
    if status:
        filtered = [
            r for r in filtered if str(r.get("status", "")).lower() == status.lower()
        ]

    payload = {
        "summary": summary,
        "filters": {
            "tier": tier.upper() if tier else None,
            "provider": provider,
            "status": status,
        },
        "breakers": app.state.breaker.snapshot(),
        "total": len(filtered),
        "probes": filtered,
    }

    if "text/html" in request.headers.get("accept", ""):
        return HTMLResponse(_probe_html(payload))
    return payload


@app.post("/v1/probe/run")
async def run_probe_endpoint(payload: dict = Body(default_factory=dict)):
    """Trigger a probe run on-demand."""
    registry: Registry = app.state.registry
    adapter = OpenAICompatAdapter(app.state.client)
    tier = payload.get("tier")
    provider = payload.get("provider")

    candidates = registry.all()
    if tier:
        candidates = [
            o for o in candidates if o.tier.value.upper() == str(tier).upper()
        ]
    if provider:
        candidates = [
            o for o in candidates if o.provider.lower() == str(provider).lower()
        ]

    results = await probe_all(adapter, candidates)
    store_probes(settings.db_path, results)
    applied = apply_to_registry(registry, {r["key"]: r for r in results})
    summary = summarize_by_tier(results)
    return {
        "applied": applied,
        "summary": summary,
        "total": len(results),
        "probes": results,
    }


@app.get("/v1/pricing")
async def pricing_status(request: Request):
    """How each offering's paidness is proven.

    A separate endpoint, because the question "why is this model not picked"
    is asked more often than one would like, and the answer is here.
    """
    offerings = sorted(app.state.registry.all(), key=lambda o: o.key)
    payload = {
        "free_only": settings.free_only,
        "counts": {
            "free": sum(1 for o in offerings if o.free),
            "paid": sum(1 for o in offerings if not o.free),
            "undeclared": sum(
                1 for o in offerings if o.free_source == pricing.Source.UNKNOWN.value
            ),
        },
        "billed_responses": app.state.executor.snapshot().get("billed", 0),
        "models": [
            {
                "id": o.key,
                "free": o.free,
                "source": o.free_source,
                "detail": o.free_detail,
                "usable": o.usable,
                "is_local": o.is_local,
                "provider": o.provider,
                "tier": o.tier.value,
                "selectable": o.usable and (o.free or not settings.free_only),
            }
            for o in offerings
        ],
    }
    if "text/html" in request.headers.get("accept", ""):
        return HTMLResponse(_pricing_html(payload))
    return payload


@app.get("/v1/local")
@app.get("/local")
async def local_models_endpoint(request: Request, refresh: bool = False):
    """Dedicated endpoint for on-device inference engines and local offline models."""
    registry: Registry = app.state.registry
    adapter = OpenAICompatAdapter(app.state.client)

    if refresh:
        try:
            from .discovery import discover_local_backends

            res = await discover_local_backends(adapter, registry)
            app.state.discovery = res
        except Exception:
            pass

    from .discovery import LOCAL_DEFAULTS

    local_offerings = [o for o in registry.all() if o.is_local]

    backends = []
    for name, url_getter in LOCAL_DEFAULTS:
        url = url_getter()
        prov_models = [o for o in local_offerings if o.provider.lower() == name.lower()]
        active = any(o.usable for o in prov_models)
        backends.append(
            {
                "name": name,
                "url": url,
                "models_count": len(prov_models),
                "status": "online" if active else "offline",
                "models": [o.model_id for o in prov_models],
            }
        )

    known_names = {b["name"].lower() for b in backends}
    for o in local_offerings:
        if o.provider.lower() not in known_names:
            known_names.add(o.provider.lower())
            prov_models = [
                x for x in local_offerings if x.provider.lower() == o.provider.lower()
            ]
            backends.append(
                {
                    "name": o.provider,
                    "url": o.base_url,
                    "models_count": len(prov_models),
                    "status": "online"
                    if any(x.usable for x in prov_models)
                    else "offline",
                    "models": [x.model_id for x in prov_models],
                }
            )

    models_data = [
        {
            "id": o.model_id,
            "key": o.key,
            "provider": o.provider,
            "tier": o.tier.value,
            "ctx": o.ctx_window,
            "usable": o.usable,
            "caps": [c.value for c in o.caps],
            "base_url": o.base_url,
            "ttft_p50_ms": o.ttft_p50_ms,
        }
        for o in local_offerings
    ]

    payload = {
        "active_engines": sum(1 for b in backends if b["status"] == "online"),
        "total_engines": len(backends),
        "backends": backends,
        "total_models": len(models_data),
        "models": models_data,
    }

    if "text/html" in request.headers.get("accept", ""):
        return HTMLResponse(_local_html(payload))
    return payload


@app.get("/v1/discovery")
async def discovery():
    """Results of the auto-discovery of free models."""
    return {"results": app.state.discovery}


@app.get("/v1/jobs")
async def jobs():
    """Control-plane state: what ran when, and what failed."""
    return {"jobs": app.state.control.snapshot()}


@app.post("/v1/jobs/{name}/run")
async def run_job(name: str):
    try:
        result = await app.state.control.run_once(name)
    except KeyError:
        return JSONResponse(status_code=404, content={"error": f"unknown job {name}"})
    return {"job": name, "result": result}


def _build_stats_payload() -> dict:
    registry = app.state.registry
    pool: dict[str, int] = {}
    for o in registry.all():
        pool[o.lifecycle] = pool.get(o.lifecycle, 0) + 1
    return {
        "quota": app.state.ledger.snapshot(),
        "breakers": app.state.breaker.snapshot(),
        # The signals the router now actually acts on. Without them in
        # /v1/stats a degrading provider is only visible in the score.
        "latency": app.state.latency.snapshot(),
        "rate_limits": app.state.rate_governor.snapshot(),
        "inflight": dict(app.state.inflight),
        "pool": pool,
        "cache_hit_rate": round(app.state.cache.hit_rate(), 3),
        "semantic_hit_rate": round(app.state.semantic.hit_rate(), 3),
        "bandit": app.state.bandit.snapshot() if app.state.bandit else {},
        "executor": app.state.executor.snapshot(),
        "cascade": app.state.telemetry.cascade_stats(),
        "hedging": app.state.telemetry.hedge_stats(),
        "savings": app.state.telemetry.savings_stats(),
        "idempotency": app.state.idempotency.snapshot(),
        "batch": app.state.batch.stats() if app.state.batch else {},
        "jobs": app.state.control.snapshot(),
        "embeddings": (
            app.state.embeddings.snapshot()
            if app.state.embeddings
            else {"backend": "not loaded"}
        ),
        "compression": {"mode": app.state.compressor.backend},
        "last_24h": app.state.telemetry.summary(),
    }


@app.get("/v1/stats")
async def stats(request: Request):
    payload = _build_stats_payload()
    if "text/html" in request.headers.get("accept", ""):
        return HTMLResponse(_stats_html(payload))
    return payload


def _chat_html() -> str:
    """Minimalist Anthropic Claude-style interactive web chat."""
    tmpl = """<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Waypost — Chat</title>
<link rel=icon href="__FAVICON__">
<style>
__THEME_CSS__

html, body {
  height: 100%;
  overflow: hidden;
}
.chat-app {
  display: flex;
  flex-direction: column;
  height: 100vh;
  background: var(--bg);
}
.chat-topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 8px 20px;
  background: var(--card);
  border-bottom: 1px solid var(--border);
  gap: 12px;
  flex-wrap: wrap;
}
.model-pill {
  display: flex;
  align-items: center;
  gap: 8px;
  background: var(--bg-subtle);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 4px 10px;
  font-size: 13px;
  font-weight: 500;
}
.model-select {
  background: transparent;
  border: none;
  font-size: 13px;
  font-weight: 600;
  color: var(--text);
  outline: none;
  cursor: pointer;
  font-family: var(--font-sans);
}
.chat-options {
  display: flex;
  align-items: center;
  gap: 14px;
  font-size: 12px;
  color: var(--text-secondary);
}
.chat-options label {
  display: flex;
  align-items: center;
  gap: 5px;
  cursor: pointer;
}
.btn-icon {
  background: transparent;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  color: var(--text-secondary);
  padding: 4px 8px;
  cursor: pointer;
  font-size: 12px;
  font-weight: 500;
  transition: all 0.15s ease;
}
.btn-icon:hover {
  background: var(--bg-subtle);
  color: var(--text);
}

/* Message stream */
.chat-messages {
  flex: 1;
  overflow-y: auto;
  padding: 24px 16px;
  scroll-behavior: smooth;
}
.messages-inner {
  max-width: 760px;
  margin: 0 auto;
  display: flex;
  flex-direction: column;
  gap: 24px;
}

/* Welcome Hero */
.welcome-hero {
  text-align: center;
  padding: 40px 20px 20px;
  margin: auto 0;
}
.welcome-icon {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 52px;
  height: 52px;
  border-radius: var(--radius-md);
  background: var(--bg-subtle);
  color: var(--text);
  border: 1px solid var(--border);
  margin-bottom: 16px;
}
.welcome-title {
  font-family: var(--font-sans);
  font-size: 24px;
  font-weight: 600;
  color: var(--text);
  margin: 0 0 8px;
  letter-spacing: -0.02em;
}
.welcome-sub {
  color: var(--text-secondary);
  font-size: 13.5px;
  max-width: 480px;
  margin: 0 auto 28px;
  line-height: 1.5;
}
.prompt-chips {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 10px;
  max-width: 620px;
  margin: 0 auto;
}
.prompt-chip {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  padding: 12px 14px;
  font-size: 13px;
  color: var(--text);
  text-align: left;
  cursor: pointer;
  transition: all 0.15s ease;
  box-shadow: var(--shadow-sm);
}
.prompt-chip:hover {
  border-color: var(--border-focus);
  background: var(--card-hover);
  transform: translateY(-1px);
}

/* Message items */
.message-row {
  display: flex;
  gap: 12px;
  width: 100%;
}
.message-row.user {
  justify-content: flex-end;
}
.message-bubble {
  max-width: 85%;
  font-size: 14px;
  line-height: 1.6;
}
.message-row.user .message-bubble {
  background: var(--card);
  border: 1px solid var(--border);
  padding: 10px 15px;
  border-radius: 14px 14px 2px 14px;
  box-shadow: var(--shadow-sm);
  color: var(--text);
  white-space: pre-wrap;
}
.message-row.assistant {
  justify-content: flex-start;
}
.assistant-avatar {
  width: 28px;
  height: 28px;
  border-radius: var(--radius-sm);
  background: var(--bg-subtle);
  color: var(--text);
  border: 1px solid var(--border);
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  margin-top: 2px;
}
.message-row.assistant .message-bubble {
  flex: 1;
  color: var(--text);
}
.router-pill {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  background: var(--bg-subtle);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 3px 9px;
  font-size: 11px;
  color: var(--text-secondary);
  margin-top: 8px;
  font-family: var(--font-mono);
}
.router-pill b { color: var(--text); }

/* Markdown typography */
.message-bubble h1, .message-bubble h2, .message-bubble h3 {
  font-family: var(--font-sans);
  margin: 14px 0 6px;
  font-weight: 600;
  color: var(--text);
  letter-spacing: -0.01em;
}
.message-bubble h1 { font-size: 18px; }
.message-bubble h2 { font-size: 15px; }
.message-bubble h3 { font-size: 13.5px; }
.message-bubble p { margin: 0 0 10px; }
.message-bubble p:last-child { margin-bottom: 0; }
.message-bubble ul, .message-bubble ol {
  margin: 8px 0 12px;
  padding-left: 20px;
}
.message-bubble li { margin-bottom: 4px; }
.message-bubble blockquote {
  margin: 10px 0;
  padding: 4px 12px;
  border-left: 2px solid var(--text-muted);
  color: var(--text-secondary);
  background: var(--bg-subtle);
  border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
}
.message-bubble code {
  font-family: var(--font-mono);
  font-size: 12px;
  background: var(--bg-subtle);
  border: 1px solid var(--border-subtle);
  padding: 2px 5px;
  border-radius: 4px;
}
.code-block-wrap {
  position: relative;
  margin: 12px 0;
  border-radius: var(--radius-sm);
  border: 1px solid var(--border);
  overflow: hidden;
  background: var(--card);
}
.code-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 6px 12px;
  background: var(--bg-subtle);
  border-bottom: 1px solid var(--border-subtle);
  font-size: 11px;
  color: var(--text-muted);
  font-family: var(--font-mono);
  text-transform: uppercase;
}
.btn-copy {
  background: transparent;
  border: none;
  color: var(--text-secondary);
  font-size: 11px;
  cursor: pointer;
  padding: 2px 6px;
  border-radius: 4px;
}
.btn-copy:hover { background: var(--border); color: var(--text); }
.code-block-wrap pre {
  margin: 0;
  padding: 12px 14px;
  overflow-x: auto;
  font-family: var(--font-mono);
  font-size: 12.5px;
  line-height: 1.5;
}
.code-block-wrap pre code {
  background: transparent;
  border: none;
  padding: 0;
}

/* Input bar */
.chat-bottom {
  padding: 12px 16px 20px;
  background: var(--bg);
  border-top: 1px solid var(--border-subtle);
}
.input-container {
  max-width: 760px;
  margin: 0 auto;
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 8px 12px 8px 16px;
  box-shadow: var(--shadow-md);
  display: flex;
  align-items: flex-end;
  gap: 8px;
  transition: border-color 0.15s ease;
}
.input-container:focus-within {
  border-color: var(--border-focus);
}
#chat-input {
  flex: 1;
  background: transparent;
  border: none;
  outline: none;
  font-size: 14px;
  font-family: var(--font-sans);
  color: var(--text);
  resize: none;
  max-height: 180px;
  min-height: 24px;
  line-height: 1.5;
  padding: 4px 0;
}
.btn-send {
  width: 30px;
  height: 30px;
  border-radius: 50%;
  background: var(--accent);
  color: var(--accent-fg);
  border: none;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  font-size: 14px;
  font-weight: 700;
  transition: all 0.15s ease;
  flex-shrink: 0;
}
.btn-send:hover {
  background: var(--accent-hover);
  transform: scale(1.03);
}
.btn-send:disabled {
  background: var(--border);
  color: var(--text-muted);
  cursor: not-allowed;
  transform: none;
}
.input-hint {
  text-align: center;
  font-size: 11px;
  color: var(--text-muted);
  margin-top: 6px;
}
.cursor-blink {
  display: inline-block;
  width: 5px;
  height: 14px;
  background: var(--text);
  margin-left: 2px;
  vertical-align: -2px;
  animation: blink 0.9s infinite;
}
@keyframes blink {
  0%, 100% { opacity: 1; }
  50% { opacity: 0; }
}
</style></head><body>
<div class="chat-app">
  __NAV_HEADER__
  <div class="chat-topbar">
    <div class="model-pill">
      <span class="model-indicator"></span>
      <select id="model-select" class="model-select">
        <option value="auto">auto (Smart Router)</option>
        <option value="Tier S">Tier S · Fast</option>
        <option value="Tier M">Tier M · General</option>
        <option value="Tier L">Tier L · Reasoning</option>
      </select>
    </div>
    <div class="chat-options">
      <label><input type="checkbox" id="opt-stream" checked> Streaming</label>
      <label>Privacy:
        <select id="opt-privacy" style="background:transparent;border:1px solid var(--border);border-radius:4px;color:var(--text);font-size:11px;padding:2px 4px">
          <option value="default">default</option>
          <option value="strict">strict (no cloud data)</option>
        </select>
      </label>
      <button class="btn-icon" id="btn-clear" title="Clear chat (⌘K)">Clear</button>
    </div>
  </div>

  <div class="chat-messages" id="messages-container">
    <div class="messages-inner" id="messages-list">
      <div class="welcome-hero" id="welcome-hero">
        <div class="welcome-icon">
          <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg>
        </div>
        <h2 class="welcome-title">How can I help you today?</h2>
        <p class="welcome-sub">Waypost routes your prompt across the fastest and best free model providers locally.</p>
        <div class="prompt-chips">
          <div class="prompt-chip" onclick="usePrompt(this.innerText)">Compare Rust vs Go for high-throughput networking</div>
          <div class="prompt-chip" onclick="usePrompt(this.innerText)">Write a Python decorator to rate-limit async functions</div>
          <div class="prompt-chip" onclick="usePrompt(this.innerText)">Explain transformer self-attention mechanisms simply</div>
          <div class="prompt-chip" onclick="usePrompt(this.innerText)">How does circuit breaking work in distributed systems?</div>
        </div>
      </div>
    </div>
  </div>

  <div class="chat-bottom">
    <div class="input-container">
      <textarea id="chat-input" placeholder="Message Waypost..." rows="1"></textarea>
      <button class="btn-send" id="btn-send" title="Send message (Enter)">↑</button>
    </div>
    <div class="input-hint">Waypost Smart Router · Enter to send · Shift+Enter for new line</div>
  </div>
</div>

<script>
const messagesList = document.getElementById('messages-list');
const messagesContainer = document.getElementById('messages-container');
const chatInput = document.getElementById('chat-input');
const btnSend = document.getElementById('btn-send');
const btnClear = document.getElementById('btn-clear');
const modelSelect = document.getElementById('model-select');
const optStream = document.getElementById('opt-stream');
const optPrivacy = document.getElementById('opt-privacy');
const welcomeHero = document.getElementById('welcome-hero');

let history = [];
let isGenerating = false;
let abortController = null;

// Populate specific models in select
fetch('/v1/models').then(r => r.json()).then(res => {
  if (res && res.data) {
    res.data.forEach(m => {
      if (m.id && m.id !== 'auto' && !['Tier S', 'Tier M', 'Tier L'].includes(m.id)) {
        const opt = document.createElement('option');
        opt.value = m.id;
        opt.textContent = m.id;
        modelSelect.appendChild(opt);
      }
    });
  }
}).catch(() => {});

// Auto-expand textarea
chatInput.addEventListener('input', () => {
  chatInput.style.height = 'auto';
  chatInput.style.height = Math.min(chatInput.scrollHeight, 180) + 'px';
});

chatInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
  if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
    e.preventDefault();
    clearChat();
  }
});

btnSend.addEventListener('click', () => {
  if (isGenerating) {
    stopGeneration();
  } else {
    sendMessage();
  }
});

btnClear.addEventListener('click', clearChat);

function usePrompt(text) {
  chatInput.value = text;
  sendMessage();
}

function clearChat() {
  history = [];
  messagesList.innerHTML = '';
  if (welcomeHero) messagesList.appendChild(welcomeHero);
  chatInput.focus();
}

function escapeHtml(str) {
  if (str === null || str === undefined) return '';
  const s = typeof str === 'string' ? str : String(str);
  return s.replace(/&/g, '&amp;')
          .replace(/</g, '&lt;')
          .replace(/>/g, '&gt;')
          .replace(/"/g, '&quot;')
          .replace(/'/g, '&#039;');
}

function extractText(obj) {
  if (!obj) return '';
  if (typeof obj === 'string') return obj;
  if (Array.isArray(obj)) {
    return obj.map(item => {
      if (typeof item === 'string') return item;
      if (item && item.text) return item.text;
      if (item && item.content) return extractText(item.content);
      return '';
    }).join('');
  }
  if (obj.text) return typeof obj.text === 'string' ? obj.text : extractText(obj.text);
  if (obj.content) return typeof obj.content === 'string' ? obj.content : extractText(obj.content);
  if (obj.reasoning) return typeof obj.reasoning === 'string' ? obj.reasoning : extractText(obj.reasoning);
  if (obj.reasoning_content) return typeof obj.reasoning_content === 'string' ? obj.reasoning_content : extractText(obj.reasoning_content);
  return '';
}

// Lightweight clean Markdown parser
function renderMarkdown(md) {
  if (md === null || md === undefined) return '';
  let str = typeof md === 'string' ? md : String(md);
  if (!str) return '';

  const codeBlockCount = (str.match(/```/g) || []).length;
  if (codeBlockCount % 2 === 1) {
    str = str + '\\n```';
  }

  // 1. Code blocks
  let text = str.replace(/```([a-zA-Z0-9_-]*)\\n([\\s\\S]*?)```/g, function(match, lang, code) {
    const l = lang ? lang.trim() : 'text';
    const escaped = escapeHtml(code.replace(/\\n$/, ''));
    return '<div class="code-block-wrap"><div class="code-header"><span>' + l + '</span><button class="btn-copy" onclick="copyCode(this)">Copy</button></div><pre><code>' + escaped + '</code></pre></div>';
  });

  // 2. Inline code
  text = text.replace(/`([^`]+)`/g, function(match, code) {
    return '<code>' + escapeHtml(code) + '</code>';
  });

  // 3. Headings
  text = text.replace(/^### (.*$)/gim, '<h3>$1</h3>')
             .replace(/^## (.*$)/gim, '<h2>$1</h2>')
             .replace(/^# (.*$)/gim, '<h1>$1</h1>');

  // 4. Bold & italic
  text = text.replace(/\\*\\*(.*?)\\*\\*/g, '<strong>$1</strong>')
             .replace(/\\*(.*?)\\*/g, '<em>$1</em>');

  // 5. Blockquotes
  text = text.replace(/^\\> (.*$)/gim, '<blockquote>$1</blockquote>');

  // 6. Lists
  text = text.replace(/^\\s*[-*+] (.*$)/gim, '<li>$1</li>');
  text = text.replace(/(<li>.*<\\/li>)/s, '<ul>$1</ul>');

  // 7. Paragraphs
  const paragraphs = text.split(/\\n\\n+/);
  return paragraphs.map(p => {
    p = p.trim();
    if (!p) return '';
    if (p.startsWith('<div') || p.startsWith('<h') || p.startsWith('<ul') || p.startsWith('<blockquote')) {
      return p;
    }
    return '<p>' + p.replace(/\\n/g, '<br>') + '</p>';
  }).join('');
}

function copyCode(btn) {
  const pre = btn.parentElement.nextElementSibling;
  if (pre) {
    navigator.clipboard.writeText(pre.innerText).then(() => {
      const orig = btn.innerText;
      btn.innerText = 'Copied!';
      setTimeout(() => { btn.innerText = orig; }, 1500);
    });
  }
}

function appendUserMessage(content) {
  if (welcomeHero && welcomeHero.parentNode) {
    welcomeHero.parentNode.removeChild(welcomeHero);
  }
  const row = document.createElement('div');
  row.className = 'message-row user';
  row.innerHTML = '<div class="message-bubble">' + escapeHtml(content) + '</div>';
  messagesList.appendChild(row);
  messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function createAssistantMessage() {
  const row = document.createElement('div');
  row.className = 'message-row assistant';
  row.innerHTML = `
    <div class="assistant-avatar">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg>
    </div>
    <div class="message-bubble"><div class="bubble-content"><span class="cursor-blink"></span></div><div class="bubble-meta"></div></div>
  `;
  messagesList.appendChild(row);
  messagesContainer.scrollTop = messagesContainer.scrollHeight;
  return {
    row: row,
    contentEl: row.querySelector('.bubble-content'),
    metaEl: row.querySelector('.bubble-meta')
  };
}

function stopGeneration() {
  if (abortController) {
    abortController.abort();
    abortController = null;
  }
  setGenerating(false);
}

function setGenerating(gen) {
  isGenerating = gen;
  if (gen) {
    btnSend.textContent = '■';
    btnSend.title = 'Stop generating';
  } else {
    btnSend.textContent = '↑';
    btnSend.title = 'Send message';
  }
}

async function sendMessage() {
  const text = chatInput.value.trim();
  if (!text || isGenerating) return;

  chatInput.value = '';
  chatInput.style.height = 'auto';

  appendUserMessage(text);
  history.push({ role: 'user', content: text });

  const isStream = optStream ? optStream.checked : true;
  const model = (modelSelect && modelSelect.value) ? modelSelect.value : 'auto';
  const privacy = (optPrivacy && optPrivacy.value) ? optPrivacy.value : 'default';

  const cleanMessages = history.filter(m => m && m.content && String(m.content).trim().length > 0);

  const payload = {
    model: model,
    messages: cleanMessages,
    stream: isStream,
    privacy: privacy,
  };

  const assistantMsg = createAssistantMessage();
  let fullContent = '';
  let routerMeta = null;

  setGenerating(true);
  abortController = new AbortController();

  try {
    const response = await fetch('/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': isStream ? 'text/event-stream, application/json' : 'application/json'
      },
      body: JSON.stringify(payload),
      signal: abortController.signal
    });

    if (!response.ok) {
      const errJson = await response.json().catch(() => ({}));
      const errMsg = (errJson.error && errJson.error.message) ? errJson.error.message : response.statusText;
      assistantMsg.contentEl.innerHTML = '<span style="color:var(--text-muted);border-left:2px solid var(--text);padding-left:8px;display:inline-block">Error: ' + escapeHtml(errMsg) + '</span>';
      setGenerating(false);
      return;
    }

    if (isStream) {
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed.startsWith('data:')) continue;
          const dataStr = trimmed.slice(5).trim();
          if (dataStr === '[DONE]') break;
          try {
            const chunk = JSON.parse(dataStr);
            if (chunk.error) {
              const errMsg = typeof chunk.error === 'string' ? chunk.error : (chunk.error.message || JSON.stringify(chunk.error));
              assistantMsg.contentEl.innerHTML = '<span style="color:var(--text-muted);border-left:2px solid var(--text);padding-left:8px;display:inline-block">Error: ' + escapeHtml(errMsg) + '</span>';
              return;
            }
            const c0 = (chunk.choices && chunk.choices[0]) || {};
            const d = c0.delta || {};
            const delta = extractText(d) || (typeof d.content === 'string' ? d.content : '');
            if (delta) {
              fullContent += delta;
              assistantMsg.contentEl.innerHTML = renderMarkdown(fullContent) + '<span class="cursor-blink"></span>';
              messagesContainer.scrollTop = messagesContainer.scrollHeight;
            }
            if (chunk.router) {
              routerMeta = chunk.router;
            }
          } catch (e) {}
        }
      }
      if (!fullContent) {
        assistantMsg.contentEl.innerHTML = '<span style="color:var(--text-muted)">(Empty response from model)</span>';
      } else {
        assistantMsg.contentEl.innerHTML = renderMarkdown(fullContent);
      }
    } else {
      const data = await response.json();
      if (data.error) {
        const errMsg = typeof data.error === 'string' ? data.error : (data.error.message || JSON.stringify(data.error));
        assistantMsg.contentEl.innerHTML = '<span style="color:var(--text-muted);border-left:2px solid var(--text);padding-left:8px;display:inline-block">Error: ' + escapeHtml(errMsg) + '</span>';
        setGenerating(false);
        return;
      }
      const m0 = (data.choices && data.choices[0] && data.choices[0].message) || {};
      fullContent = extractText(m0) || (typeof m0.content === 'string' ? m0.content : '');
      routerMeta = data.router;
      if (!fullContent) {
        assistantMsg.contentEl.innerHTML = '<span style="color:var(--text-muted)">(Empty response from model)</span>';
      } else {
        assistantMsg.contentEl.innerHTML = renderMarkdown(fullContent);
      }
    }

    if (fullContent && fullContent.trim()) {
      history.push({ role: 'assistant', content: fullContent });
    }

    if (routerMeta) {
      const prov = routerMeta.provider || '—';
      const mName = routerMeta.model || '—';
      const tier = routerMeta.complexity_tier || '—';
      const lat = routerMeta.latency_ms ? routerMeta.latency_ms + 'ms' : '';
      const cached = routerMeta.cache === 'hit' ? 'cached' : 'fresh';
      assistantMsg.metaEl.innerHTML = `
        <div class="router-pill">
          <span><b>Router:</b> ${escapeHtml(prov)} · ${escapeHtml(mName)} · Tier ${escapeHtml(tier)} ${lat ? '· ' + lat : ''} · ${cached}</span>
        </div>
      `;
    }
  } catch (err) {
    if (err.name !== 'AbortError') {
      assistantMsg.contentEl.innerHTML = '<span style="color:var(--text-muted);border-left:2px solid var(--text);padding-left:8px;display:inline-block">Connection error: ' + escapeHtml(err.message) + '</span>';
    } else {
      assistantMsg.contentEl.innerHTML = renderMarkdown(fullContent) + ' <span style="color:var(--text-muted);font-size:12px">(stopped)</span>';
    }
  } finally {
    setGenerating(false);
    abortController = null;
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
  }
}
</script>
</body></html>"""
    return (
        tmpl.replace("__FAVICON__", _FAVICON)
        .replace("__THEME_CSS__", _claude_theme_css())
        .replace("__NAV_HEADER__", _nav_header("chat"))
    )


@app.get("/")
@app.get("/chat")
@app.get("/v1/chat")
async def chat_page(request: Request):
    """Interactive Anthropic Claude-style chat web interface."""
    return HTMLResponse(render_chat_html())


@app.get("/dashboard")
async def dashboard_page(request: Request):
    """Operational 6-block dashboard interface."""
    stats_data = _build_stats_payload()
    cov_data = app.state.telemetry.outcome_coverage()
    cov = cov_data.get("outcome_coverage", 1.0)
    attempts = app.state.telemetry.attempt_logs(limit=15)
    traces = []
    for a in attempts:
        traces.append(
            {
                "time_str": time.strftime("%H:%M:%S", time.localtime(a.get("ts", 0))),
                "request_id": a.get("request_id", ""),
                "path_summary": f"{a.get('routing_source', 'l0')} → {a.get('backend', '')}/{a.get('model', '')} → {a.get('outcome', 'unknown')}",
                "duration_s": (a.get("latency_ms", 0) or 0) / 1000.0,
                "is_exploration": bool(a.get("is_exploration", False)),
            }
        )

    quotas = []
    if hasattr(app.state.ledger, "detailed_snapshot"):
        detailed_quotas = app.state.ledger.detailed_snapshot()
        for p_name, snap in detailed_quotas.items():
            quotas.append(snap)
    else:
        for p_name, snap in stats_data.get("quota", {}).items():
            quotas.append(
                {
                    "provider": p_name,
                    "used_pct": 0,
                    "remaining_pct": 100,
                    "remaining_summary": "Active",
                    "used_summary": "0 used",
                    "burn_rate": "0/min",
                    "binding": "limits active",
                    "exhaustion_eta": "100% capacity available",
                }
            )

    cascade = stats_data.get("cascade", {})
    last_24h_list = stats_data.get("last_24h", [])
    lat_pcts = app.state.telemetry.latency_percentiles(
        local_offerings={o.key for o in app.state.registry.all() if o.is_local}
    )
    total_reqs = (
        sum(item.get("attempts", 0) for item in last_24h_list)
        if isinstance(last_24h_list, list)
        else 0
    )

    top_models = []
    if last_24h_list and isinstance(last_24h_list, list):
        sorted_24h = sorted(
            last_24h_list, key=lambda x: x.get("attempts", 0), reverse=True
        )
        for item in sorted_24h[:5]:
            off = item.get("offering", "")
            att = item.get("attempts", 0)
            succ = item.get("success_rate", 1.0) * 100
            tok = item.get("tokens", 0)
            backend = (
                "mlx" if "mlx" in off else ("ollama" if "local" in off else "cloud")
            )
            top_models.append(
                {
                    "model": off,
                    "backend": backend,
                    "thinking_supported": "qwen" in off.lower()
                    or "reason" in off.lower(),
                    "primary_calls": att,
                    "escalation_calls": 0,
                    "success_rate_pct": succ,
                    # Measured, not the average scaled by a made-up factor.
                    "p50_ms": lat_pcts["per_offering"].get(off, {}).get("p50", 0),
                    "p95_ms": lat_pcts["per_offering"].get(off, {}).get("p95", 0),
                    "total_tokens": tok,
                }
            )

    if not top_models:
        for o in app.state.registry.all()[:5]:
            backend = (
                "mlx" if "mlx" in o.key else ("ollama" if o.is_local else o.provider)
            )
            top_models.append(
                {
                    "model": o.key,
                    "backend": backend,
                    "thinking_supported": bool(getattr(o, "supports_thinking", False)),
                    "primary_calls": 0,
                    "escalation_calls": 0,
                    # Nothing was measured for this offering yet: show the
                    # manifest's declared TTFT, and zero where there is no
                    # basis for a p95 at all.
                    "success_rate_pct": 0.0,
                    "p50_ms": int(o.ttft_p50_ms or 0),
                    "p95_ms": 0,
                    "total_tokens": 0,
                }
            )

    dash_payload = {
        "summary": {
            "total_requests": total_reqs or cascade.get("answers", 0),
            "escalation_rate_pct": cascade.get("escalation_rate", 0.0) * 100,
            "local_vs_cloud_threshold": app.state.router.local_vs_cloud_threshold(),
            "funnel": {
                "in": total_reqs or cascade.get("answers", 0),
                "cache": int(total_reqs * stats_data.get("cache_hit_rate", 0.0)),
                "local": sum(
                    item.get("attempts", 0)
                    for item in last_24h_list
                    if "local" in item.get("offering", "")
                    or "mlx" in item.get("offering", "")
                ),
                "cloud": sum(
                    item.get("attempts", 0)
                    for item in last_24h_list
                    if "local" not in item.get("offering", "")
                    and "mlx" not in item.get("offering", "")
                ),
                "escalated": cascade.get("escalations", 0),
                "failed": 0,
            },
        },
        "quotas": quotas,
        "latencies": {
            "local_p50": lat_pcts["local"]["p50"],
            "local_p95": lat_pcts["local"]["p95"],
            "local_n": lat_pcts["local"]["n"],
            "cloud_p50": lat_pcts["cloud"]["p50"],
            "cloud_p95": lat_pcts["cloud"]["p95"],
            "cloud_n": lat_pcts["cloud"]["n"],
        },
        "escalations": cascade.get("reasons", {}),
        "top_models": top_models,
        "recent_traces": traces,
        "outcome_coverage": cov,
    }
    return HTMLResponse(render_dashboard_html(dash_payload))


def _beta_metric() -> dict[str, Any]:
    """Read the last real measurement written by scripts/measure_beta.py.

    Hardcoding this panel stated the opposite of what the measurement
    concluded — the page said "ensemble deferred" while the file on disk
    said the ceiling is worth chasing.
    """
    path = Path(settings.beta_path)
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {
            "p_best": 0.0,
            "beta": 0.0,
            "verdict": f"not measured — run scripts/measure_beta.py ({path})",
        }
    return {
        "p_best": float(data.get("p_best", 0.0)),
        "beta": float(data.get("beta", 0.0)),
        "verdict": str(data.get("verdict", "no verdict recorded")),
    }


def _keys_health(registry: Registry) -> list[dict[str, Any]]:
    """One row per provider, counted off the live registry. Hardcoding
    these numbers hid a dead backend behind a green badge."""
    rows: dict[str, dict[str, Any]] = {}
    for o in sorted(registry.all(), key=lambda x: (x.provider, x.model_id)):
        row = rows.get(o.provider)
        if row is None:
            if o.is_local:
                status, badge = "local (no key needed)", "badge-pass"
            elif not o.api_key_env:
                status, badge = "no auth", "badge-pass"
            elif o.api_key:
                keys = o.key_count
                status = f"valid ({keys} key{'s' if keys > 1 else ''})"
                badge = "badge-pass"
            else:
                status, badge = f"missing {o.api_key_env}", "badge-warn"
            row = rows[o.provider] = {
                "provider": o.provider,
                "base_url": o.base_url,
                "key_status": status,
                "key_badge": badge,
                "model_count": 0,
            }
        if o.usable:
            row["model_count"] += 1
    return list(rows.values())


@app.get("/setup")
async def setup_page(request: Request):
    """Setup & configuration interface."""
    cov_data = app.state.telemetry.outcome_coverage()
    registry: Registry = app.state.registry
    hedge = app.state.telemetry.hedge_stats()
    ex = app.state.executor.snapshot()
    cascade_stats = app.state.telemetry.cascade_stats()
    local_offerings = [o for o in registry.all() if o.is_local]
    savings = app.state.telemetry.savings_stats()
    setup_payload = {
        "subsystems": [
            {
                "name": "Embeddings & Vector Cache",
                "status": "on" if app.state.embeddings else "not loaded",
                "effect": f"{settings.embed_model} — semantic cache"
                + (" & L1 classifier" if settings.enable_l1_classifier else ""),
            },
            {
                "name": f"Semantic Cache ({settings.semantic_threshold:g} threshold)",
                "status": "on" if settings.enable_semantic_cache else "off",
                "effect": f"hit rate {app.state.semantic.hit_rate() * 100:.1f}%"
                + (
                    f", cross-encoder verifier ≥{settings.rerank_threshold:g}"
                    if settings.enable_semantic_rerank
                    else ", no cross-encoder verifier"
                ),
            },
            {
                "name": "Exact Cache",
                "status": "on",
                "effect": f"hit rate {app.state.cache.hit_rate() * 100:.1f}%, "
                f"TTL {settings.cache_ttl_s // 3600}h",
            },
            {
                "name": "Hedge Requests",
                "status": "on" if settings.enable_hedging else "off",
                "effect": f"budget {settings.hedge_budget * 100:.0f}% of requests; "
                f"fired {ex.get('hedges', 0)} this session, "
                f"won {hedge.get('hedged', 0)}/{hedge.get('attempts', 0)} in 24h",
            },
            {
                "name": "Cloud-First Ladder + Thinking Escalation",
                "status": "on" if settings.enable_verifier else "escalation off",
                "effect": "cloud first, local tail fallback "
                f"({len(local_offerings)} local model(s)); privacy_only forces local. "
                f"Escalations {cascade_stats.get('escalations', 0)}/"
                f"{cascade_stats.get('answers', 0)}",
            },
        ],
        "engines": [
            {
                "name": f"{o.provider} ({o.base_url})",
                "model": o.model_id,
                "mem_used_gb": sysmem.engine_footprint_gb(o.base_url),
                "mem_limit_gb": sysmem.total_ram_gb(),
                "warmup_status": ("usable" if o.usable else "disabled")
                + f" · tier {o.tier.value} · ctx {o.ctx_window:,}"
                + (f" · concurrency {o.concurrency}" if o.concurrency else ""),
            }
            for o in local_offerings
        ]
        or [
            {
                "name": "no local engines registered",
                "model": "—",
                "mem_used_gb": None,
                "mem_limit_gb": None,
                "warmup_status": "start MLX on :8081 and re-run discovery",
            }
        ],
        "keys_health": _keys_health(registry),
        "data_health": {
            "outcome_coverage": cov_data.get("outcome_coverage", 1.0),
            "attempt_log_count": cov_data.get("total_attempts", 0),
            "exploration_rate_pct": settings.explore_rate * 100,
        },
        "beta_metric": _beta_metric(),
        "pricing": {
            "saved_usd": savings.get("total_saved_usd", 0.0),
            "total_tokens": savings.get("total_tokens", 0),
            "usage_coverage_pct": savings.get("usage_coverage_pct", 0.0),
            "measured_requests": savings.get("measured_requests", 0),
            "total_requests": savings.get("total_requests", 0),
            "exact_savings_requests": savings.get("exact_savings_requests", 0),
            "estimated_savings_requests": savings.get(
                "estimated_savings_requests", 0
            ),
            "baseline": savings.get("baseline", "Waypost commercial baseline v1"),
        },
    }
    return HTMLResponse(render_setup_html(setup_payload))


@app.get("/metrics")
async def metrics_endpoint():
    if not settings.enable_metrics:
        return PlainTextResponse("metrics disabled\n", status_code=404)
    return PlainTextResponse(
        app.state.metrics.render(), media_type="text/plain; version=0.0.4"
    )


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest, request: Request):
    # Routing profile: explicit in body or passed via header/query
    profile_hdr = request.headers.get("x-waypost-profile") or request.query_params.get(
        "profile"
    )
    if profile_hdr and profile_hdr in (
        "auto",
        "code_completion",
        "reasoning",
        "privacy_only",
        "balanced",
    ):
        req.profile = profile_hdr

    # Idempotency: a key from the body or the header. A retry after a network
    # drop must not charge the quota a second time.
    idem = req.idempotency_key or request.headers.get("idempotency-key")
    if idem and not req.stream:
        store: IdempotencyStore = app.state.idempotency
        async with store.lock(idem):
            if (hit := store.get(idem)) is not None:
                hit["router"] = {**hit.get("router", {}), "replay": True}
                log.info("IDEMPOTENT replay %s", idem[:16])
                return hit
            body = await run_chat(req)
            store.put(idem, body)
            return body

    if req.stream:
        meta = RouterMeta()
        profile = classify(
            req,
            enable_l1=settings.enable_l1_classifier,
            head_path=str(settings.head_path),
        )
        decision = app.state.policy.apply(req, profile)
        if decision.blocked:
            raise RouterError(decision.block_reason, 400, meta)
        meta.policy = decision.as_meta()
        meta.task_class = profile.task_class
        meta.complexity_tier = profile.tier.value
        plan = app.state.router.plan(
            req, profile, quality_floor=decision.quality_floor, prefix=prefix_hash(req)
        )

        # Starlette sends the HTTP status and headers before it starts
        # iterating StreamingResponse.  Pull one upstream chunk here so a
        # dead provider can still fall through the executor ladder, and a
        # total failure is returned as an honest JSON 502 instead of a 200
        # followed by an incomplete chunked body.
        upstream = app.state.executor.stream(req, profile, plan, meta).__aiter__()
        try:
            first_chunk = await upstream.__anext__()
        except StopAsyncIteration:
            first_chunk = b""

        async def gen():
            last_chunk = first_chunk
            try:
                if first_chunk:
                    yield first_chunk
                async for chunk in upstream:
                    last_chunk = chunk
                    yield chunk
                if not last_chunk.strip().endswith(b"[DONE]"):
                    yield b"data: [DONE]\n\n"
            finally:
                aclose = getattr(upstream, "aclose", None)
                if aclose is not None:
                    await aclose()

        app.state.metrics.inc("waypost_requests_total", status="stream")
        return StreamingResponse(gen(), media_type="text/event-stream")

    return await run_chat(req)


@app.post("/v1/responses")
async def responses_endpoint(request: Request, payload: dict = Body(...)):
    """OpenAI Responses compatibility over Waypost's existing chat hot path.

    The translation deliberately happens only at the protocol edge: policy,
    caching, routing, quota accounting, verification and provider fallback are
    exactly the same as for ``/v1/chat/completions``.
    """
    try:
        req = responses_to_chat(payload)
    except (TypeError, ValueError) as exc:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": str(exc),
                    "type": "invalid_request_error",
                }
            },
        )

    profile_hdr = request.headers.get(
        "x-waypost-profile"
    ) or request.query_params.get("profile")
    if profile_hdr in (
        "auto",
        "code_completion",
        "reasoning",
        "privacy_only",
        "balanced",
    ):
        req.profile = profile_hdr

    idem = req.idempotency_key or request.headers.get("idempotency-key")

    if not req.stream:
        if idem:
            store: IdempotencyStore = app.state.idempotency
            async with store.lock(idem):
                if (hit := store.get(idem)) is not None:
                    hit["router"] = {**hit.get("router", {}), "replay": True}
                    return chat_to_response(hit, payload)
                body = await run_chat(req)
                store.put(idem, body)
                return chat_to_response(body, payload)
        return chat_to_response(await run_chat(req), payload)

    meta = RouterMeta()
    profile = classify(
        req,
        enable_l1=settings.enable_l1_classifier,
        head_path=str(settings.head_path),
    )
    decision = app.state.policy.apply(req, profile)
    if decision.blocked:
        raise RouterError(decision.block_reason, 400, meta)
    meta.policy = decision.as_meta()
    meta.task_class = profile.task_class
    meta.complexity_tier = profile.tier.value
    plan = app.state.router.plan(
        req, profile, quality_floor=decision.quality_floor, prefix=prefix_hash(req)
    )

    upstream = app.state.executor.stream(req, profile, plan, meta).__aiter__()
    try:
        first_chunk = await upstream.__anext__()
    except StopAsyncIteration:
        first_chunk = b""
    translator = ResponsesStreamTranslator(payload)

    async def gen():
        try:
            for event in translator.start():
                yield event
            if first_chunk:
                for event in translator.feed(first_chunk):
                    yield event
            async for chunk in upstream:
                for event in translator.feed(chunk):
                    yield event
            for event in translator.finish():
                yield event
        finally:
            aclose = getattr(upstream, "aclose", None)
            if aclose is not None:
                await aclose()

    app.state.metrics.inc("waypost_requests_total", status="stream")
    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/v1/embeddings")
async def embeddings_endpoint(payload: dict = Body(...)):
    """Local embeddings. There is no reason to spend provider quota on them, and
    for privacy: strict they must not be sent to the cloud at all."""
    raw = payload.get("input")
    if raw is None:
        return JSONResponse(status_code=400, content={"error": "missing input field"})
    texts = [raw] if isinstance(raw, str) else list(raw)
    svc = get_embeddings(app)
    vectors = await svc.aencode([str(t) for t in texts])
    app.state.metrics.inc("waypost_embeddings_total", len(texts))
    return {
        "object": "list",
        "model": svc.backend,
        "data": [
            {
                "object": "embedding",
                "index": i,
                "embedding": [round(float(x), 6) for x in vec],
            }
            for i, vec in enumerate(vectors)
        ],
        "usage": {"prompt_tokens": sum(len(str(t)) // 4 for t in texts)},
    }


@app.post("/v1/rerank")
async def rerank_endpoint(payload: dict = Body(...)):
    """RAG chunk selection before sending. Keeping 5 of 20 found
    relevant ones is usually better than any prompt compression."""
    query = payload.get("query")
    documents = payload.get("documents") or []
    if not query or not documents:
        return JSONResponse(
            status_code=400, content={"error": "query and documents fields required"}
        )
    top_n = int(payload.get("top_n") or len(documents))
    docs = [d if isinstance(d, str) else str(d.get("text", d)) for d in documents]
    reranker = get_reranker(app)
    ranked = await asyncio.to_thread(reranker.rank, str(query), docs, top_n)
    app.state.metrics.inc("waypost_rerank_total", len(docs))
    return {
        "object": "list",
        "model": reranker.backend,
        "results": [
            {"index": i, "relevance_score": round(score, 6), "document": docs[i]}
            for i, score in ranked
        ],
    }


# ------------------------------------------------------------- batch API


@app.post("/v1/batches")
async def create_batch(payload: dict = Body(...)):
    if app.state.batch is None:
        return JSONResponse(status_code=404, content={"error": "batch API disabled"})
    requests = payload.get("requests")
    if not requests and (raw := payload.get("input_jsonl")):
        requests = BatchQueue.parse_jsonl(raw)
    if not requests:
        return JSONResponse(
            status_code=400, content={"error": "requests[] or input_jsonl required"}
        )
    window = float(payload.get("completion_window_h") or settings.batch_window_h)
    try:
        return app.state.batch.create(requests, window)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})


@app.get("/v1/batches")
async def list_batches(limit: int = 20):
    if app.state.batch is None:
        return {"data": []}
    return {"object": "list", "data": app.state.batch.list(limit)}


@app.get("/v1/batches/{batch_id}")
async def get_batch(batch_id: str):
    if app.state.batch is None or (b := app.state.batch.get(batch_id)) is None:
        return JSONResponse(status_code=404, content={"error": "no such job"})
    return b


@app.get("/v1/batches/{batch_id}/output")
async def batch_output(batch_id: str):
    if app.state.batch is None or app.state.batch.get(batch_id) is None:
        return JSONResponse(status_code=404, content={"error": "no such job"})
    return {"object": "list", "data": app.state.batch.output(batch_id)}


@app.post("/v1/batches/{batch_id}/cancel")
async def cancel_batch(batch_id: str):
    if app.state.batch is None or app.state.batch.get(batch_id) is None:
        return JSONResponse(status_code=404, content={"error": "no such job"})
    return app.state.batch.cancel(batch_id)


def main() -> None:
    import uvicorn

    uvicorn.run(
        "waypost.server:app", host=settings.host, port=settings.port, reload=False
    )


if __name__ == "__main__":
    main()
