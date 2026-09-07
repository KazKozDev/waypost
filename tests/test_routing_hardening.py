"""Regressions for the routing hardening pass.

Each test here pins one defect that made the router lose a live provider,
hold a client for minutes, or learn the wrong thing from a refusal.
"""
import asyncio
import os
import tempfile
import time

import httpx
import pytest

from waypost.bandit import Bandit
from waypost.breaker import CircuitBreaker
from waypost.classify import classify_l0
from waypost.executor import Executor
from waypost.latency import LatencyTracker
from waypost.ledger import Ledger
from waypost.providers.openai_compat import OpenAICompatAdapter
from waypost.ratelimit import RateGovernor
from waypost.registry import Offering, Registry
from waypost.router import Router, _diversify, Candidate
from waypost.schemas import Capability, ChatMessage, ChatRequest, RouterMeta
from waypost.telemetry import Telemetry

CAPS = {Capability.STREAM, Capability.JSON}


def ok_body(name="x", content="ok"):
    return {
        "id": "chatcmpl-1",
        "model": name,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ],
        "usage": {"prompt_tokens": 11, "completion_tokens": 5, "total_tokens": 16},
    }


@pytest.fixture
def db():
    with tempfile.TemporaryDirectory() as d:
        yield os.path.join(d, "t.db")


def cloud(name, port, **kw):
    kw.setdefault("caps", CAPS)
    kw.setdefault("quality_score", 0.9)
    kw.setdefault("limit_rpd", 100)
    return Offering(
        provider=name, model_id="m", base_url=f"http://{name}:{port}/v1", **kw
    )


def build(handler, offerings, db, **kw):
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    registry = Registry(offerings)
    ledger = Ledger(db)
    for o in registry.all():
        ledger.register(o)
    breaker = CircuitBreaker(threshold=3)
    telemetry = Telemetry(db)
    latency = LatencyTracker()
    inflight: dict[str, int] = {}
    governor = RateGovernor(ledger)
    router = Router(
        registry, ledger, breaker, latency=latency, inflight=inflight, **{}
    )
    executor = Executor(
        OpenAICompatAdapter(client),
        ledger,
        breaker,
        telemetry,
        retries_per_provider=2,
        timeout_s=30.0,
        latency=latency,
        rate_governor=governor,
        inflight=inflight,
        **kw,
    )
    return router, executor, ledger, breaker, latency, governor


async def run(router, executor, text="привет", **kw):
    req = ChatRequest(messages=[ChatMessage(role="user", content=text)], **kw)
    profile = classify_l0(req)
    meta = RouterMeta()
    plan = router.plan(req, profile)
    return await executor.execute(req, profile, plan, meta), meta


# ------------------------------------------------------- circuit breaker


def test_planning_does_not_consume_the_half_open_token():
    """can_admit() is called for every candidate of every request while the
    router is only planning. It must not take the single probe token — the
    candidate is often never called, and then nothing gives it back."""
    b = CircuitBreaker(threshold=2, base_cooldown_s=0.05)
    b.on_failure("p")
    b.on_failure("p")
    assert b.state("p") == "open"
    time.sleep(0.06)
    assert b.state("p") == "half_open"

    for _ in range(50):  # the router, planning, over and over
        assert b.can_admit("p") is True

    token = b.acquire_probe("p")  # the executor, actually calling
    assert token is not None and not CircuitBreaker.probe_lost_race(token)
    assert b.can_admit("p") is False  # the token is out
    b.release_probe(token)
    assert b.can_admit("p") is True  # ...and comes back


def test_lost_probe_is_reclaimed_by_ttl():
    b = CircuitBreaker(threshold=1, base_cooldown_s=0.05, probe_ttl_s=0.1)
    b.on_failure("p")
    time.sleep(0.06)
    token = b.acquire_probe("p")
    assert token is not None
    assert b.can_admit("p") is False
    time.sleep(0.11)  # the probing task died without a verdict
    assert b.can_admit("p") is True


def test_half_open_failure_reopens_with_doubled_cooldown():
    b = CircuitBreaker(threshold=1, base_cooldown_s=0.05)
    b.on_failure("p")
    time.sleep(0.06)
    token = b.acquire_probe("p")
    assert token is not None
    b.on_failure("p")
    assert b.state("p") == "open"
    assert b._get("p").cooldown_s == pytest.approx(0.1)


@pytest.mark.asyncio
async def test_429_does_not_open_the_breaker(db):
    """Being rate limited is not being unhealthy. Counting 429s as failures
    used to open a live provider after four refusals."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.port == 1:
            return httpx.Response(429, json={"error": "rate limit"},
                                  headers={"retry-after": "1"})
        return httpx.Response(200, json=ok_body("b"))

    a, b = cloud("a", 1, limit_rpm=30), cloud("b", 2, quality_score=0.5)
    router, executor, ledger, breaker, _, _ = build(handler, [a, b], db)
    for i in range(4):
        ledger.penalize(a, 0.0)  # unblock so it keeps being tried
        await run(router, executor, f"q{i}")
    assert breaker.state("a") == "closed"


@pytest.mark.asyncio
async def test_5xx_still_opens_the_breaker(db):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.port == 1:
            return httpx.Response(500, json={"error": "boom"})
        return httpx.Response(200, json=ok_body("b"))

    a, b = cloud("a", 1), cloud("b", 2, quality_score=0.5)
    router, executor, _, breaker, _, _ = build(handler, [a, b], db)
    for i in range(3):
        await run(router, executor, f"q{i}")
    assert breaker.state("a") in ("open", "half_open")


# -------------------------------------------------------------- deadline


@pytest.mark.asyncio
async def test_request_honours_its_deadline(db):
    """Four candidates x two retries x a 120 s timeout is sixteen minutes of
    a client holding a socket. The ladder gets one budget for all of it."""

    async def handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(30.0)
        return httpx.Response(200, json=ok_body("slow"))

    offerings = [cloud(f"p{i}", i + 1) for i in range(4)]
    router, executor, *_ = build(
        handler, offerings, db, deadlines={"interactive": 1.5}, enable_hedging=False
    )
    t0 = time.monotonic()
    with pytest.raises(Exception):
        await run(router, executor)
    assert time.monotonic() - t0 < 4.0
    assert executor.snapshot()["deadline_exceeded"] >= 1


@pytest.mark.asyncio
async def test_deadline_leaves_room_for_the_local_fallback(db):
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.port == 1:
            await asyncio.sleep(10.0)
        return httpx.Response(200, json=ok_body("local"))

    slow = cloud("slow", 1)
    local = Offering(
        provider="local",
        model_id="q",
        base_url="http://local:9/v1",
        is_local=True,
        trains_on_data=False,
        caps=CAPS,
        quality_score=0.4,
    )
    router, executor, *_ = build(
        handler, [slow, local], db, deadlines={"interactive": 6.0},
        enable_hedging=False,
    )
    _, meta = await run(router, executor)
    assert meta.provider == "local"


# ---------------------------------------------------------------- bandit


def test_bandit_evidence_decays(db):
    old = Bandit(db, decay_per_hour=0.5)
    for _ in range(20):
        old.update("chat", "p/m", 1.0)
    fresh = old.quality("chat", "p/m")
    # Rewind the clock a day: the estimate must fall back toward the prior.
    key = ("chat", "p/m")
    old._ts[key] -= 24 * 3600
    aged = old.quality("chat", "p/m", prior=0.5)
    assert fresh > 0.9
    assert aged < fresh
    assert aged == pytest.approx(0.5, abs=0.05)


def test_bandit_confidence_is_bounded(db):
    b = Bandit(db, decay_per_hour=1.0, max_evidence=20.0)
    for _ in range(200):
        b.update("chat", "p/m", 1.0)
    assert b.evidence("chat", "p/m") <= 20.0


@pytest.mark.asyncio
async def test_reward_follows_the_verifier_not_the_status_code(db):
    """A 200 carrying malformed JSON is not a success. Rewarding it taught
    the bandit to prefer exactly the model that breaks the contract."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=ok_body("a", content="this is not json {{"))

    a = cloud("a", 1)
    bandit = Bandit(db)
    router, executor, *_ = build(handler, [a], db, bandit=bandit)
    req = ChatRequest(
        messages=[ChatMessage(role="user", content="дай json")],
        response_format={"type": "json_object"},
    )
    profile = classify_l0(req)
    await executor.execute(req, profile, router.plan(req, profile), RouterMeta())
    assert bandit.quality(profile.task_class, "a/m", prior=0.5) < 0.5


@pytest.mark.asyncio
async def test_refusals_do_not_teach_the_bandit_about_quality(db):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.port == 1:
            return httpx.Response(429, json={"error": "rate limit"},
                                  headers={"retry-after": "1"})
        return httpx.Response(200, json=ok_body("b"))

    a, b = cloud("a", 1), cloud("b", 2, quality_score=0.5)
    bandit = Bandit(db)
    router, executor, *_ = build(handler, [a, b], db, bandit=bandit)
    await run(router, executor)
    # 'a' was rate limited, never judged: no evidence recorded for it.
    assert bandit.evidence("chat", "a/m") == 0.0


# ------------------------------------------------------- quota pressure


def test_pressure_tracks_the_remaining_quota(db):
    """burn_ratio returned 1/limit — a constant. A provider with one
    request left scored exactly like a full one, and the pool cascaded."""
    ledger = Ledger(db)
    o = cloud("a", 1, limit_rpm=10, limit_rpd=None)
    ledger.register(o)
    assert ledger.pressure(o) == pytest.approx(0.0)
    for _ in range(9):
        ledger.reserve(o, 10)
    assert ledger.pressure(o) == pytest.approx(0.9)


def test_pressure_uses_the_best_key(db, monkeypatch):
    monkeypatch.setenv("PRESSURE_KEY", "k1")
    monkeypatch.setenv("PRESSURE_KEY_2", "k2")
    ledger = Ledger(db)
    o = cloud("a", 1, limit_rpm=10, api_key_env="PRESSURE_KEY")
    ledger.register(o)
    assert o.key_count == 2
    for _ in range(10):
        ledger.reserve(o, 10, key_index=0)
    # One key is spent, the other is untouched: the offering is not stuck.
    assert ledger.pressure(o) == pytest.approx(0.0)


def test_blocked_bucket_is_maximum_pressure(db):
    ledger = Ledger(db)
    o = cloud("a", 1, limit_rpm=10)
    ledger.register(o)
    ledger.penalize(o, 60.0)
    assert ledger.pressure(o) == 1.0


# ------------------------------------------------------- rate governor


def test_governor_learns_the_real_limit(db):
    ledger = Ledger(db)
    o = cloud("a", 1, limit_rpm=20)
    ledger.register(o)
    g = RateGovernor(ledger, burst_threshold=2)
    g.on_rate_limit(o, 0)
    assert g.factor(o) == 1.0  # one refusal is noise
    g.on_rate_limit(o, 0)
    assert g.factor(o) == pytest.approx(0.75)
    assert ledger._buckets["a/m#0"]["rpm"].limit == 15


def test_governor_recovers_after_calm(db):
    ledger = Ledger(db)
    o = cloud("a", 1, limit_rpm=20)
    ledger.register(o)
    g = RateGovernor(ledger, burst_threshold=2, calm_s=0.01)
    g.on_rate_limit(o, 0)
    g.on_rate_limit(o, 0)
    time.sleep(0.02)
    g.on_success(o, 0)
    assert g.factor(o) > 0.75


# ------------------------------------------------------------- latency


def test_latency_ema_is_asymmetric():
    """Degradation is believed at once, recovery is earned. One lucky fast
    answer must not put a flapping provider back on top."""
    t = LatencyTracker(alpha_up=0.3, alpha_down=0.1)
    t.seed("p/m", 2000)
    up = t.observe("p/m", 20000)
    t2 = LatencyTracker(alpha_up=0.3, alpha_down=0.1)
    t2.seed("p/m", 20000)
    down = t2.observe("p/m", 2000)
    assert up - 2000 > 20000 - down  # rises faster than it falls


def test_drift_is_relative_to_the_offerings_own_baseline():
    t = LatencyTracker(drift_ratio=2.5, min_samples=3)
    t.seed("fast/m", 500)
    for _ in range(5):
        t.observe("fast/m", 520)
    assert t.drifted("fast/m") is False
    for _ in range(5):
        t.observe("fast/m", 9000)
    assert t.drifted("fast/m") is True


def test_scoring_uses_measured_latency_not_the_manifest(db):
    """The whole point: a provider degrading while still returning 200 OK
    has to lose its place in the plan."""
    ledger = Ledger(db)
    fast = cloud("fast", 1, ttft_p50_ms=800.0, quality_score=0.8)
    slow = cloud("slow", 2, ttft_p50_ms=800.0, quality_score=0.8)
    for o in (fast, slow):
        ledger.register(o)
    latency = LatencyTracker()
    router = Router(
        Registry([fast, slow]),
        ledger,
        CircuitBreaker(),
        latency=latency,
        stochastic=False,
    )
    req = ChatRequest(messages=[ChatMessage(role="user", content="привет")])
    profile = classify_l0(req)
    for _ in range(10):
        latency.observe("slow/m", 24000)
        latency.observe("fast/m", 900)
    plan = router.plan(req, profile)
    assert plan[0].offering.key == "fast/m"


# ------------------------------------------------------------ diversify


def test_plan_does_not_stack_one_provider():
    """Four models behind one host is not a ladder: when the host refuses,
    every rung fails together."""
    made = [
        Candidate(cloud("a", 1), 1.0, {}),
        Candidate(cloud("a", 1), 0.9, {}),
        Candidate(cloud("a", 1), 0.8, {}),
        Candidate(cloud("b", 2), 0.7, {}),
    ]
    out = _diversify(made, per_provider=2)
    assert [c.offering.provider for c in out[:3]] == ["a", "a", "b"]
    assert len(out) == 4  # nothing is lost, the excess moves to the tail


# ------------------------------------------------------------- inflight


@pytest.mark.asyncio
async def test_inflight_is_tracked_and_released(db):
    async def handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.05)
        return httpx.Response(200, json=ok_body("a"))

    a = cloud("a", 1, limit_rpd=1000, limit_rpm=1000)
    router, executor, _, _, _, _ = build(handler, [a], db)
    await asyncio.gather(*(run(router, executor, f"q{i}") for i in range(3)))
    assert executor.inflight == {}  # every entry released
