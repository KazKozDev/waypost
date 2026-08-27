"""Ladder steps not in the basic set: key rotation, hedging, parallelism
limit, an approximate answer from the cache."""
import asyncio
import os
import tempfile

import httpx
import numpy as np
import pytest

from waypost.breaker import CircuitBreaker
from waypost.cache import SemanticCache
from waypost.classify import classify_l0
from waypost.executor import Executor
from waypost.ledger import Ledger
from waypost.providers.openai_compat import OpenAICompatAdapter
from waypost.registry import Offering, Registry
from waypost.router import Router
from waypost.schemas import (
    Capability,
    ChatMessage,
    ChatRequest,
    RouterError,
    RouterMeta,
)
from waypost.telemetry import Telemetry

CAPS = {Capability.STREAM, Capability.JSON}


def ok_body(name):
    return {
        "id": "chatcmpl-1",
        "model": name,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": f"от {name}"},
            }
        ],
        "usage": {"prompt_tokens": 11, "completion_tokens": 5, "total_tokens": 16},
    }


@pytest.fixture
def db():
    with tempfile.TemporaryDirectory() as d:
        yield os.path.join(d, "t.db")


def build(handler, offerings, db, **kw):
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    registry = Registry(offerings)
    ledger = Ledger(db)
    for o in registry.all():
        ledger.register(o)
    breaker = CircuitBreaker(threshold=3)
    telemetry = Telemetry(db)
    router = Router(registry, ledger, breaker)
    executor = Executor(
        OpenAICompatAdapter(client),
        ledger,
        breaker,
        telemetry,
        retries_per_provider=2,
        timeout_s=5.0,
        **kw,
    )
    return router, executor, ledger


async def run(router, executor, text="привет", **kw):
    req = ChatRequest(messages=[ChatMessage(role="user", content=text)], **kw)
    profile = classify_l0(req)
    meta = RouterMeta()
    plan = router.plan(req, profile)
    return await executor.execute(req, profile, plan, meta), meta


# --------------------------------------------------------- key rotation


@pytest.mark.asyncio
async def test_second_key_takes_over_after_429(db, monkeypatch):
    monkeypatch.setenv("PROV_KEY", "key-one")
    monkeypatch.setenv("PROV_KEY_2", "key-two")
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        auth = request.headers.get("authorization", "")
        seen.append(auth)
        if auth.endswith("key-one"):
            return httpx.Response(
                429, json={"error": "rate limit"}, headers={"retry-after": "60"}
            )
        return httpx.Response(200, json=ok_body("cloud"))

    offerings = [
        Offering(
            provider="cloud",
            model_id="big",
            base_url="http://a:1/v1",
            api_key_env="PROV_KEY",
            caps=CAPS,
            quality_score=0.9,
            limit_rpd=100,
        ),
        Offering(
            provider="local",
            model_id="qwen",
            is_local=True,
            trains_on_data=False,
            base_url="http://c:3/v1",
            caps=CAPS,
            quality_score=0.4,
        ),
    ]
    router, executor, ledger = build(handler, offerings, db)
    resp, meta = await run(router, executor)

    # The first key hit the limit, the second of the same provider answered:
    # the model is the same, the warmed prefix is not lost.
    assert meta.provider == "cloud"
    assert meta.key_index == 1
    assert seen == ["Bearer key-one", "Bearer key-two"]


@pytest.mark.asyncio
async def test_exhausted_keys_fall_through_to_next_candidate(db, monkeypatch):
    monkeypatch.setenv("PROV_KEY", "k1")
    monkeypatch.setenv("PROV_KEY_2", "k2")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.port == 1:
            return httpx.Response(
                429, json={"error": "quota"}, headers={"retry-after": "60"}
            )
        return httpx.Response(200, json=ok_body("local"))

    offerings = [
        Offering(
            provider="cloud",
            model_id="big",
            base_url="http://a:1/v1",
            api_key_env="PROV_KEY",
            caps=CAPS,
            quality_score=0.9,
            limit_rpd=100,
        ),
        Offering(
            provider="local",
            model_id="qwen",
            is_local=True,
            trains_on_data=False,
            base_url="http://c:3/v1",
            caps=CAPS,
            quality_score=0.4,
        ),
    ]
    router, executor, ledger = build(handler, offerings, db)
    _, meta = await run(router, executor)
    assert meta.provider == "local"
    # Both keys are marked exhausted, not just one.
    assert not ledger.can_afford(offerings[0], 10, key_index=0)
    assert not ledger.can_afford(offerings[0], 10, key_index=1)


# ------------------------------------------------------------ hedging


@pytest.mark.asyncio
async def test_hedge_starts_when_primary_is_slow(db):
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.port == 1:
            await asyncio.sleep(2.0)  # silent longer than the TTFT budget
            return httpx.Response(200, json=ok_body("slow"))
        return httpx.Response(200, json=ok_body("fast"))

    offerings = [
        Offering(
            provider="slow",
            model_id="big",
            base_url="http://a:1/v1",
            caps=CAPS,
            quality_score=0.9,
            ttft_p50_ms=100,
            limit_rpd=100,
        ),
        Offering(
            provider="fast",
            model_id="mid",
            base_url="http://b:2/v1",
            caps=CAPS,
            quality_score=0.8,
            ttft_p50_ms=100,
            limit_rpd=100,
        ),
    ]
    router, executor, _ = build(handler, offerings, db, hedge_budget=1.0)
    resp, meta = await run(router, executor)
    assert meta.hedged is True
    assert meta.provider == "fast"
    assert executor.snapshot()["hedges"] == 1


@pytest.mark.asyncio
async def test_batch_class_never_hedges(db):
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.port == 1:
            await asyncio.sleep(1.2)
        return httpx.Response(200, json=ok_body("x"))

    offerings = [
        Offering(
            provider="slow",
            model_id="big",
            base_url="http://a:1/v1",
            caps=CAPS,
            quality_score=0.9,
            ttft_p50_ms=100,
            limit_rpd=100,
        ),
        Offering(
            provider="fast",
            model_id="mid",
            base_url="http://b:2/v1",
            caps=CAPS,
            quality_score=0.8,
            ttft_p50_ms=100,
            limit_rpd=100,
        ),
    ]
    router, executor, _ = build(handler, offerings, db, hedge_budget=1.0)
    # A background task must not double the quota spend.
    _, meta = await run(router, executor, latency_class="batch")
    assert meta.hedged is False
    assert executor.snapshot()["hedges"] == 0


@pytest.mark.asyncio
async def test_cancelled_hedge_returns_its_quota(db):
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.port == 1:
            await asyncio.sleep(2.0)
            return httpx.Response(200, json=ok_body("slow"))
        return httpx.Response(200, json=ok_body("fast"))

    slow = Offering(
        provider="slow",
        model_id="big",
        base_url="http://a:1/v1",
        caps=CAPS,
        quality_score=0.9,
        ttft_p50_ms=100,
        limit_rpd=100,
        limit_tpm=10_000,
    )
    fast = Offering(
        provider="fast",
        model_id="mid",
        base_url="http://b:2/v1",
        caps=CAPS,
        quality_score=0.8,
        ttft_p50_ms=100,
        limit_rpd=100,
        limit_tpm=10_000,
    )
    router, executor, ledger = build(handler, [slow, fast], db, hedge_budget=1.0)
    _, meta = await run(router, executor)
    assert meta.provider == "fast"
    # The losing branch is canceled — its token quota is returned.
    assert ledger.snapshot()["slow/big"]["tpm"] == 10_000


# ---------------------------------------------------- local parallelism


@pytest.mark.asyncio
async def test_local_concurrency_is_capped(db):
    inflight = {"now": 0, "max": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        inflight["now"] += 1
        inflight["max"] = max(inflight["max"], inflight["now"])
        await asyncio.sleep(0.05)
        inflight["now"] -= 1
        return httpx.Response(200, json=ok_body("local"))

    local = Offering(
        provider="local",
        model_id="qwen",
        is_local=True,
        trains_on_data=False,
        base_url="http://c:3/v1",
        caps=CAPS,
        quality_score=0.5,
        concurrency=1,
    )
    router, executor, _ = build(handler, [local], db)
    await asyncio.gather(*(run(router, executor, f"вопрос {i}") for i in range(4)))
    # mlx-lm holds weights in shared memory: two heavy generations at once
    # contend for memory, not speed.
    assert inflight["max"] == 1


# ------------------------------------------- approximate answer from the cache


@pytest.mark.asyncio
async def test_degraded_cache_is_the_last_rung(db):
    """When all refused, an approximate answer is better than emptiness — but it
    must be marked approximate."""
    sc = SemanticCache(enabled=True, threshold=0.95, db_path=db)
    emb = np.array([1.0, 0.0, 0.0])
    sc.put(
        "как дела",
        emb,
        "M:ru:abc",
        {"choices": [{"message": {"role": "assistant", "content": "норм"}}]},
    )

    close_vec = np.array([0.94, 0.34, 0.0])
    assert sc.lookup("как дела", close_vec, "M:ru:abc") is None  # normal lookup
    assert sc.lookup_degraded("как дела", close_vec, "M:ru:abc") is not None
    assert sc.stats["degraded"] == 1


@pytest.mark.asyncio
async def test_all_providers_down_still_raises_without_cache(db):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "down"})

    offerings = [
        Offering(
            provider="cloud",
            model_id="big",
            base_url="http://a:1/v1",
            caps=CAPS,
            quality_score=0.9,
            limit_rpd=100,
        ),
    ]
    router, executor, _ = build(handler, offerings, db)
    with pytest.raises(RouterError) as exc:
        await run(router, executor)
    assert exc.value.status_code == 502
