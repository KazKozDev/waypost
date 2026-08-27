"""End-to-end checks of the degradation ladder.

Upstreams are replaced with httpx.MockTransport: the provider behavior
(429, 500, 400, success) is set deterministically, without network.
"""
import os
import tempfile

import httpx
import pytest

from waypost.breaker import CircuitBreaker
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
from waypost.classify import classify_l0

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


class Stack:
    """Assembling the router over controlled upstreams."""

    def __init__(self, behaviours: dict[str, object], db: str):
        self.calls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            host = f"{request.url.host}:{request.url.port}"
            self.calls.append(host)
            b = behaviours[host]
            if callable(b):
                b = b(len([c for c in self.calls if c == host]))
            if b == "ok":
                return httpx.Response(200, json=ok_body(host))
            if b == 429:
                return httpx.Response(
                    429, json={"error": "rate limit"}, headers={"retry-after": "1"}
                )
            return httpx.Response(b, json={"error": "upstream said no"})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        self.registry = Registry(self.offerings())
        self.ledger = Ledger(db)
        for o in self.registry.all():
            self.ledger.register(o)
        self.breaker = CircuitBreaker(threshold=3)
        self.telemetry = Telemetry(db)
        self.router = Router(self.registry, self.ledger, self.breaker)
        self.executor = Executor(
            OpenAICompatAdapter(client),
            self.ledger,
            self.breaker,
            self.telemetry,
            retries_per_provider=2,
            timeout_s=5.0,
            enable_exploration=False,
            enable_hedging=False,
        )

    @staticmethod
    def offerings():
        return [
            Offering(
                provider="cloud_a",
                model_id="big",
                base_url="http://a:1/v1",
                caps=CAPS,
                quality_score=0.9,
                ttft_p50_ms=100,
                limit_rpd=100,
            ),
            Offering(
                provider="cloud_b",
                model_id="mid",
                base_url="http://b:2/v1",
                caps=CAPS,
                quality_score=0.7,
                ttft_p50_ms=400,
                limit_rpd=100,
            ),
            Offering(
                provider="local",
                model_id="qwen",
                is_local=True,
                trains_on_data=False,
                base_url="http://c:3/v1",
                caps=CAPS,
                quality_score=0.5,
                ttft_p50_ms=900,
            ),
        ]

    async def run(self, text="привет", **kw):
        req = ChatRequest(messages=[ChatMessage(role="user", content=text)], **kw)
        profile = classify_l0(req)
        meta = RouterMeta()
        plan = self.router.plan(req, profile)
        return await self.executor.execute(req, profile, plan, meta), meta


@pytest.fixture
def db():
    with tempfile.TemporaryDirectory() as d:
        yield os.path.join(d, "t.db")


@pytest.mark.asyncio
async def test_happy_path_picks_best_scored(db):
    s = Stack({"a:1": "ok", "b:2": "ok", "c:3": "ok"}, db)
    resp, meta = await s.run()
    assert meta.provider == "cloud_a"
    assert meta.attempts == 1
    assert resp.usage.total_tokens == 16


@pytest.mark.asyncio
async def test_429_switches_immediately_without_retry(db):
    s = Stack({"a:1": 429, "b:2": "ok", "c:3": "ok"}, db)
    resp, meta = await s.run()
    assert meta.provider == "cloud_b"
    # 429 — no reason to retry the same provider
    assert s.calls.count("a:1") == 1
    assert meta.fallback_path == ["cloud_a/big", "cloud_b/mid"]


@pytest.mark.asyncio
async def test_429_marks_quota_exhausted_for_next_request(db):
    s = Stack({"a:1": 429, "b:2": "ok", "c:3": "ok"}, db)
    await s.run()
    before = len(s.calls)
    await s.run()
    # The second request to the refused provider no longer goes: the ledger remembers
    assert "a:1" not in s.calls[before:]


@pytest.mark.asyncio
async def test_5xx_retries_same_provider_then_switches(db):
    s = Stack({"a:1": 503, "b:2": "ok", "c:3": "ok"}, db)
    resp, meta = await s.run()
    assert s.calls.count("a:1") == 2  # retries_per_provider
    assert meta.provider == "cloud_b"


@pytest.mark.asyncio
async def test_falls_all_the_way_to_local(db):
    s = Stack({"a:1": 429, "b:2": 429, "c:3": "ok"}, db)
    resp, meta = await s.run()
    assert meta.provider == "local"
    assert "от c:3" in resp.choices[0]["message"]["content"]


@pytest.mark.asyncio
async def test_fatal_400_does_not_walk_the_ladder(db):
    s = Stack({"a:1": 400, "b:2": "ok", "c:3": "ok"}, db)
    with pytest.raises(RouterError) as exc:
        await s.run()
    assert exc.value.status_code == 400
    assert s.calls == ["a:1"]  # an invalid request the plan cannot save


@pytest.mark.asyncio
async def test_privacy_strict_never_touches_cloud(db):
    s = Stack({"a:1": "ok", "b:2": "ok", "c:3": "ok"}, db)
    resp, meta = await s.run("секрет", privacy="strict")
    assert meta.provider == "local"
    assert s.calls == ["c:3"]


@pytest.mark.asyncio
async def test_breaker_opens_after_repeated_failures(db):
    s = Stack({"a:1": 503, "b:2": "ok", "c:3": "ok"}, db)
    await s.run()
    await s.run()
    assert s.breaker.state("cloud_a") == "open"
    before = len(s.calls)
    await s.run()
    assert "a:1" not in s.calls[before:]  # no longer hammering the down one


@pytest.mark.asyncio
async def test_all_down_raises_structured_error(db):
    s = Stack({"a:1": 503, "b:2": 503, "c:3": 503}, db)
    with pytest.raises(RouterError) as exc:
        await s.run()
    assert exc.value.status_code == 502
    assert len(exc.value.meta.fallback_path) == 3


@pytest.mark.asyncio
async def test_recovery_after_transient_failure(db):
    """The first call fails, the second succeeds — half_open must close."""
    s = Stack({"a:1": lambda n: 503 if n <= 2 else "ok", "b:2": "ok", "c:3": "ok"}, db)
    _, first = await s.run()
    assert first.provider == "cloud_b"
    _, second = await s.run()
    assert second.provider == "cloud_a"
    assert s.breaker.state("cloud_a") == "closed"


@pytest.mark.asyncio
async def test_model_not_found_400_walks_the_ladder(db):
    """A 400 "Model not found" is not a request defect but an incompatibility
    with the provider: the ladder must move to the next model."""

    def handler(request: httpx.Request) -> httpx.Response:
        host = f"{request.url.host}:{request.url.port}"
        if host == "a:1":
            return httpx.Response(400, json={"error": "Model not found: big"})
        return httpx.Response(200, json=ok_body(host))

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    registry = Registry(Stack.offerings())
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
        retries_per_provider=1,
        timeout_s=5.0,
    )

    req = ChatRequest(messages=[ChatMessage(role="user", content="привет")])
    profile = classify_l0(req)
    meta = RouterMeta()
    plan = router.plan(req, profile)
    _resp, meta = await executor.execute(req, profile, plan, meta), meta
    assert meta.provider == "cloud_b"
    assert meta.fallback_path[0] == "cloud_a/big"
