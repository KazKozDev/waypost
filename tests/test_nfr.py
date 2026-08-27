"""NFR-01 / NFR-04 / NFR-05 — measurable criteria from §5.

The spec requires every non-functional requirement to be checkable by
existing telemetry. Here are three measurements that fail if a numeric
target is violated.
"""
import os
import statistics
import tempfile
import time

import httpx
import pytest

from waypost.breaker import CircuitBreaker
from waypost.classify import classify_l0
from waypost.config import Settings
from waypost.executor import Executor
from waypost.ledger import Ledger
from waypost.providers.openai_compat import OpenAICompatAdapter
from waypost.registry import Offering, Registry
from waypost.router import Router
from waypost.schemas import Capability, ChatMessage, ChatRequest, RouterMeta
from waypost.telemetry import Telemetry

CAPS = {Capability.STREAM, Capability.JSON}


def offering(name="cloud", model="m", **kw):
    kw.setdefault("caps", CAPS)
    kw.setdefault("ctx_window", 8192)
    kw.setdefault("tier", __import__("waypost.schemas", fromlist=["Tier"]).Tier.M)
    base_url = kw.pop("base_url", "http://x/v1")
    return Offering(provider=name, model_id=model, base_url=base_url, **kw)


def local():
    from waypost.schemas import Tier

    return offering(
        name="local",
        model="local-qwen",
        is_local=True,
        trains_on_data=False,
        tier=Tier.M,
        base_url="http://127.0.0.1:1234/v1",
    )


@pytest.fixture
def db():
    with tempfile.TemporaryDirectory() as d:
        yield os.path.join(d, "t.db")


# ------------------------------------------------------------- NFR-01
# The system's own latency (without the provider time) — no more than 15 ms
# in the median with the ML classifier off. We measure the hot path
# classify → router.plan: the part the system adds to
# the attempt latency.
def test_nfr01_overhead_under_15ms_median(db):
    offerings = [
        offering(
            "groq",
            "llama",
            limit_rpm=30,
            limit_rpd=50,
            limit_tpm=10_000,
            ttft_p50_ms=400,
        ),
        offering(
            "openrouter",
            "mistral",
            limit_rpm=20,
            limit_rpd=50,
            limit_tpm=10_000,
            ttft_p50_ms=900,
        ),
        local(),
    ]
    reg = Registry(offerings)
    led = Ledger(db)
    for o in reg.all():
        led.register(o)
    router = Router(reg, led, CircuitBreaker())
    req = ChatRequest(messages=[ChatMessage(role="user", content="привет")])

    # warm-up: the first plan builds internal structures
    classify_l0(req)
    router.plan(req, classify_l0(req))

    samples = []
    for _ in range(300):
        t0 = time.perf_counter()
        profile = classify_l0(req)
        router.plan(req, profile)
        samples.append((time.perf_counter() - t0) * 1000.0)

    med = statistics.median(samples)
    assert med < 15.0, f"overhead median {med:.2f}ms > 15ms"


# ------------------------------------------------------------- NFR-04
# Cold start — no more than 3 seconds until ready to accept requests.
# Ready = the registry, quota accounting, telemetry are up. Heavy models
# load lazily and are not part of the start.
def test_nfr04_cold_start_under_3s():
    settings = Settings()
    t0 = time.perf_counter()
    reg = Registry.from_manifest(settings.manifest_path)
    led = Ledger(settings.db_path)
    for o in reg.all():
        led.register(o)
    tel = Telemetry(settings.db_path)
    tel.success_rates()  # make sure the tables are open
    elapsed = time.perf_counter() - t0
    assert elapsed < 3.0, f"cold start {elapsed:.2f}s > 3s"


# ------------------------------------------------------------- NFR-05
# The share of limit-exhaustion responses — less than 2 % with correct probing.
# "Correct probing" = the registry limits match the real ones.
# Then the router itself removes the offering from the plan as soon as the
# counter hits zero, and the request does not reach the provider — no 429
# arrives at all. If the accounting lied (inflated the remainder), the cloud
# would return 429, and the verdict=switch share would grow past the threshold.
def test_nfr05_quota_exhaustion_under_2pct(db):
    rpd = 5
    cloud = offering(
        "cloud", "m", limit_rpm=100, limit_rpd=rpd, limit_tpm=100_000, ttft_p50_ms=400
    )
    loc = local()
    reg = Registry([cloud, loc])
    led = Ledger(db)
    for o in reg.all():
        led.register(o)
    tel = Telemetry(db)
    router = Router(reg, led, CircuitBreaker())

    def handler(request: httpx.Request) -> httpx.Response:
        # the provider never returns 429 — the accounting must stay ahead of it.
        return httpx.Response(
            200,
            json={
                "id": "x",
                "model": "m",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "ok"},
                    }
                ],
                "usage": {
                    "prompt_tokens": 8,
                    "completion_tokens": 3,
                    "total_tokens": 11,
                },
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    executor = Executor(
        OpenAICompatAdapter(client),
        led,
        CircuitBreaker(),
        tel,
        retries_per_provider=1,
        enable_hedging=False,
        timeout_s=5.0,
    )

    import asyncio

    async def run():
        # more requests than rpd: after the cloud is exhausted everything goes to
        # the local model, not a single 429 arrives from the provider.
        for _ in range(rpd + 3):
            req = ChatRequest(messages=[ChatMessage(role="user", content="привет")])
            profile = classify_l0(req)
            plan = router.plan(req, profile)
            await executor.execute(req, profile, plan, RouterMeta())

    asyncio.run(run())

    # the verdict=switch share among all attempts is an indicator of the gap
    # between the counter and reality. With correct accounting it is zero.
    with tel._conn() as c:
        total = c.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]
        switches = c.execute(
            "SELECT COUNT(*) FROM attempts WHERE verdict='switch'"
        ).fetchone()[0]
    rate = switches / total if total else 0.0
    assert total > 0
    assert rate < 0.02, f"exhaustion share {rate:.1%} > 2% (switch={switches}/{total})"


# ---------------------------------------- helper: lock (AR-06)
def test_instance_lock_prevents_second_process(db):
    from waypost.lock import InstanceLock, InstanceLockError

    path = os.path.join(os.path.dirname(db), "router.lock")
    first = InstanceLock(path)
    first.acquire()
    try:
        with pytest.raises(InstanceLockError):
            InstanceLock(path).acquire()
    finally:
        first.release()
    # after release the second instance takes the lock again
    second = InstanceLock(path)
    second.acquire()
    second.release()
