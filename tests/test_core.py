import os
import tempfile

import pytest

from waypost.breaker import CircuitBreaker
from waypost.cache import ExactCache, cacheable, canonical_key
from waypost.classify import classify_l0, estimate_tokens
from waypost.ledger import Ledger
from waypost.providers.openai_compat import Verdict, classify_error
from waypost.registry import Offering, Registry
from waypost.router import Router
from waypost.schemas import Capability, ChatMessage, ChatRequest, Tier


def req(text="привет", **kw):
    return ChatRequest(messages=[ChatMessage(role="user", content=text)], **kw)


def offering(name="p", model="m", **kw):
    kw.setdefault("caps", {Capability.STREAM, Capability.JSON})
    o = Offering(provider=name, model_id=model, base_url="http://x/v1", **kw)
    return o


@pytest.fixture
def db():
    with tempfile.TemporaryDirectory() as d:
        yield os.path.join(d, "t.db")


# ------------------------------------------------------------- classifier


def test_short_extraction_is_tier_s():
    p = classify_l0(req("извлеки email из строки: a@b.com"))
    assert p.tier is Tier.S


def test_code_request_escalates():
    p = classify_l0(req("```python\ndef f(x):\n  return x\n```\nпочему падает?"))
    assert p.tier in (Tier.M, Tier.L)
    assert p.task_class == "code"


def test_tools_become_required_capability():
    p = classify_l0(req("посчитай", tools=[{"type": "function"}]))
    assert Capability.TOOLS in p.required_caps


def test_cyrillic_costs_more_tokens():
    assert estimate_tokens("привет мир" * 20) > estimate_tokens("hello world" * 20)


# ----------------------------------------------------------------- ledger


def test_rpd_exhaustion_blocks_offering(db):
    led = Ledger(db)
    o = offering()
    o.limit_rpd = 2
    led.register(o)
    assert led.can_afford(o, 100)
    led.reserve(o, 100)
    led.reserve(o, 100)
    assert not led.can_afford(o, 100)


def test_local_offering_is_unlimited(db):
    led = Ledger(db)
    o = offering(name="local", is_local=True)
    led.register(o)
    for _ in range(1000):
        led.reserve(o, 10_000)
    assert led.can_afford(o, 10_000)


def test_quota_survives_restart(db):
    o = offering()
    o.limit_rpd = 3
    led = Ledger(db)
    led.register(o)
    led.reserve(o, 10)
    led.reserve(o, 10)

    revived = Ledger(db)  # like after a process restart
    revived.register(o)
    assert revived.can_afford(o, 10)
    revived.reserve(o, 10)
    assert not revived.can_afford(o, 10)


def test_429_blocks_even_with_remaining_quota(db):
    led = Ledger(db)
    o = offering()
    o.limit_rpm = 100
    led.register(o)
    led.penalize(o, retry_after_s=30)
    assert not led.can_afford(o, 10)


# ---------------------------------------------------------------- breaker


def test_breaker_opens_then_half_opens():
    b = CircuitBreaker(threshold=2, base_cooldown_s=0.05)
    assert b.allows("p")
    b.on_failure("p")
    b.on_failure("p")
    assert b.state("p") == "open"
    assert not b.allows("p")

    import time

    time.sleep(0.06)
    assert b.state("p") == "half_open"
    assert b.allows("p")  # exactly one probe request
    assert not b.allows("p")
    b.on_success("p")
    assert b.state("p") == "closed"


# ------------------------------------------------------------------ cache


def test_canonical_key_is_order_stable():
    a = canonical_key(req("x", temperature=0.0))
    b = canonical_key(req("x", temperature=0.0))
    assert a == b
    assert a != canonical_key(req("y", temperature=0.0))


def test_session_id_does_not_affect_cache_key():
    a = canonical_key(req("x", temperature=0.0, session_id="s1"))
    b = canonical_key(req("x", temperature=0.0, session_id="s2"))
    assert a == b


def test_nonzero_temperature_is_not_cached():
    assert cacheable(req("x", temperature=0.0))
    assert not cacheable(req("x", temperature=0.7))
    assert not cacheable(req("x", temperature=0.0, stream=True))


def test_cache_roundtrip(db):
    c = ExactCache(db)
    c.put("k", {"ok": True})
    assert c.get("k") == {"ok": True}
    assert c.get("missing") is None
    assert c.hit_rate() == 0.5


# ----------------------------------------------------------------- router


def _router(offerings, db):
    reg = Registry(offerings)
    led = Ledger(db)
    for o in offerings:
        led.register(o)
    return Router(reg, led, CircuitBreaker()), led


def test_privacy_strict_excludes_training_providers(db):
    cloud = offering("cloud", "big", trains_on_data=True)
    local = offering("local", "small", is_local=True, trains_on_data=False)
    r, _ = _router([cloud, local], db)

    plan = r.plan(req("секрет", privacy="strict"), classify_l0(req("секрет")))
    assert [c.offering.provider for c in plan] == ["local"]


def test_missing_capability_filters_out(db):
    no_tools = offering("a", "m1", caps={Capability.STREAM})
    with_tools = offering("b", "m2", caps={Capability.STREAM, Capability.TOOLS})
    r, _ = _router([no_tools, with_tools], db)

    rq = req("посчитай", tools=[{"type": "function"}])
    plan = r.plan(rq, classify_l0(rq))
    assert all(c.offering.provider == "b" for c in plan)


def test_context_overflow_filters_out(db):
    small = offering("a", "m1", ctx_window=100)
    big = offering("b", "m2", ctx_window=200_000)
    r, _ = _router([small, big], db)

    rq = req("длинный текст " * 5000)
    plan = r.plan(rq, classify_l0(rq))
    assert [c.offering.provider for c in plan] == ["b"]


def test_local_always_present_in_plan(db):
    fast = offering("cloud", "m1", quality_score=0.9, ttft_p50_ms=100)
    local = offering("local", "m2", is_local=True, quality_score=0.3, ttft_p50_ms=900)
    r, _ = _router([fast, local], db)

    plan = r.plan(req(), classify_l0(req()))
    assert plan[0].offering.provider == "cloud"
    assert any(c.offering.is_local for c in plan)  # end of the ladder


def test_explicit_dead_local_model_falls_back_without_calling_it(db):
    dead_local = offering(
        "mlx",
        "qwen",
        is_local=True,
        trains_on_data=False,
        runtime_available=False,
        quality_score=0.9,
    )
    cloud = offering("cloud", "safe", quality_score=0.5)
    r, _ = _router([dead_local, cloud], db)

    rq = req(model="mlx/qwen")
    plan = r.plan(rq, classify_l0(rq))

    assert [c.offering.key for c in plan] == ["cloud/safe"]


def test_exhausted_quota_removes_candidate(db):
    a = offering("a", "m1", quality_score=0.9)
    a.limit_rpd = 1
    b = offering("b", "m2", quality_score=0.5)
    r, led = _router([a, b], db)

    assert r.plan(req(), classify_l0(req()))[0].offering.provider == "a"
    led.reserve(a, 100)
    assert r.plan(req(), classify_l0(req()))[0].offering.provider == "b"


def test_batch_class_prefers_quota_thrift(db):
    """The background must not burn the quota needed by interactive traffic."""
    scarce = offering("scarce", "m1", quality_score=0.7, ttft_p50_ms=100)
    scarce.limit_rpd = 10
    roomy = offering("roomy", "m2", quality_score=0.7, ttft_p50_ms=4000)
    roomy.limit_rpd = 10_000
    r, _ = _router([scarce, roomy], db)

    interactive = r.plan(req(latency_class="interactive"), classify_l0(req()))
    batch = r.plan(req(latency_class="batch"), classify_l0(req()))
    assert interactive[0].offering.provider == "scarce"
    assert batch[0].offering.provider == "roomy"


def test_free_only_excludes_paid_models(db):
    paid = offering("paid", "big", quality_score=0.95, free=False)
    free = offering("free", "small", quality_score=0.5, free=True)
    r, _ = _router([paid, free], db)
    r.free_only = True
    plan = r.plan(req(), classify_l0(req()))
    assert all(c.offering.free for c in plan)
    assert not any(c.offering.provider == "paid" for c in plan)

    r.free_only = False
    plan2 = r.plan(req(), classify_l0(req()))
    assert plan2[0].offering.provider == "paid"  # the paid one wins on score


def test_explicit_paid_model_refused_under_free_only(db):
    """An explicitly requested paid model does not bypass free_only: a rule
    must not be outweighed by an explicit name."""
    paid = offering("paid", "big", quality_score=0.95, free=False)
    free = offering("free", "small", quality_score=0.5, free=True)
    r, _ = _router([paid, free], db)
    r.free_only = True
    plan = r.plan(req("сделай", model="big"), classify_l0(req("сделай")))
    assert plan, "a plan of free ones must remain"
    assert all(c.offering.free for c in plan)
    assert not any(c.offering.provider == "paid" for c in plan)


def test_task_specific_quality_routes_by_task(db):
    """A code task must prefer a code model, a chat task a chat model,
    even if the overall quality_score is the same."""
    code_model = offering(
        "a", "code", quality_score=0.7, quality={"code": 0.9, "chat": 0.5}
    )
    chat_model = offering(
        "b", "chat", quality_score=0.7, quality={"code": 0.5, "chat": 0.9}
    )
    r, _ = _router([code_model, chat_model], db)

    code_req = req("```python\ndef f():\n  return 1\n```\nпочему падает?")
    code_plan = r.plan(code_req, classify_l0(code_req))
    assert code_plan[0].offering.provider == "a"  # code model

    chat_req = req("привет, как дела?")
    chat_plan = r.plan(chat_req, classify_l0(chat_req))
    assert chat_plan[0].offering.provider == "b"  # chat model


# ------------------------------------------------------- error normalizer


@pytest.mark.parametrize(
    "status,body,expected",
    [
        (429, "rate limit exceeded", Verdict.SWITCH),
        (200, "", Verdict.OK),
        (503, "upstream unavailable", Verdict.RETRY),
        (400, "invalid schema", Verdict.FATAL),
        (401, "bad key", Verdict.SWITCH),
        (402, "quota exhausted", Verdict.SWITCH),
        (400, "Model not found: grok-2", Verdict.SWITCH),
        (400, "unknown model", Verdict.SWITCH),
    ],
)
def test_error_classification(status, body, expected):
    assert classify_error(status, body).verdict is expected


def test_router_picks_local_small_model_for_tier_s(db):
    """For simple tasks (Tier S), router chooses the small model ('малютка') over heavy L-tier model."""
    local_small = offering(
        "ollama",
        "qwen3:4b-instruct",
        tier=Tier.S,
        is_local=True,
        trains_on_data=False,
        ttft_p50_ms=150.0,
        quality_score=0.5,
    )
    local_heavy = offering(
        "ollama",
        "qwen3.6:27b",
        tier=Tier.L,
        is_local=True,
        trains_on_data=False,
        ttft_p50_ms=1000.0,
        quality_score=0.8,
    )
    r, _ = _router([local_small, local_heavy], db)

    simple_req = req("извлеки число: 42")
    profile = classify_l0(simple_req)
    assert profile.tier is Tier.S

    plan = r.plan(simple_req, profile)
    assert plan[0].offering.model_id == "qwen3:4b-instruct"


def test_router_targets_local_provider_and_switches(db):
    """User can target 'ollama' or 'local' directly, and router selects best model and provides fallback."""
    ollama_s = offering("ollama", "llama3.2:3b", tier=Tier.S, is_local=True)
    ollama_l = offering("ollama", "qwen3.6:27b", tier=Tier.L, is_local=True)
    lmstudio_s = offering("lmstudio", "smollm:135m", tier=Tier.S, is_local=True)

    r, _ = _router([ollama_s, ollama_l, lmstudio_s], db)

    # Targeting provider 'ollama'
    targeted_req = req("привет", model="ollama")
    profile = classify_l0(targeted_req)
    plan = r.plan(targeted_req, profile)

    assert plan[0].offering.provider == "ollama"
    # Fallback to other local provider is present
    providers_in_plan = [c.offering.provider for c in plan]
    assert "ollama" in providers_in_plan
    assert "lmstudio" in providers_in_plan


def test_router_places_local_model_as_fallback_ladder(db):
    """Local model is purely a fallback at the end of the ladder when cloud candidates are present."""
    cloud_m = offering("groq", "compound", tier=Tier.M, is_local=False, quality_score=0.8)
    cloud_l = offering("nvidia", "nemotron", tier=Tier.L, is_local=False, quality_score=0.9)
    local_qwen = offering("mlx", "mlx-community/Qwen3.6-27B-4bit", tier=Tier.L, is_local=True, quality_score=0.85)

    r, _ = _router([cloud_m, cloud_l, local_qwen], db)

    # Standard auto routing
    auto_req = req("напиши эссе про квантовую физику")
    profile = classify_l0(auto_req)
    plan = r.plan(auto_req, profile, limit=3)

    assert len(plan) == 3
    # Cloud models lead the ladder
    assert not plan[0].offering.is_local
    assert not plan[1].offering.is_local
    # Local Qwen is at the end as fallback
    assert plan[2].offering.is_local
    assert plan[2].offering.provider == "mlx"
