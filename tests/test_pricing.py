"""Determining paidness: a mechanism instead of trusting the manifest."""
import pytest

from waypost import pricing
from waypost.discovery import _recheck_prices, is_free_openrouter
from waypost.pricing import Cost, Source
from waypost.registry import Offering, Registry, _parse_offering


def off(provider="p", model="m", **kw):
    return Offering(provider=provider, model_id=model, base_url="http://x/v1", **kw)


# ------------------------------------------------------------- verdicts


def test_undeclared_model_is_not_free():
    """Manifest silence is not "free". This is exactly where money leaked."""
    v = pricing.classify("some/model")
    assert v.cost is Cost.UNKNOWN
    assert not v.is_free


def test_manifest_declaration_is_respected():
    assert pricing.classify("m", declared=True).is_free
    assert pricing.classify("m", declared=False).cost is Cost.PAID


def test_api_price_beats_manifest():
    """The provider said "paid" — the manifest does not argue."""
    v = pricing.classify(
        "m", api_model={"pricing": {"prompt": "0.0000006"}}, declared=True
    )
    assert v.cost is Cost.PAID
    assert v.source is Source.API


def test_zero_prices_are_free():
    v = pricing.classify(
        "x:free",
        api_model={"id": "x:free", "pricing": {"prompt": "0", "completion": "0"}},
    )
    assert v.is_free


def test_free_suffix_without_prices():
    assert pricing.classify("google/gemma:free").is_free
    assert not pricing.classify("google/gemma").is_free


def test_dynamic_price_is_unknown_not_free():
    """-1 on OpenRouter means a "floating rate", not "free"."""
    v = pricing.from_api({"pricing": {"prompt": "-1"}})
    assert v.cost is Cost.UNKNOWN


def test_flat_price_fields_are_parsed():
    v = pricing.from_api({"input_cost_per_token": 0.000002})
    assert v.cost is Cost.PAID


def test_explicit_free_flag_in_api():
    assert pricing.from_api({"free": False}).cost is Cost.PAID
    assert pricing.from_api({"free": True}).cost is Cost.FREE


def test_billed_beats_everything():
    v = pricing.classify(
        "x:free", api_model={"pricing": {"prompt": "0"}}, declared=True, billed=0.004
    )
    assert v.cost is Cost.PAID
    assert v.source is Source.BILLED


@pytest.mark.parametrize(
    "usage,expected",
    [
        ({"cost": 0.0012}, 0.0012),
        ({"total_cost": "0.5"}, 0.5),
        ({"cost_details": {"upstream_inference_cost": 0.02}}, 0.02),
        ({"prompt_tokens": 10}, None),
    ],
)
def test_billed_cost_extraction(usage, expected):
    assert pricing.billed_cost({"usage": usage}) == expected


# -------------------------------------------------------------- registry


def test_manifest_without_free_yields_paid_offering():
    o = _parse_offering({"name": "p", "base_url": "http://x/v1"}, {"id": "m"})
    assert o.free is False
    assert o.free_source == Source.UNKNOWN.value


def test_provider_level_free_propagates():
    o = _parse_offering(
        {"name": "p", "base_url": "http://x/v1", "free": True}, {"id": "m"}
    )
    assert o.free and o.free_source == Source.MANIFEST.value


def test_model_level_free_overrides_provider():
    o = _parse_offering(
        {"name": "p", "base_url": "http://x/v1", "free": True},
        {"id": "m", "free": False},
    )
    assert not o.free


def test_billed_verdict_is_sticky():
    """A bill is irreversible proof: the manifest cannot override it."""
    o = off(free=True)
    o.set_pricing(pricing.Verdict(Cost.PAID, Source.BILLED, "счёт 0.01"))
    assert not o.free
    o.set_pricing(pricing.Verdict(Cost.FREE, Source.MANIFEST, "манифест"))
    assert not o.free


# ------------------------------------------------------------ discovery


def test_openrouter_free_detection():
    assert is_free_openrouter({"id": "a:free", "pricing": {"prompt": "0"}})
    assert not is_free_openrouter({"id": "a", "pricing": {"prompt": "0.001"}})


def test_recheck_demotes_model_that_started_charging():
    o = off("groq", "kimi", free=True)
    reg = Registry([o])
    changed = _recheck_prices(
        reg, o, [{"id": "kimi", "pricing": {"prompt": "0.0000005"}}]
    )
    assert changed and not o.free
    assert o.free_source == Source.API.value


def test_recheck_ignores_models_of_other_providers():
    o = off("groq", "kimi", free=True)
    reg = Registry([o])
    other = off("nvidia", "kimi", free=True)
    assert (
        _recheck_prices(reg, other, [{"id": "kimi", "pricing": {"prompt": "1"}}]) == []
    )
    assert o.free


# ------------------------------------------------- watchdog in the executor


@pytest.mark.asyncio
async def test_billed_response_disables_model_for_next_request(tmp_path):
    """End-to-end scenario: the provider sent a bill → the next request already
    goes to another model. One paid call is the cost of discovery."""
    import httpx
    from waypost.breaker import CircuitBreaker
    from waypost.classify import classify_l0
    from waypost.executor import Executor
    from waypost.ledger import Ledger
    from waypost.providers.openai_compat import OpenAICompatAdapter
    from waypost.router import Router
    from waypost.schemas import Capability, ChatMessage, ChatRequest, RouterMeta
    from waypost.telemetry import Telemetry

    caps = {Capability.STREAM, Capability.JSON}
    sneaky = off("sneaky", "big", free=True, caps=caps, quality_score=0.9)
    honest = off("honest", "small", free=True, caps=caps, quality_score=0.5)

    def handler(request: httpx.Request) -> httpx.Response:
        body = {
            "id": "1",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "ok"},
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
        }
        if "sneaky" in str(request.url) or request.url.host == "x":
            body["usage"]["cost"] = 0.0031
        return httpx.Response(200, json=body)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    db = str(tmp_path / "t.db")
    registry = Registry([sneaky, honest])
    ledger = Ledger(db)
    for o in registry.all():
        ledger.register(o)
    breaker = CircuitBreaker()
    router = Router(registry, ledger, breaker)
    executor = Executor(
        OpenAICompatAdapter(client), ledger, breaker, Telemetry(db), timeout_s=5.0
    )

    req = ChatRequest(messages=[ChatMessage(role="user", content="привет")])
    profile = classify_l0(req)
    plan = router.plan(req, profile)
    assert plan[0].offering is sneaky

    await executor.execute(req, profile, plan, RouterMeta())
    assert not sneaky.free
    assert sneaky.free_source == Source.BILLED.value
    assert executor.snapshot()["billed"] == 1

    plan2 = router.plan(req, profile)
    assert [c.offering for c in plan2] == [honest]


def test_baseline_cost_and_savings_calculation():
    # 1M prompt + 1M completion for Tier M
    cost_m = pricing.baseline_cost(1_000_000, 1_000_000, "M")
    assert cost_m == 18.0  # $3 input + $15 output

    # Tier S
    cost_s = pricing.baseline_cost(1_000_000, 1_000_000, "S")
    assert cost_s == 0.75  # $0.15 input + $0.60 output

    # Tier L
    cost_l = pricing.baseline_cost(1_000_000, 1_000_000, "L")
    assert cost_l == 75.0  # $15 input + $60 output

    # Savings when routed to free tier (actual_cost = 0)
    saved = pricing.calculate_savings(100_000, 20_000, "M", actual_cost=0.0)
    # (100k / 1M * 3) + (20k / 1M * 15) = 0.30 + 0.30 = 0.60
    assert saved == 0.60
