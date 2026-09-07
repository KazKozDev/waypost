"""Cost of error, pre-empted escalation, and counterfactual evidence.

Three things the router had no notion of: that some wrong answers cost
more than others, that a cheap first attempt is sometimes a tax rather
than a saving, and that it only ever saw the outcome of the model it
happened to pick.
"""
import numpy as np
import pytest

from waypost.families import model_family
from waypost.neighbors import NeighborIndex
from waypost.policy import Policy
from waypost.schemas import (
    Capability,
    ChatMessage,
    ChatRequest,
    RequestProfile,
    Tier,
)
from waypost.stakes import classify_stakes

CAPS = {Capability.STREAM, Capability.JSON}


def req(text="привет", **kw):
    return ChatRequest(messages=[ChatMessage(role="user", content=text)], **kw)


def profile(tier=Tier.M):
    return RequestProfile(task_probs={"chat": 1.0}, tier=tier)


# ------------------------------------------------------------- stakes


def test_an_ordinary_request_is_normal():
    assert classify_stakes(req()).level == "normal"


def test_a_tool_call_raises_the_stakes():
    """A tool call is not a message, it is an action the answer causes.
    Wrong arguments have consequences outside the conversation."""
    r = ChatRequest(
        messages=[ChatMessage(role="user", content="погода в Барселоне")],
        tools=[{"type": "function", "function": {"name": "w", "parameters": {}}}],
    )
    assert classify_stakes(r).level == "high"


def test_structured_output_raises_the_stakes():
    """A schema means a machine parses this. Malformed output does not get
    a puzzled human, it gets an exception downstream."""
    assert classify_stakes(req(response_format={"type": "json_object"})).level == "high"


def test_inference_may_raise_a_declared_level_but_never_lower_it():
    """A caller who says 'low' and then asks for a tool call gets the
    tool-call treatment. A caller who says 'critical' is never talked
    down to by a heuristic."""
    r = ChatRequest(
        messages=[ChatMessage(role="user", content="x")],
        tools=[{"type": "function", "function": {"name": "w", "parameters": {}}}],
        stakes="low",
    )
    assert classify_stakes(r, r.stakes).level == "high"
    assert classify_stakes(req(stakes="critical")).level == "critical"


def test_a_casual_message_is_not_talked_up():
    for text in ("расскажи анекдот", "как дела", "переведи слово cat"):
        assert classify_stakes(req(text)).level == "normal", text


def test_stakes_raise_the_quality_floor():
    """A high-stakes request refuses models a low-stakes one would take."""
    policy = Policy(enable_pii=False, guard=None)
    low = policy.apply(req(stakes="low"), profile())
    high = policy.apply(req(stakes="critical"), profile())
    assert high.quality_floor > low.quality_floor


def test_stakes_move_the_time_budget():
    policy = Policy(enable_pii=False, guard=None)
    assert policy.apply(req(stakes="low"), profile()).deadline_factor < 1.0
    assert policy.apply(req(stakes="critical"), profile()).deadline_factor > 1.0


def test_critical_stakes_pull_the_tier_up():
    policy = Policy(enable_pii=False, guard=None)
    p = profile(Tier.S)
    policy.apply(req(stakes="critical"), p)
    assert p.tier is Tier.L


def test_the_executor_scales_its_budget_by_the_stakes(monkeypatch):
    from waypost.executor import Executor

    ex = Executor.__new__(Executor)
    ex.deadlines = {"interactive": 20.0}
    assert ex.budget_for(req(), 1.0) == 20.0
    assert ex.budget_for(req(), 2.5) == 50.0
    assert ex.budget_for(req(), 0.7) == pytest.approx(14.0)


# -------------------------------------------------- pre-empted escalation


def test_a_neighbourhood_of_failures_reads_as_escalation_risk():
    """The cascade pays for the cheap attempt twice whenever it escalates.
    Where queries like this one have needed escalating, the cheap rung is
    a tax, not a saving."""
    idx = NeighborIndex(k=20, min_neighbors=2, full_trust_n=4)
    for _ in range(12):
        idx.add([1.0, 0.0], "weak/model", 0.0)
    risk, trust = idx.escalation_risk([1.0, 0.0])
    assert risk > 0.8
    assert trust >= 0.6


def test_a_neighbourhood_of_successes_does_not():
    idx = NeighborIndex(k=20, min_neighbors=2, full_trust_n=4)
    for _ in range(12):
        idx.add([1.0, 0.0], "good/model", 1.0)
    risk, _ = idx.escalation_risk([1.0, 0.0])
    assert risk < 0.2


def test_an_empty_neighbourhood_predicts_nothing():
    assert NeighborIndex().escalation_risk([1.0, 0.0]) == (0.0, 0.0)


def test_the_router_starts_higher_when_similar_queries_failed(tmp_path):
    from waypost.breaker import CircuitBreaker
    from waypost.ledger import Ledger
    from waypost.registry import Registry
    from waypost.router import Router

    idx = NeighborIndex(k=20, min_neighbors=2, full_trust_n=4)
    for _ in range(12):
        idx.add([1.0, 0.0], "weak/model", 0.0)
    router = Router(
        Registry([]), Ledger(tmp_path / "l.db"), CircuitBreaker(), neighbors=idx
    )
    p = profile(Tier.S)
    p.embedding = [1.0, 0.0]
    assert router.predicted_tier(p) is Tier.M


def test_thin_evidence_does_not_give_up_the_cascade(tmp_path):
    """Raising the tier on two anecdotes would surrender the saving the
    cascade exists for."""
    from waypost.breaker import CircuitBreaker
    from waypost.ledger import Ledger
    from waypost.registry import Registry
    from waypost.router import Router

    idx = NeighborIndex(k=20, min_neighbors=2, full_trust_n=50)
    for _ in range(3):
        idx.add([1.0, 0.0], "weak/model", 0.0)
    router = Router(
        Registry([]), Ledger(tmp_path / "l.db"), CircuitBreaker(), neighbors=idx
    )
    p = profile(Tier.S)
    p.embedding = [1.0, 0.0]
    assert router.predicted_tier(p) is Tier.S


# ------------------------------------------------------------ families


def test_the_same_weights_at_two_hosts_are_one_family():
    assert model_family("groq/llama-3.3-70b") == model_family(
        "openrouter/meta-llama/llama-3.3-70b-instruct:free"
    )


def test_a_derivative_belongs_to_its_base_family():
    assert model_family("nvidia/nemotron-3-super-120b") == "llama"


def test_a_coder_is_not_its_vendors_chat_model():
    """A coder and a chat model from one vendor fail on different things,
    which is the only thing family diversity is for."""
    assert model_family("qwen2.5-coder-32b") != model_family("qwen3-32b")


def test_an_unknown_model_is_its_own_family():
    a = model_family("acme/mystery-7b")
    b = model_family("other/mystery-7b")
    assert a != b  # over-estimate diversity rather than under-estimate it


# ---------------------------------------------------------- shadow calls


@pytest.mark.asyncio
async def test_a_shadow_records_the_counterfactual(tmp_path):
    """Everything the router learns is the outcome of the model it chose.
    This is the one place the runner-up gets to answer."""
    import httpx

    from waypost.breaker import CircuitBreaker
    from waypost.executor import Executor
    from waypost.ledger import Ledger
    from waypost.providers.openai_compat import OpenAICompatAdapter
    from waypost.registry import Offering
    from waypost.router import Candidate
    from waypost.telemetry import Telemetry

    def handler(request):
        return httpx.Response(
            200,
            json={
                "id": "c1", "model": "alt",
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": "ответ"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3,
                          "total_tokens": 8},
            },
        )

    o = Offering(
        provider="alt", model_id="m", base_url="http://alt/v1", caps=CAPS,
        limit_rpm=100, limit_rpd=100,
    )
    ledger = Ledger(tmp_path / "l.db")
    ledger.register(o)
    idx = NeighborIndex()
    ex = Executor(
        OpenAICompatAdapter(httpx.AsyncClient(transport=httpx.MockTransport(handler))),
        ledger,
        CircuitBreaker(),
        Telemetry(tmp_path / "t.db"),
        neighbors=idx,
    )
    p = profile()
    p.embedding = [0.3, 0.7]
    await ex.shadow(Candidate(o, 1.0, {}), req(), p)
    assert idx.snapshot()["rows"] == 1
    assert ex.snapshot()["shadows"] == 1


@pytest.mark.asyncio
async def test_a_shadow_stands_down_under_quota_pressure(tmp_path):
    """Counterfactual data is worth having, not worth a 429."""
    import httpx

    from waypost.breaker import CircuitBreaker
    from waypost.executor import Executor
    from waypost.ledger import Ledger
    from waypost.providers.openai_compat import OpenAICompatAdapter
    from waypost.registry import Offering
    from waypost.router import Candidate
    from waypost.telemetry import Telemetry

    o = Offering(
        provider="alt", model_id="m", base_url="http://alt/v1", caps=CAPS, limit_rpm=10
    )
    ledger = Ledger(tmp_path / "l.db")
    ledger.register(o)
    for _ in range(9):
        ledger.reserve(o, 1)  # 90% spent

    idx = NeighborIndex()
    ex = Executor(
        OpenAICompatAdapter(
            httpx.AsyncClient(transport=httpx.MockTransport(lambda r: None))
        ),
        ledger,
        CircuitBreaker(),
        Telemetry(tmp_path / "t.db"),
        neighbors=idx,
    )
    p = profile()
    p.embedding = [0.3, 0.7]
    await ex.shadow(Candidate(o, 1.0, {}), req(), p)
    assert idx.snapshot()["rows"] == 0
    assert ex.snapshot()["shadows"] == 0


def test_shadows_stay_inside_their_budget(tmp_path):
    import httpx

    from waypost.breaker import CircuitBreaker
    from waypost.executor import Executor
    from waypost.ledger import Ledger
    from waypost.providers.openai_compat import OpenAICompatAdapter
    from waypost.registry import Offering
    from waypost.router import Candidate
    from waypost.telemetry import Telemetry

    ex = Executor(
        OpenAICompatAdapter(
            httpx.AsyncClient(transport=httpx.MockTransport(lambda r: None))
        ),
        Ledger(tmp_path / "l.db"),
        CircuitBreaker(),
        Telemetry(tmp_path / "t.db"),
        neighbors=NeighborIndex(),
        shadow_budget=0.05,
    )
    ex._counters["requests"] = 100
    ex._counters["shadows"] = 5  # already at 5%
    o = Offering(provider="alt", model_id="m", base_url="http://a/v1", caps=CAPS)
    assert ex.spawn_shadow(Candidate(o, 1.0, {}), req(), profile()) is False
