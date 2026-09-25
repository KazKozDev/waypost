"""Regressions for contextual routing and truthful quality evidence."""

import time

import httpx
import numpy as np
import pytest

from waypost.bandit import Bandit
from waypost.breaker import CircuitBreaker
from waypost.classify import classify_l0
from waypost.executor import Executor
from waypost.feedback import FeedbackCollector, Signal
from waypost.ledger import Ledger
from waypost.neighbors import NeighborIndex
from waypost.predictor import ModelQualityPredictor
from waypost.providers.openai_compat import OpenAICompatAdapter
from waypost.registry import Offering, Registry
from waypost.router import Candidate, Router
from waypost.schemas import Capability, ChatRequest, RouterMeta
from waypost.telemetry import Telemetry


def request(**kwargs):
    return ChatRequest(
        messages=[{"role": "user", "content": "extract a number"}], **kwargs
    )


def body(content):
    return {
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": content,
                },
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def models():
    return [
        Offering(
            provider=p,
            model_id="m",
            base_url=f"http://{p}/v1",
            caps={Capability.JSON, Capability.STREAM},
            free=True,
        )
        for p in ("a", "b")
    ]


@pytest.mark.parametrize("stochastic", [False, True])
@pytest.mark.parametrize("context", ["predictor", "neighbors"])
def test_context_can_reverse_a_learned_class_preference(tmp_path, context, stochastic):
    a, b = models()
    bandit = Bandit(decay_per_hour=1.0)
    for _ in range(100):
        bandit.update("extraction", a.key, 1.0)
        bandit.update("extraction", b.key, 0.0)
    predictor = ModelQualityPredictor()
    index = NeighborIndex(k=40, min_neighbors=2, full_trust_n=5)
    if context == "predictor":
        predictor.weights = {
            a.key: {"bias": 0, "coefs": [-8.0, 8.0], "n": 200},
            b.key: {"bias": 0, "coefs": [8.0, -8.0], "n": 200},
        }
        predictor.is_trained = True
    else:
        for _ in range(10):
            index.add([1.0, 0.0], a.key, 0.0)
            index.add([1.0, 0.0], b.key, 1.0)
            index.add([0.0, 1.0], a.key, 1.0)
            index.add([0.0, 1.0], b.key, 0.0)
    router = Router(
        Registry([a, b]),
        Ledger(tmp_path / "ledger.db"),
        CircuitBreaker(),
        bandit,
        predictor=predictor,
        neighbors=index,
        stochastic=stochastic,
    )
    req = request()
    profile = classify_l0(req)
    for vector, expected in (([1.0, 0.0], b), ([0.0, 1.0], a)):
        profile.embedding = np.array(vector)
        router._refresh_neighbors(profile)
        scores = [router._score(o, req, profile) for o in (a, b)]
        assert max(scores, key=lambda c: c.score).offering.key == expected.key


@pytest.mark.asyncio
async def test_attempts_learn_their_own_verdict_and_feedback_wins(tmp_path):
    a, b = models()
    req = request(response_format={"type": "json_object"})
    profile = classify_l0(req)
    profile.embedding = [1.0, 0.0]
    db = tmp_path / "router.db"
    telemetry = Telemetry(db)
    bandit = Bandit(db, decay_per_hour=1.0)
    neighbors = NeighborIndex(min_neighbors=1)
    ledger = Ledger(db)
    for o in (a, b):
        ledger.register(o)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(
                200, json=body("broken" if r.url.host == "a" else '{"n":42}')
            )
        )
    ) as client:
        executor = Executor(
            OpenAICompatAdapter(client),
            ledger,
            CircuitBreaker(),
            telemetry,
            bandit=bandit,
            neighbors=neighbors,
            enable_hedging=False,
        )
        meta = RouterMeta(request_id="cascade")
        await executor.execute(req, profile, [Candidate(a, 1.0, {})], meta)
        meta.escalated = True
        await executor.execute(req, profile, [Candidate(b, 1.0, {})], meta)
    estimates = neighbors.estimate(profile.embedding)
    assert estimates[a.key][0] == 0.0
    assert estimates[b.key][0] == 1.0
    assert bandit.evidence(profile.task_class, a.key) == 1.0
    assert bandit.evidence(profile.task_class, b.key) == 0.25
    assert neighbors.snapshot()["rows"] == 2
    assert {r["offering"]: r["reward"] for r in telemetry.training_rows()} == {
        a.key: 0.0,
        b.key: 1.0,
    }

    # A user rates the delivered answer, never the earlier failed attempt.
    telemetry.log_feedback(
        request_id="cascade",
        offering=b.key,
        task_class=profile.task_class,
        kind="rated_bad",
        reward=0.0,
        source="explicit",
    )
    # A later implicit signal must not dilute an explicit verdict.
    telemetry.log_feedback(
        request_id="cascade",
        offering=b.key,
        task_class=profile.task_class,
        kind="continued",
        reward=0.75,
        source="implicit",
    )
    rows = {r["offering"]: r for r in telemetry.training_rows()}
    assert rows[a.key]["reward"] == 0.0
    assert rows[b.key]["reward"] == 0.0
    assert rows[b.key]["weight"] == 1.0
    restored = NeighborIndex(min_neighbors=1)
    restored.load(list(rows.values()))
    assert restored.estimate([1.0, 0.0])[b.key][0] == 0.0


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_delivery_without_quality_evidence_does_not_train(tmp_path, stream):
    a = models()[0]
    req = request(stream=stream)
    profile = classify_l0(req)
    profile.embedding = [1.0, 0.0]
    db = tmp_path / "router.db"
    telemetry = Telemetry(db)
    bandit = Bandit(db)
    index = NeighborIndex(min_neighbors=1)
    ledger = Ledger(db)
    ledger.register(a)
    sse = (
        b'data: {"choices":[{"delta":{"content":"not verified"}}]}\n\ndata: [DONE]\n\n'
    )

    def handler(r):
        if stream:
            return httpx.Response(
                200, content=sse, headers={"content-type": "text/event-stream"}
            )
        return httpx.Response(200, json=body("Not independently verified."))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        executor = Executor(
            OpenAICompatAdapter(client),
            ledger,
            CircuitBreaker(),
            telemetry,
            bandit=bandit,
            neighbors=index,
            enable_hedging=False,
        )
        plan = [Candidate(a, 1.0, {})]
        if stream:
            chunks = [
                c async for c in executor.stream(req, profile, plan, RouterMeta())
            ]
            assert b"".join(chunks) == sse
        else:
            await executor.execute(req, profile, plan, RouterMeta(request_id="prose"))
    assert bandit.evidence(profile.task_class, a.key) == 0.0
    assert index.snapshot()["rows"] == 0
    assert telemetry.training_rows() == []


def test_implicit_feedback_has_less_weight_and_explicit_is_not_repeated(tmp_path):
    db = tmp_path / "r.db"
    bandit = Bandit(db, decay_per_hour=1.0)
    collector = FeedbackCollector(Telemetry(db), bandit)
    weak = Signal("r", "a/m", "chat", "continued", 0.75)
    strong = Signal("r", "a/m", "chat", "rated_bad", 0.0, "explicit")
    assert collector.record(weak)
    assert bandit.evidence("chat", "a/m") == 0.25
    assert collector.record(strong)
    assert not collector.record(strong)
    assert bandit.evidence("chat", "a/m") == 1.25
    assert bandit.quality("chat", "a/m") < 0.5


def test_neighbor_confidence_respects_weight_and_age():
    now = time.time()
    index = NeighborIndex(min_neighbors=1, full_trust_n=1, half_life_days=1)
    index.add([1.0, 0.0], "strong", 1.0, now)
    index.add([1.0, 0.0], "weak", 1.0, now, weight=0.25)
    index.add([1.0, 0.0], "old", 1.0, now - 10 * 86400)
    est = index.estimate([1.0, 0.0])
    assert est["old"][1] < est["weak"][1] < est["strong"][1]


@pytest.mark.parametrize(
    "outcome,status,detail,expected",
    [
        ("pass", "ok", '{"weak":true}', None),
        ("pass", "ok", '{"cache":"exact"}', None),
        ("unknown", "timeout", "{}", None),
        ("fail", "refused", '{"refusal":true}', None),
        ("fail", "error", '{"reason":"invalid_json"}', 0.0),
    ],
)
def test_legacy_unknown_evidence_is_not_used_as_a_label(
    tmp_path, outcome, status, detail, expected
):
    from waypost.telemetry import _serialize_embedding

    telemetry = Telemetry(tmp_path / "legacy.db")
    with telemetry._conn() as conn:
        conn.execute(
            "INSERT INTO attempt_log (request_id, attempt_no, embedding, provider, model, "
            "tier, outcome, status, ts, outcome_detail, is_final, is_exploration) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                "old",
                1,
                _serialize_embedding([1.0, 0.0]),
                "cache" if '"cache"' in detail else "a",
                "m",
                "M",
                outcome,
                status,
                time.time(),
                detail,
                1,
                0,
            ),
        )
    rows = telemetry.training_rows()
    if expected is None:
        assert rows == []
    else:
        assert len(rows) == 1
        assert rows[0]["reward"] == expected
