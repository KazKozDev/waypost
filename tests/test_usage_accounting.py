import json
import sqlite3

import httpx
import pytest

from waypost.bandit import Bandit
from waypost.breaker import CircuitBreaker
from waypost.classify import classify_l0
from waypost.executor import Executor, _StreamUsageTracker
from waypost.ledger import Ledger
from waypost.providers.openai_compat import OpenAICompatAdapter
from waypost.registry import Offering
from waypost.router import Candidate
from waypost.schemas import Capability, ChatMessage, ChatRequest, RouterMeta
from waypost.telemetry import Telemetry


def _request(*, stream: bool = False) -> ChatRequest:
    return ChatRequest(
        messages=[ChatMessage(role="user", content="hello")], stream=stream
    )


def test_stream_usage_tracker_handles_split_sse_chunks():
    tracker = _StreamUsageTracker()
    tracker.feed(b'data: {"choices":[]}\n\ndata: {"usage":{"prompt_tok')
    tracker.feed(b'ens":12,"completion_tokens":3,"total_tokens":15}}\n')
    tracker.feed(b"\ndata: [DONE]\n\n")
    tracker.finish()

    assert tracker.body is not None
    assert tracker.body["usage"]["prompt_tokens"] == 12
    assert tracker.body["usage"]["completion_tokens"] == 3


@pytest.mark.asyncio
async def test_stream_usage_is_persisted_and_updates_ledger(tmp_path):
    sse = (
        b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
        b'data: {"choices":[],"usage":{"prompt_tokens":12,'
        b'"completion_tokens":3,"total_tokens":15}}\n\n'
        b"data: [DONE]\n\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=sse, headers={"content-type": "text/event-stream"})

    db = tmp_path / "router.db"
    offering = Offering(
        provider="cloud",
        model_id="model",
        base_url="http://provider.test/v1",
        caps={Capability.STREAM},
        limit_tpm=10_000,
        free=True,
    )
    ledger = Ledger(db)
    ledger.register(offering)
    telemetry = Telemetry(db)
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    executor = Executor(
        OpenAICompatAdapter(client),
        ledger,
        CircuitBreaker(),
        telemetry,
        enable_hedging=False,
        enable_exploration=False,
    )
    req = _request(stream=True)
    profile = classify_l0(req)
    chunks = [
        chunk
        async for chunk in executor.stream(
            req, profile, [Candidate(offering, 1.0, {})], RouterMeta()
        )
    ]
    await client.aclose()

    assert b"".join(chunks) == sse
    with sqlite3.connect(db) as conn:
        row = conn.execute(
            "SELECT tokens, prompt_tokens, completion_tokens, actual_cost_usd, "
            "usage_measured FROM attempts WHERE verdict = 'ok_stream'"
        ).fetchone()
    assert row == (15, 12, 3, 0.0, 1)

    stats = telemetry.savings_stats()
    assert stats["total_tokens"] == 15
    assert stats["usage_coverage_pct"] == 100.0
    assert stats["exact_savings_requests"] == 1


@pytest.mark.asyncio
async def test_unsupported_stream_options_retries_without_them():
    payloads = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        payloads.append(payload)
        if "stream_options" in payload:
            return httpx.Response(400, json={"error": "unknown stream_options"})
        return httpx.Response(200, content=b"data: [DONE]\n\n")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    adapter = OpenAICompatAdapter(client)
    offering = Offering(
        provider="cloud",
        model_id="model",
        base_url="http://provider.test/v1",
        is_local=False,
    )
    chunks = [chunk async for chunk in adapter.stream(offering, _request(stream=True), 5)]
    await client.aclose()

    assert chunks == [b"data: [DONE]\n\n"]
    assert payloads[0]["stream_options"] == {"include_usage": True}
    assert "stream_options" not in payloads[1]


def _stream_setup(tmp_path, sse: bytes, *, bandit=None):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=sse, headers={"content-type": "text/event-stream"}
        )

    db = tmp_path / "router.db"
    offering = Offering(
        provider="cloud",
        model_id="model",
        base_url="http://provider.test/v1",
        caps={Capability.STREAM},
        limit_tpm=10_000,
        free=True,
    )
    ledger = Ledger(db)
    ledger.register(offering)
    telemetry = Telemetry(db)
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    executor = Executor(
        OpenAICompatAdapter(client),
        ledger,
        CircuitBreaker(),
        telemetry,
        bandit=bandit,
        enable_hedging=False,
        enable_exploration=False,
    )
    return db, client, executor, offering


async def _drain_stream(executor, offering):
    req = _request(stream=True)
    profile = classify_l0(req)
    chunks = [
        chunk
        async for chunk in executor.stream(
            req, profile, [Candidate(offering, 1.0, {})], RouterMeta()
        )
    ]
    return req, profile, chunks


def test_stream_tracker_taps_answer_text():
    tracker = _StreamUsageTracker()
    tracker.feed(b'data: {"choices":[{"delta":{"content":"hel"}}]}\n\n')
    tracker.feed(
        b'data: {"choices":[{"delta":{"content":"lo"},"finish_reason":"stop"}]}\n\n'
    )
    tracker.feed(b"data: [DONE]\n\n")
    tracker.finish()

    assert tracker.text == "hello"
    assert tracker.finish_reason == "stop"
    assert tracker.saw_chat_shape is True


def test_stream_tracker_ignores_non_chat_shapes():
    tracker = _StreamUsageTracker()
    tracker.feed(b'data: {"foo": 1}\n\ndata: [DONE]\n\n')
    tracker.finish()

    assert tracker.saw_chat_shape is False
    assert tracker.text == ""


@pytest.mark.asyncio
async def test_empty_stream_is_a_logged_failure(tmp_path):
    sse = (
        b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
        b"data: [DONE]\n\n"
    )
    bandit = Bandit()
    db, client, executor, offering = _stream_setup(tmp_path, sse, bandit=bandit)
    req, profile, _ = await _drain_stream(executor, offering)
    await client.aclose()

    with sqlite3.connect(db) as conn:
        row = conn.execute(
            "SELECT ok, verdict FROM attempts ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
        log_row = conn.execute(
            "SELECT outcome, outcome_detail, status FROM attempt_log"
        ).fetchone()
    assert row[0] == 0
    assert row[1] == "empty"
    assert log_row[0] == "fail"
    assert log_row[2] == "error"
    # An empty answer is quality evidence against the model.
    assert bandit.quality(profile.task_class, offering.key) < 0.5


@pytest.mark.asyncio
async def test_truncated_stream_is_a_logged_failure(tmp_path):
    sse = (
        b'data: {"choices":[{"delta":{"content":"partial"},"finish_reason":"length"}]}\n\n'
        b"data: [DONE]\n\n"
    )
    db, client, executor, offering = _stream_setup(tmp_path, sse)
    _, _, _ = await _drain_stream(executor, offering)
    await client.aclose()

    with sqlite3.connect(db) as conn:
        row = conn.execute(
            "SELECT ok, verdict FROM attempts ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
        log_row = conn.execute(
            "SELECT outcome, status FROM attempt_log"
        ).fetchone()
    assert row[0] == 0
    assert row[1] == "truncated"
    assert log_row == ("fail", "truncated")


@pytest.mark.asyncio
async def test_ok_stream_writes_attempt_log_without_training(tmp_path):
    sse = (
        b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
        b'data: {"choices":[],"usage":{"prompt_tokens":12,'
        b'"completion_tokens":3,"total_tokens":15}}\n\n'
        b"data: [DONE]\n\n"
    )
    bandit = Bandit()
    db, client, executor, offering = _stream_setup(tmp_path, sse, bandit=bandit)
    req, profile, _ = await _drain_stream(executor, offering)
    await client.aclose()

    with sqlite3.connect(db) as conn:
        row = conn.execute(
            "SELECT ok, verdict FROM attempts ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
        log_row = conn.execute(
            "SELECT outcome, outcome_detail, ttft_ms FROM attempt_log"
        ).fetchone()
    assert row[0] == 1
    assert row[1] == "ok_stream"
    assert log_row[0] == "pass"
    detail = json.loads(log_row[1])
    assert detail["stream"] is True
    assert detail["quality_reward"] is None
    assert log_row[2] >= 0
    # Plain prose is unknown quality, not evidence for anyone.
    assert bandit.evidence(profile.task_class, offering.key) == 0.0


def test_legacy_totals_are_kept_but_marked_as_estimated(tmp_path):
    telemetry = Telemetry(tmp_path / "router.db")
    profile = classify_l0(_request())
    telemetry.log_attempt(
        "cloud/legacy", profile, ok=True, verdict="ok", tokens=100
    )
    telemetry.log_attempt(
        "cloud/new",
        profile,
        ok=True,
        verdict="ok",
        tokens=20,
        prompt_tokens=15,
        completion_tokens=5,
        actual_cost_usd=0.0,
        usage_measured=True,
    )

    # Reopening runs the same non-destructive backfill used for an existing
    # pre-migration database.
    telemetry = Telemetry(tmp_path / "router.db")
    stats = telemetry.savings_stats()
    assert stats["total_tokens"] == 120
    assert stats["measured_requests"] == 2
    assert stats["exact_savings_requests"] == 1
    assert stats["estimated_savings_requests"] == 1
    assert stats["baseline"] == "Waypost commercial baseline v1"


@pytest.mark.asyncio
async def test_null_completion_token_details_do_not_crash(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-null-details",
                "model": "model",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "weather", "arguments": "{}"},
                                }
                            ],
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 4,
                    "total_tokens": 14,
                    "completion_tokens_details": None,
                },
            },
        )

    db = tmp_path / "router.db"
    offering = Offering(
        provider="cloud",
        model_id="model",
        base_url="http://provider.test/v1",
        caps={Capability.TOOLS},
        limit_tpm=10_000,
        free=True,
    )
    ledger = Ledger(db)
    ledger.register(offering)
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    executor = Executor(
        OpenAICompatAdapter(client),
        ledger,
        CircuitBreaker(),
        Telemetry(db),
        enable_hedging=False,
        enable_exploration=False,
    )
    req = _request()
    response = await executor.execute(
        req,
        classify_l0(req),
        [Candidate(offering, 1.0, {})],
        RouterMeta(),
    )
    await client.aclose()

    assert response.choices[0]["finish_reason"] == "tool_calls"
    assert response.usage.total_tokens == 14
