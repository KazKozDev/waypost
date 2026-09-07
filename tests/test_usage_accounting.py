import json
import sqlite3

import httpx
import pytest

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
