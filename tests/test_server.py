"""Check the HTTP facade end-to-end: OpenAI compatibility, cache, /v1/stats."""
import os
import tempfile

import httpx
import pytest
from fastapi.testclient import TestClient

MANIFEST = """
providers:
  - name: cloud
    base_url: http://a:1/v1
    trains_on_data: true
    free: true
    limits: { rpm: 60, rpd: 100 }
    models:
      - id: big
        tier: L
        ctx_window: 32768
        caps: [tools, json, stream]
        quality_score: 0.9
        ttft_p50_ms: 100
  - name: local
    base_url: http://c:3/v1
    is_local: true
    trains_on_data: false
    free: true
    models:
      - id: qwen
        tier: L
        ctx_window: 32768
        caps: [json, stream]
        quality_score: 0.5
"""


@pytest.fixture
def client(monkeypatch):
    tmp = tempfile.TemporaryDirectory()
    manifest = os.path.join(tmp.name, "providers.yaml")
    with open(manifest, "w") as f:
        f.write(MANIFEST)
    monkeypatch.setenv("ROUTER_MANIFEST_PATH", manifest)
    monkeypatch.setenv("ROUTER_DB_PATH", os.path.join(tmp.name, "r.db"))
    monkeypatch.setenv("ROUTER_ENABLE_DISCOVERY", "false")
    monkeypatch.setenv("ROUTER_ENABLE_EXPLORATION", "false")

    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(f"{request.url.host}:{request.url.port}")
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-x",
                "model": "big",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "42"},
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "total_tokens": 12,
                },
            },
        )

    import waypost.server as srv

    srv.settings = srv.Settings()

    original = httpx.AsyncClient

    def patched(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return original(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", patched)

    with TestClient(srv.app) as c:
        c.upstream_calls = calls
        yield c
    tmp.cleanup()


def test_models_endpoint_exposes_data_policy(client):
    data = client.get("/v1/models").json()["data"]
    assert len(data) == 1
    assert data[0]["id"] == "auto"
    assert data[0]["trains_on_data"] is False


def test_path_normalization_handles_whitespace_and_missing_v1(client):
    r = client.post(
        "/v1 /chat/completions",
        json={
            "model": "auto",
            "messages": [{"role": "user", "content": "сколько будет 6*7"}],
        },
    )
    assert r.status_code == 200

    r2 = client.post(
        "/chat/completions",
        json={
            "model": "auto",
            "messages": [{"role": "user", "content": "сколько будет 6*7"}],
        },
    )
    assert r2.status_code == 200

    r3 = client.get("/v1 /models")
    assert r3.status_code == 200

    r4 = client.get("/models")
    assert r4.status_code == 200


def test_chat_completion_is_openai_shaped(client):
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "auto",
            "messages": [{"role": "user", "content": "сколько будет 6*7"}],
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["choices"][0]["message"]["content"] == "42"
    assert body["usage"]["total_tokens"] == 12
    assert body["router"]["provider"] == "cloud"
    assert body["router"]["cache"] == "miss"


def test_exact_cache_serves_second_identical_request(client):
    payload = {
        "model": "auto",
        "temperature": 0,
        "messages": [{"role": "user", "content": "стабильный вопрос"}],
    }
    first = client.post("/v1/chat/completions", json=payload).json()
    n_after_first = len(client.upstream_calls)
    second = client.post("/v1/chat/completions", json=payload).json()

    assert first["router"]["cache"] == "miss"
    assert second["router"]["cache"] == "exact"
    assert len(client.upstream_calls) == n_after_first  # upstream untouched


def test_nonzero_temperature_bypasses_cache(client):
    payload = {
        "model": "auto",
        "temperature": 0.7,
        "messages": [{"role": "user", "content": "творческий вопрос"}],
    }
    client.post("/v1/chat/completions", json=payload)
    n = len(client.upstream_calls)
    client.post("/v1/chat/completions", json=payload)
    assert len(client.upstream_calls) == n + 1


def test_privacy_strict_routes_local(client):
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "auto",
            "privacy": "strict",
            "messages": [{"role": "user", "content": "мой паспорт 1234"}],
        },
    ).json()
    assert r["router"]["provider"] == "local"


def test_stats_reports_quota_and_cache(client):
    client.post(
        "/v1/chat/completions",
        json={"model": "auto", "messages": [{"role": "user", "content": "привет"}]},
    )
    stats = client.get("/v1/stats").json()
    assert "cloud/big" in stats["quota"]
    assert stats["quota"]["cloud/big"]["rpd"] < 100  # quota was charged
    assert stats["last_24h"][0]["attempts"] >= 1


def test_tools_support_and_message_ordering(client):
    from waypost.prefix import order_messages

    # Ensure chronological order is preserved for conversation with tools
    msgs = [
        {"role": "user", "content": "weather in Paris?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "20C"},
    ]
    ordered = order_messages(msgs)
    assert [m["role"] for m in ordered] == ["user", "assistant", "tool"]

    # System/developer messages move to front
    msgs_with_system = [
        {"role": "user", "content": "hi"},
        {"role": "system", "content": "you are helpful"},
    ]
    ordered_sys = order_messages(msgs_with_system)
    assert [m["role"] for m in ordered_sys] == ["system", "user"]

    # Request with tools endpoint check
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "auto",
            "messages": msgs,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "get weather",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        },
    )
    assert r.status_code == 200


def test_chat_html_endpoints(client):
    for path in ("/", "/chat", "/v1/chat"):
        r = client.get(path)
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]
        assert "Waypost — Chat" in r.text
        assert "How can I help you today?" in r.text
        assert "--accent:" in r.text
        assert "claude-header" in r.text


def test_models_and_stats_html_rendering(client):
    # Unified Models & Providers HTML
    r_models = client.get("/v1/models", headers={"Accept": "text/html"})
    assert r_models.status_code == 200
    assert "Models & Providers" in r_models.text
    assert "claude-header" in r_models.text

    # Local Providers HTML
    r_local = client.get("/v1/local", headers={"Accept": "text/html"})
    assert r_local.status_code == 200
    assert "Models & Providers" in r_local.text or "Local Providers" in r_local.text
    assert "claude-header" in r_local.text

    # Analytics & Stats HTML
    r_stats = client.get("/v1/stats", headers={"Accept": "text/html"})
    assert r_stats.status_code == 200
    assert (
        "Analytics & Stats" in r_stats.text or "Analytics & Telemetry" in r_stats.text
    )
    assert "claude-header" in r_stats.text

    # Pricing HTML
    r_pricing = client.get("/v1/pricing", headers={"Accept": "text/html"})
    assert r_pricing.status_code == 200
    assert "claude-header" in r_pricing.text


def test_routing_profile_and_savings_in_server(client):
    # Test request with X-Waypost-Profile header: privacy_only forces routing to local model
    r = client.post(
        "/v1/chat/completions",
        headers={"X-Waypost-Profile": "privacy_only"},
        json={
            "model": "auto",
            "messages": [{"role": "user", "content": "секретный код"}],
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert "router" in data
    assert data["router"]["routing_profile"] == "privacy_only"
    assert data["model"] == "local/qwen"
    assert "saved_usd" in data["router"]

    # Test /v1/stats JSON contains savings
    r_stats = client.get("/v1/stats")
    assert r_stats.status_code == 200
    stats_data = r_stats.json()
    assert "savings" in stats_data
    assert "total_saved_usd" in stats_data["savings"]
    assert "total_tokens" in stats_data["savings"]

    # Test /v1/stats HTML contains Savings section
    r_stats_html = client.get("/v1/stats", headers={"Accept": "text/html"})
    assert r_stats_html.status_code == 200
    assert "Cost Savings vs Paid Commercial APIs" in r_stats_html.text
