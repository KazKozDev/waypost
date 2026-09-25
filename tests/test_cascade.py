"""End-to-end checks of new features via the HTTP facade: PII → strict, cascade."""
import os
import tempfile
from contextlib import contextmanager

import httpx
from fastapi.testclient import TestClient

MANIFEST = """
providers:
  - name: cheap
    base_url: http://a:1/v1
    trains_on_data: true
    free: true
    limits: { rpm: 60, rpd: 100 }
    models:
      - id: small
        tier: S
        ctx_window: 32768
        caps: [json, stream]
        quality_score: 0.9
        ttft_p50_ms: 50
  - name: good
    base_url: http://b:2/v1
    trains_on_data: true
    free: true
    limits: { rpm: 60, rpd: 100 }
    models:
      - id: big
        tier: L
        ctx_window: 32768
        caps: [json, stream]
        quality_score: 0.9
        ttft_p50_ms: 200
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


@contextmanager
def _make_client(monkeypatch, handler):
    tmp = tempfile.TemporaryDirectory()
    manifest = os.path.join(tmp.name, "providers.yaml")
    with open(manifest, "w") as f:
        f.write(MANIFEST)
    monkeypatch.setenv("ROUTER_MANIFEST_PATH", manifest)
    monkeypatch.setenv("ROUTER_DB_PATH", os.path.join(tmp.name, "r.db"))
    monkeypatch.setenv("ROUTER_ENABLE_DISCOVERY", "false")

    import waypost.server as srv

    srv.settings = srv.Settings()
    original = httpx.AsyncClient

    def with_local_discovery(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "qwen"}]})
        return handler(request)

    def patched(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(with_local_discovery)
        return original(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", patched)
    with TestClient(srv.app) as c:
        yield c
    tmp.cleanup()


def test_pii_forces_strict_without_explicit_flag(monkeypatch):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(f"{request.url.host}:{request.url.port}")
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
                    "prompt_tokens": 5,
                    "completion_tokens": 2,
                    "total_tokens": 7,
                },
            },
        )

    with _make_client(monkeypatch, handler) as c:
        r = c.post(
            "/v1/chat/completions",
            json={
                "model": "auto",
                "messages": [{"role": "user", "content": "мой email a.b@example.com"}],
            },
        ).json()
    # PII found → privacy strict → local model only
    assert r["router"]["provider"] == "local"
    assert calls == ["c:3"]


def test_cascade_escalates_on_bad_cheap_response(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        host = f"{request.url.host}:{request.url.port}"
        if host == "a:1":  # cheap model — empty answer
            content, finish = "", "stop"
        else:  # expensive — normal
            content, finish = "42", "stop"
        return httpx.Response(
            200,
            json={
                "id": "x",
                "model": "m",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": finish,
                        "message": {"role": "assistant", "content": content},
                    }
                ],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 2,
                    "total_tokens": 7,
                },
            },
        )

    with _make_client(monkeypatch, handler) as c:
        r = c.post(
            "/v1/chat/completions",
            json={
                "model": "auto",
                "messages": [{"role": "user", "content": "извлеки число"}],
            },
        ).json()
    # Empty answer from the cheap model → escalation to a higher tier
    assert r["choices"][0]["message"]["content"] == "42"
    assert r["router"]["provider"] == "good"


def test_cascade_falls_back_when_upstream_returns_no_choices(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        host = f"{request.url.host}:{request.url.port}"
        if host == "a:1":
            return httpx.Response(200, json={"choices": [], "usage": {"total_tokens": 1}})
        return httpx.Response(200, json={"choices": [{"index": 0, "finish_reason": "stop",
            "message": {"role": "assistant", "content": "42"}}]})

    with _make_client(monkeypatch, handler) as client:
        response = client.post("/v1/chat/completions", json={
            "model": "auto", "messages": [{"role": "user", "content": "извлеки число"}],
        }).json()
    assert response["choices"][0]["message"]["content"] == "42"
    assert response["router"]["provider"] == "good"


def test_http_cascade_records_each_model_and_accepts_negative_feedback(monkeypatch):
    import waypost.server as srv

    for option in ("ENSEMBLE", "SHADOW", "HEDGING", "SEMANTIC_CACHE", "L1_CLASSIFIER"):
        monkeypatch.setenv(f"ROUTER_ENABLE_{option}", "false")
    monkeypatch.setenv("ROUTER_STOCHASTIC_ROUTING", "false")
    original_classify = srv.classify

    def classify_with_embedding(*args, **kwargs):
        profile = original_classify(*args, **kwargs)
        profile.embedding = [1.0, 0.0]
        return profile

    monkeypatch.setattr(srv, "classify", classify_with_embedding)

    def handler(request):
        content = "invalid JSON" if request.url.host == "a" else '{"n":42}'
        return httpx.Response(200, json={
            "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": content,
            }}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        })

    with _make_client(monkeypatch, handler) as client:
        response = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "extract the number 42"}],
            "session_id": "quality-session", "model": "small",
            "response_format": {"type": "json_object"},
        })
        assert response.status_code == 200
        result = response.json()
        assert result["router"]["escalated"] is True, result["router"]
        assert result["router"]["provider"] == "good"
        assert result["choices"][0]["message"]["content"] == '{"n":42}'
        # The facade must not add a third, negative row for the rescue model.
        assert srv.app.state.neighbors.snapshot()["rows"] == 2
        rows = {r["offering"]: r for r in srv.app.state.telemetry.training_rows()}
        assert rows["cheap/small"]["reward"] == 0.
        assert rows["good/big"]["reward"] == 1.
        assert rows["good/big"]["weight"] == .25
        rated = client.post("/v1/feedback", json={
            "request_id": result["router"]["request_id"],
            "session_id": "quality-session", "rating": "bad",
        })
        assert rated.json()["applied_to_offering"] is True
        rows = {r["offering"]: r for r in srv.app.state.telemetry.training_rows()}
        assert rows["good/big"]["reward"] == 0.
        assert rows["good/big"]["weight"] == 1.
