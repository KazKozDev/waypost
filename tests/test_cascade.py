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

    def patched(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
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
