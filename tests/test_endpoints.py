"""Endpoints on top of chat: embeddings, rerank, batches, metrics, idempotency."""
import os
import tempfile
import time

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
        caps: [json, stream]
        quality_score: 0.9
        ttft_p50_ms: 100
  - name: local
    base_url: http://c:3/v1
    is_local: true
    trains_on_data: false
    free: true
    concurrency: 1
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


# ------------------------------------------------------------ embeddings


def test_embeddings_are_local_and_openai_shaped(client):
    before = len(client.upstream_calls)
    r = client.post("/v1/embeddings", json={"input": ["привет", "hello"]}).json()
    assert r["object"] == "list"
    assert len(r["data"]) == 2
    assert r["data"][0]["object"] == "embedding"
    assert len(r["data"][0]["embedding"]) > 16
    # The main property: provider quota is not spent on embeddings.
    assert len(client.upstream_calls) == before


def test_embeddings_accept_single_string(client):
    r = client.post("/v1/embeddings", json={"input": "один текст"}).json()
    assert len(r["data"]) == 1


def test_embeddings_reject_empty_payload(client):
    assert client.post("/v1/embeddings", json={}).status_code == 400


# --------------------------------------------------------------- rerank


def test_rerank_orders_by_relevance(client):
    r = client.post(
        "/v1/rerank",
        json={
            "query": "столица Франции",
            "documents": ["рецепт борща", "столица Франции — Париж", "погода"],
            "top_n": 2,
        },
    ).json()
    assert len(r["results"]) == 2
    assert r["results"][0]["index"] == 1
    assert r["results"][0]["relevance_score"] >= r["results"][1]["relevance_score"]


def test_rerank_requires_query_and_documents(client):
    assert client.post("/v1/rerank", json={"query": "x"}).status_code == 400


# ---------------------------------------------------------------- batches


def test_batch_runs_to_completion(client):
    created = client.post(
        "/v1/batches",
        json={
            "requests": [
                {
                    "custom_id": "a",
                    "body": {"messages": [{"role": "user", "content": "раз"}]},
                },
                {
                    "custom_id": "b",
                    "body": {"messages": [{"role": "user", "content": "два"}]},
                },
            ]
        },
    ).json()
    assert created["status"] == "pending"
    assert created["request_counts"]["total"] == 2

    deadline = time.time() + 15
    while time.time() < deadline:
        state = client.get(f"/v1/batches/{created['id']}").json()
        if state["status"] == "completed":
            break
        time.sleep(0.2)
    assert state["status"] == "completed"

    output = client.get(f"/v1/batches/{created['id']}/output").json()["data"]
    assert {o["custom_id"] for o in output} == {"a", "b"}
    assert all(
        o["response"]["choices"][0]["message"]["content"] == "42" for o in output
    )


def test_batch_accepts_jsonl_and_can_be_cancelled(client):
    jsonl = (
        '{"custom_id":"x","body":{"messages":' '[{"role":"user","content":"привет"}]}}'
    )
    created = client.post("/v1/batches", json={"input_jsonl": jsonl}).json()
    cancelled = client.post(f"/v1/batches/{created['id']}/cancel").json()
    assert cancelled["status"] == "cancelled"


def test_batch_rejects_empty_request(client):
    assert client.post("/v1/batches", json={}).status_code == 400


def test_unknown_batch_is_404(client):
    assert client.get("/v1/batches/batch_nope").status_code == 404


# -------------------------------------------------------- idempotency


def test_idempotency_key_replays_without_second_upstream_call(client):
    payload = {
        "model": "auto",
        "temperature": 0.7,
        "idempotency_key": "abc-123",
        "messages": [{"role": "user", "content": "не кешируется"}],
    }
    first = client.post("/v1/chat/completions", json=payload).json()
    before = len(client.upstream_calls)
    second = client.post("/v1/chat/completions", json=payload).json()
    assert len(client.upstream_calls) == before  # the retry did not go upstream
    assert second["router"]["replay"] is True
    assert first["choices"] == second["choices"]


def test_idempotency_via_header(client):
    payload = {
        "model": "auto",
        "temperature": 0.7,
        "messages": [{"role": "user", "content": "через заголовок"}],
    }
    client.post(
        "/v1/chat/completions", json=payload, headers={"Idempotency-Key": "hdr-1"}
    )
    before = len(client.upstream_calls)
    r = client.post(
        "/v1/chat/completions", json=payload, headers={"Idempotency-Key": "hdr-1"}
    ).json()
    assert len(client.upstream_calls) == before
    assert r["router"]["replay"] is True


# ------------------------------------------------------------- metrics


def test_metrics_expose_prometheus_text(client):
    client.post(
        "/v1/chat/completions",
        json={"model": "auto", "messages": [{"role": "user", "content": "привет"}]},
    )
    body = client.get("/metrics").text
    assert "waypost_requests_total" in body
    assert "waypost_quota_remaining" in body
    assert "waypost_latency_ms_bucket" in body
    assert "# TYPE" in body


def test_jobs_endpoint_lists_control_plane(client):
    jobs = client.get("/v1/jobs").json()["jobs"]
    assert {"discovery", "probe", "health", "purge"} <= set(jobs)
    assert jobs["health"]["failures"] == 0


def test_run_job_manually(client):
    r = client.post("/v1/jobs/health/run").json()
    assert r["job"] == "health"
    assert "breakers" in r["result"]


def test_unknown_job_is_404(client):
    assert client.post("/v1/jobs/nope/run").status_code == 404


def test_stats_include_new_sections(client):
    client.post(
        "/v1/chat/completions",
        json={"model": "auto", "messages": [{"role": "user", "content": "привет"}]},
    )
    stats = client.get("/v1/stats").json()
    for section in (
        "cascade",
        "hedging",
        "executor",
        "batch",
        "jobs",
        "idempotency",
        "embeddings",
        "compression",
    ):
        assert section in stats
    assert stats["cascade"]["answers"] >= 1


# ------------------------------------------------------------- probe monitoring


def test_probe_endpoint_json_and_tier_summary(client):
    r = client.get("/v1/probe")
    assert r.status_code == 200
    data = r.json()
    assert "summary" in data
    assert "by_tier" in data["summary"]
    assert "S" in data["summary"]["by_tier"]
    assert "M" in data["summary"]["by_tier"]
    assert "L" in data["summary"]["by_tier"]
    assert data["total"] >= 2
    assert any(p["tier"] == "L" for p in data["probes"])


def test_probe_endpoint_tier_filtering(client):
    r_l = client.get("/v1/probe?tier=L").json()
    assert r_l["filters"]["tier"] == "L"
    assert all(p["tier"] == "L" for p in r_l["probes"])

    r_s = client.get("/v1/probe?tier=S").json()
    assert r_s["filters"]["tier"] == "S"
    assert len(r_s["probes"]) == 0


def test_probe_endpoint_html(client):
    r = client.get("/v1/probe", headers={"Accept": "text/html"})
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert (
        "Health & Probes" in r.text
        or "Waypost — Provider & Model Health Monitor" in r.text
        or "Waypost — Provider &amp; Model Health Monitor" in r.text
    )
    assert "Tier L" in r.text


def test_probe_run_post(client):
    r = client.post("/v1/probe/run", json={"tier": "L"}).json()
    assert "summary" in r
    assert "probes" in r
    assert r["total"] >= 1
    assert all(p["tier"] == "L" for p in r["probes"])
