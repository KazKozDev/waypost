"""Tests for Phase A and Phase B of Waypost v5 specification.

Covers:
- Phase A: Embedding service eager loading and semantic cache activation.
- Phase B: attempt_log persistence in Telemetry, Level-A checks in Verifier,
  and exploration logging in Executor.
"""
import tempfile
import time
from pathlib import Path

import pytest

from waypost.breaker import CircuitBreaker
from waypost.cache import SemanticCache
from waypost.config import Settings
from waypost.embeddings import EmbeddingService
from waypost.executor import Executor, _detect_modality
from waypost.ledger import Ledger
from waypost.registry import Offering
from waypost.router import Candidate
from waypost.schemas import (
    ChatMessage,
    ChatRequest,
    RequestProfile,
    RouterMeta,
    Tier,
)
from waypost.telemetry import AttemptLogEntry, Telemetry
from waypost.verify import Verifier, _check_code_syntax, map_outcome


def test_config_defaults():
    """Verify Phase A config defaults: semantic cache enabled, threshold 0.95, exploration."""
    cfg = Settings()
    assert cfg.enable_semantic_cache is True
    assert cfg.semantic_threshold == 0.95
    # Exploration by duplicate call is off: the router samples the bandit
    # posterior when it selects, so exploring no longer costs a second
    # request against a free-tier quota.
    assert cfg.enable_exploration is False
    assert cfg.explore_rate == 0.10
    assert cfg.explore_floor == 0.02


def test_embedding_service_loading():
    """Verify EmbeddingService instantiates and produces float vectors."""
    svc = EmbeddingService()
    vec = svc.encode_one("Hello world, this is a test query.")
    assert len(vec) == 256
    assert isinstance(vec[0], float)


def test_semantic_cache_with_threshold():
    """Verify SemanticCache uses threshold 0.95 and stores/retrieves embeddings."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_cache.db"
        cache = SemanticCache(enabled=True, threshold=0.95, db_path=db_path, ttl_s=3600)
        svc = EmbeddingService()
        v1 = svc.encode_one("how to configure nginx ssl certificate")
        cache.put(
            "how to configure nginx ssl certificate",
            v1,
            "default",
            {
                "choices": [{"message": {"content": "Use certbot"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        )

        # Exact match query embedding
        hit = cache.lookup("how to configure nginx ssl certificate", v1, "default")
        assert hit is not None
        assert hit["choices"][0]["message"]["content"] == "Use certbot"

        # Completely unrelated query
        v2 = svc.encode_one("what is the recipe for chocolate cake")
        miss = cache.lookup("what is the recipe for chocolate cake", v2, "default")
        assert miss is None


def test_attempt_log_persistence():
    """Verify AttemptLogEntry insertion, serialization, querying, and outcome coverage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "router.db"
        telemetry = Telemetry(db_path=db_path)

        now = time.time()
        embedding = [0.1 * i for i in range(256)]
        entry1 = AttemptLogEntry(
            request_id="req_123",
            attempt_no=1,
            ts=now - 5,
            embedding=embedding,
            embedder_version="minishlab/potion-multilingual-128M",
            l0_labels={"task_type": "code", "tier": "A"},
            l1_prediction=None,
            input_tokens=150,
            modality="text",
            image_count=0,
            audio_duration_s=0.0,
            vision_token_budget=None,
            provider="groq",
            backend="cloud",
            model="llama-3.3-70b-versatile",
            model_version="1.0",
            tier="A",
            thinking_mode=False,
            routing_source="l0",
            is_exploration=False,
            quota_remaining_pct=95.0,
            quota_window_reset_in_s=30,
            quota_binding_limit="requests",
            status="ok",
            error_class="",
            latency_ms=250,
            ttft_ms=80,
            output_tokens=45,
            reasoning_tokens=None,
            peak_memory_mb=None,
            is_final=True,
            outcome="pass",
            outcome_source="hard_check",
            outcome_detail={"weak": True},
        )
        telemetry.log_attempt_row(entry1)

        entry2 = AttemptLogEntry(
            request_id="req_124",
            attempt_no=1,
            ts=now,
            embedding=None,
            embedder_version="minishlab/potion-multilingual-128M",
            l0_labels={"task_type": "chat", "tier": "C"},
            l1_prediction=None,
            input_tokens=50,
            modality="text",
            image_count=0,
            audio_duration_s=0.0,
            vision_token_budget=None,
            provider="local",
            backend="mlx",
            model="qwen3.6:27b",
            model_version="",
            tier="B",
            thinking_mode=True,
            routing_source="l0",
            is_exploration=False,
            quota_remaining_pct=100.0,
            quota_window_reset_in_s=0,
            quota_binding_limit="requests",
            status="error",
            error_class="ProviderError",
            latency_ms=1200,
            ttft_ms=0,
            output_tokens=0,
            reasoning_tokens=None,
            peak_memory_mb=None,
            is_final=False,
            outcome="fail",
            outcome_source="hard_check",
            outcome_detail={"error": "syntax_error"},
        )
        telemetry.log_attempt_row(entry2)

        rows = telemetry.attempt_logs(limit=10)
        assert len(rows) == 2
        # Verify first row deserialization
        r1 = next(r for r in rows if r["request_id"] == "req_123")
        assert r1["model"] == "llama-3.3-70b-versatile"
        assert r1["outcome"] == "pass"
        assert r1["embedding"] is not None
        assert len(r1["embedding"]) == 256
        assert abs(r1["embedding"][0] - 0.0) < 1e-5
        assert abs(r1["embedding"][1] - 0.1) < 1e-5
        assert r1["l0_labels"]["task_type"] == "code"

        # Verify outcome coverage calculation
        coverage = telemetry.outcome_coverage()
        assert coverage["total"] == 2
        assert coverage["coverage_rate"] == 1.0
        assert coverage["by_outcome"]["pass"] == 1
        assert coverage["by_outcome"]["fail"] == 1


def test_level_a_syntax_check():
    """Verify python code syntax checking in verifier."""
    valid_code = "```python\ndef add(a: int, b: int) -> int:\n    return a + b\n```"
    err = _check_code_syntax(valid_code)
    assert err is None

    invalid_code = "```python\ndef broken(:\n    return 1\n```"
    err = _check_code_syntax(invalid_code)
    assert err is not None
    assert "syntax_error" in err

    text_no_python = "This is just plain text with no code blocks."
    err = _check_code_syntax(text_no_python)
    assert err is None


def test_map_outcome_logic():
    """Verify map_outcome categorizes pass, fail, unknown correctly."""
    # Clean pass
    outcome, source, detail = map_outcome(True, "", status="ok")
    assert outcome == "pass"
    assert source == "hard_check"
    assert detail == {"weak": True}

    # Hard check failures
    for reason in [
        "syntax_error",
        "language_mismatch",
        "schema_violation",
        "empty",
        "degenerate",
    ]:
        outcome, source, detail = map_outcome(False, reason, status="error")
        assert outcome == "fail"
        assert source == "hard_check"
        assert detail == {"reason": reason}

    # Refusal
    outcome, source, detail = map_outcome(False, "refusal", status="refused")
    assert outcome == "fail"
    assert source == "hard_check"
    assert detail == {"refusal": True}

    # Rate limited / timeout / server error
    outcome, source, detail = map_outcome(False, "", status="rate_limited")
    assert outcome == "unknown"
    assert source == "upstream"
    assert detail == {"status": "rate_limited"}


def test_modality_detection():
    """Verify _detect_modality parses images, audio, and text."""
    req_text = ChatRequest(
        messages=[ChatMessage(role="user", content="hello text only")]
    )
    mod, img_cnt, aud_dur, vis_bud = _detect_modality(req_text)
    assert mod == "text"
    assert img_cnt == 0
    assert aud_dur == 0.0

    req_image = ChatRequest(
        messages=[
            ChatMessage(
                role="user",
                content=[
                    {"type": "text", "text": "describe this"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,abc"},
                    },
                ],
            )
        ]
    )
    mod, img_cnt, aud_dur, vis_bud = _detect_modality(req_image)
    assert mod == "image"
    assert img_cnt == 1


@pytest.mark.asyncio
async def test_executor_attempt_log_recording():
    """Verify Executor logs attempt rows with full metadata on execution."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "router.db"
        telemetry = Telemetry(db_path=db_path)
        ledger = Ledger(db_path=db_path)
        breaker = CircuitBreaker()
        verifier = Verifier()

        class MockAdapter:
            async def complete(self, offering, req, timeout_s, **kwargs):
                return {
                    "id": "chatcmpl-1",
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "Here is code:\n```python\nx = 1 + 1\n```",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 20, "completion_tokens": 15},
                }

        executor = Executor(
            adapter=MockAdapter(),  # type: ignore[arg-type]
            ledger=ledger,
            breaker=breaker,
            telemetry=telemetry,
            verifier=verifier,
            enable_exploration=False,
        )

        offering = Offering(
            provider="test",
            model_id="m1",
            base_url="http://test/v1",
            is_local=True,
        )
        ledger.register(offering)

        req = ChatRequest(
            messages=[ChatMessage(role="user", content="write python code")]
        )
        profile = RequestProfile(
            task_probs={"code": 1.0},
            est_input_tokens=20,
            est_output_tokens=15,
            tier=Tier.M,
            complexity=0.5,
            language="en",
            required_caps=set(),
            classifier_source="rules",
        )
        cand = Candidate(offering=offering, score=1.0, reasons={})
        meta = RouterMeta()

        resp = await executor.execute(req, profile, [cand], meta)
        assert resp is not None
        assert resp.model == "test/m1"

        logs = telemetry.attempt_logs(limit=10)
        assert len(logs) == 1
        row = logs[0]
        assert row["model"] == "m1"
        assert row["backend"] == "local"
        assert row["outcome"] == "pass"
        assert row["outcome_source"] == "hard_check"
        assert row["input_tokens"] == 20
        assert row["output_tokens"] == 15
