"""Tests for Phases C, D, E, F of Waypost v5 specification.

Covers:
- Phase C: MLX LM server runner, memory caps, launchd plist, warmup script.
- Phase D: Local-first draft with dynamic thinking escalation chain.
- Phase E: Local pool narrowing to 2 routable models.
- Phase F: Hedge repair and hedge routing source diagnostics.
"""
import asyncio
import plistlib
import tempfile
from pathlib import Path

import pytest
import yaml

from waypost.breaker import CircuitBreaker
from waypost.executor import Executor
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
from waypost.telemetry import Telemetry
from waypost.verify import Verifier


def test_phase_c_mlx_runner_and_memory_caps():
    """Verify MLX runner script sets correct memory constants."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("mlx_runner", "scripts/mlx_runner.py")
    assert spec is not None and spec.loader is not None
    mlx_runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mlx_runner)

    assert mlx_runner.CACHE_LIMIT_BYTES == 4 * 1024 * 1024 * 1024  # 4 GB
    assert mlx_runner.MEMORY_LIMIT_BYTES == 22 * 1024 * 1024 * 1024  # 22 GB
    assert mlx_runner.DEFAULT_PORT == 8081


def test_phase_c_launchd_plist_structure():
    """Verify launchd plist file contains expected label and arguments."""
    plist_path = Path("scripts/launchd/com.waypost.mlx.plist")
    assert plist_path.exists()

    with open(plist_path, "rb") as f:
        data = plistlib.load(f)

    assert data["Label"] == "com.waypost.mlx"
    assert "mlx_runner.py" in data["ProgramArguments"][1]
    assert "--port" in data["ProgramArguments"]
    assert "8081" in data["ProgramArguments"]


def test_phase_e_local_pool_narrowing():
    """Verify config/providers.yaml has exactly 1 local provider (mlx) and 1 local Qwen model."""
    with open("config/providers.yaml", "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    local_providers = [p for p in data.get("providers", []) if p.get("is_local")]
    assert len(local_providers) == 1

    provider_names = {p["name"] for p in local_providers}
    assert provider_names == {"mlx"}

    # Total local models count across local providers
    local_models = [m for p in local_providers for m in p.get("models", [])]
    assert len(local_models) == 1
    assert "qwen" in local_models[0]["id"].lower()


@pytest.mark.asyncio
async def test_phase_d_thinking_payload_generation():
    """Verify ChatRequest supports thinking_mode and builds payload with dynamic thinking."""
    from waypost.prefix import build_payload

    req_no_think = ChatRequest(
        messages=[ChatMessage(role="user", content="solve puzzle")],
        thinking_mode=False,
    )
    req_think = ChatRequest(
        messages=[ChatMessage(role="user", content="solve puzzle")],
        thinking_mode=True,
    )

    o_mlx = Offering(
        provider="mlx",
        model_id="qwen3.8-27b",
        base_url="http://127.0.0.1:8081/v1",
        is_local=True,
    )

    payload_no_think = build_payload(req_no_think, o_mlx)
    assert "enable_thinking" not in payload_no_think

    payload_think = build_payload(req_think, o_mlx)
    assert payload_think.get("enable_thinking") is True
    assert payload_think.get("chat_template_kwargs", {}).get("enable_thinking") is True


@pytest.mark.asyncio
async def test_phase_f_hedge_execution_and_cancellation():
    """Verify hedging launches backup on delayed primary and records hedged=True."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "router.db"
        telemetry = Telemetry(db_path=db_path)
        ledger = Ledger(db_path=db_path)
        breaker = CircuitBreaker()
        verifier = Verifier()

        class SlowPrimaryFastBackupAdapter:
            async def complete(self, offering, req, timeout_s, **kwargs):
                if offering.provider == "primary_slow":
                    # Slow primary delays longer than hedge delay (0.5s)
                    await asyncio.sleep(1.0)
                    return {
                        "id": "chatcmpl-slow",
                        "choices": [
                            {"message": {"role": "assistant", "content": "slow"}}
                        ],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                    }
                # Fast backup responds immediately
                return {
                    "id": "chatcmpl-fast",
                    "choices": [{"message": {"role": "assistant", "content": "fast"}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                }

        executor = Executor(
            adapter=SlowPrimaryFastBackupAdapter(),  # type: ignore[arg-type]
            ledger=ledger,
            breaker=breaker,
            telemetry=telemetry,
            verifier=verifier,
            enable_hedging=True,
            enable_exploration=False,
            hedge_budget=1.0,  # allow hedge in test
        )

        o_slow = Offering(
            provider="primary_slow",
            model_id="m_slow",
            base_url="http://slow/v1",
            ttft_p50_ms=50,
        )
        o_fast = Offering(
            provider="backup_fast",
            model_id="m_fast",
            base_url="http://fast/v1",
            ttft_p50_ms=10,
        )
        ledger.register(o_slow)
        ledger.register(o_fast)

        req = ChatRequest(
            messages=[ChatMessage(role="user", content="fast question")],
            latency_class="interactive",
        )
        profile = RequestProfile(
            task_probs={"chat": 1.0},
            est_input_tokens=10,
            est_output_tokens=5,
            tier=Tier.M,
            complexity=0.3,
            language="en",
            required_caps=set(),
            classifier_source="rules",
        )

        cand_slow = Candidate(offering=o_slow, score=1.0, reasons={})
        cand_fast = Candidate(offering=o_fast, score=0.9, reasons={})
        meta = RouterMeta()

        resp = await executor.execute(req, profile, [cand_slow, cand_fast], meta)
        assert resp is not None
        assert meta.hedged is True
        assert resp.choices[0]["message"]["content"] == "fast"

        # Check telemetry logs include hedge routing source
        logs = telemetry.attempt_logs(limit=10)
        assert any(row["routing_source"] == "hedge" for row in logs)
