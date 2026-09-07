"""Tests for Phase K (Model Quality Predictor & Adaptive Quota Threshold) and Phase L (Ensemble & Self-Consistency).
"""
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from waypost.breaker import CircuitBreaker
from waypost.ensemble import (
    fanout_ensemble,
    get_model_family,
    self_consistency_sample,
)
from waypost.ledger import Ledger
from waypost.predictor import ModelQualityPredictor
from waypost.registry import Offering, Registry
from waypost.router import Candidate, Router
from waypost.schemas import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    RequestProfile,
    Tier,
)
from waypost.verify import Verifier


def test_phase_k_predictor_fallback_and_prediction():
    """Verify ModelQualityPredictor prior fallback and weight-based prediction."""
    offering = Offering(
        provider="test",
        model_id="test-model",
        base_url="http://t/v1",
        quality_score=0.75,
        quality={"code": 0.88, "chat": 0.70},
    )
    profile = RequestProfile(
        task_probs={"code": 1.0},
        est_input_tokens=10,
        est_output_tokens=10,
        tier=Tier.M,
    )

    # 1. Fallback when untrained
    pred = ModelQualityPredictor()
    p_pass = pred.predict_p_pass(offering, profile)
    assert p_pass == 0.88

    # 2. Prediction with loaded weights
    with tempfile.TemporaryDirectory() as tmpdir:
        w_path = Path(tmpdir) / "weights.json"
        pred.weights = {
            offering.key: {
                "bias": 0.5,
                "coefs": [0.2, -0.1, 0.4],
                # How many observations stand behind the fit. A prediction
                # is blended into the prior in proportion to this.
                "n": 10_000,
            }
        }
        pred.is_trained = True
        pred.save(w_path)

        pred_loaded = ModelQualityPredictor(w_path)
        assert pred_loaded.is_trained is True

        emb = [1.0, 2.0, 0.5]
        # dot product = 0.5 + (0.2*1 - 0.1*2 + 0.4*0.5) = 0.5 + (0.2 - 0.2 + 0.2) = 0.7
        # sigmoid(0.7) ≈ 0.668
        p_calc = pred_loaded.predict_p_pass(offering, profile, embedding=emb)
        assert 0.65 < p_calc < 0.70

        # The same fit with almost no data behind it must barely move the
        # prior instead of replacing it: routing confidently on a
        # regression over forty rows is worse than routing on the manifest.
        thin = ModelQualityPredictor()
        thin.weights = {
            offering.key: {"bias": 0.5, "coefs": [0.2, -0.1, 0.4], "n": 5}
        }
        thin.is_trained = True
        assert thin.predict_p_pass(offering, profile, embedding=emb) > 0.85


def test_phase_k_adaptive_quota_threshold():
    """Verify adaptive threshold reflects remaining ledger quota."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "r.db"
        registry = Registry()
        ledger = Ledger(db_path=db_path)
        breaker = CircuitBreaker()
        router = Router(registry, ledger, breaker)

        # Baseline when empty
        assert router.local_vs_cloud_threshold() == 0.50

        # When offering has low usage (0% used)
        o = Offering(
            provider="test",
            model_id="m1",
            base_url="http://t/v1",
            limit_rpm=100,
        )
        ledger.register(o)
        thresh_low = router.local_vs_cloud_threshold()
        assert 0.35 <= thresh_low <= 0.40

        # When quota is heavily used (e.g. 90% used)
        for _ in range(90):
            ledger.reserve(o, 10)
        thresh_high = router.local_vs_cloud_threshold()
        assert thresh_high > thresh_low
        assert thresh_high <= 1.0


def test_phase_l_model_family_extraction():
    """Verify model families are accurately parsed for diversity."""
    assert get_model_family("mlx/Qwen3.8-27B-4bit") == "qwen"
    assert get_model_family("openrouter/deepseek/deepseek-chat") == "deepseek"
    assert get_model_family("openrouter/meta-llama/llama-3.3-70b-instruct") == "llama"
    assert get_model_family("openrouter/google/gemma-4-26b-a4b-it") == "gemma"
    assert get_model_family("custom-provider/other-model") == "custom-provider"


@pytest.mark.asyncio
async def test_phase_l_self_consistency_sampling():
    """Verify self-consistency executes N samples and verifies output."""
    mock_executor = MagicMock()
    req = ChatRequest(messages=[ChatMessage(role="user", content="write code")])
    profile = RequestProfile(task_probs={"code": 1.0}, tier=Tier.L)
    cand = Candidate(
        offering=Offering(provider="p1", model_id="m1", base_url="http://1/v1"),
        score=1.0,
        reasons={},
    )

    # 1. First candidate returns bad syntax, second returns clean python
    resp_bad = ChatResponse(
        id="r1",
        model="m1",
        choices=[
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "```python\ndef bad(: pass\n```",
                },
            }
        ],
    )
    resp_good = ChatResponse(
        id="r2",
        model="m1",
        choices=[
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "```python\ndef good():\n    return 42\n```",
                },
            }
        ],
    )

    call_count = 0

    async def mock_execute(r, p, plan, meta=None):
        nonlocal call_count
        call_count += 1
        return resp_good if call_count > 1 else resp_bad

    mock_executor.execute = AsyncMock(side_effect=mock_execute)
    verifier = Verifier()

    result = await self_consistency_sample(
        mock_executor,
        req,
        profile,
        cand,
        n_samples=3,
        verifier=verifier,
    )
    assert result is not None
    assert "def good():" in result.choices[0]["message"]["content"]


@pytest.mark.asyncio
async def test_phase_l_fanout_ensemble():
    """Verify fan-out gathers from distinct model families within timeout."""
    mock_executor = MagicMock()
    req = ChatRequest(messages=[ChatMessage(role="user", content="logic riddle")])
    profile = RequestProfile(task_probs={"reasoning": 1.0}, tier=Tier.L)

    c1 = Candidate(
        offering=Offering(provider="p1", model_id="qwen-27b", base_url="http://1/v1"),
        score=1.0,
        reasons={},
    )
    c2 = Candidate(
        offering=Offering(provider="p2", model_id="qwen-14b", base_url="http://2/v1"),
        score=0.9,
        reasons={},
    )  # duplicate family
    c3 = Candidate(
        offering=Offering(provider="p3", model_id="llama-70b", base_url="http://3/v1"),
        score=0.8,
        reasons={},
    )

    resp = ChatResponse(
        id="fan1",
        model="qwen-27b",
        choices=[{"index": 0, "message": {"role": "assistant", "content": "Solved."}}],
    )
    mock_executor.execute = AsyncMock(return_value=resp)

    res = await fanout_ensemble(
        mock_executor,
        req,
        profile,
        [c1, c2, c3],
        max_proposers=2,
        fanout_timeout_ms=1000,
    )
    assert res is not None
    assert res.choices[0]["message"]["content"] == "Solved."
