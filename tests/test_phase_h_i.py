"""Tests for Phase H (Beta Measurement) and Phase I (Modality Pre-filtering & Decomposition).
"""
import importlib.util
import tempfile
from pathlib import Path

import pytest

from waypost.beta import compute_beta_metrics
from waypost.breaker import CircuitBreaker
from waypost.decompose import (
    analyze_multimodal_decomposition,
    get_vision_token_budget,
    is_deictic_prompt,
)
from waypost.ledger import Ledger
from waypost.registry import Offering, Registry
from waypost.router import Router
from waypost.schemas import (
    Capability,
    ChatMessage,
    ChatRequest,
    RequestProfile,
    Tier,
)


def test_phase_h_beta_metrics_computation():
    """Verify p_best, beta, ceiling, and verdict calculation."""
    eval_matrix = {
        "m1": [True, True, False, False, True],  # 3/5 = 60%
        "m2": [False, True, True, False, True],  # 3/5 = 60%
        "m3": [True, False, False, False, False],  # 1/5 = 20%
    }
    # Query index 3 is failed by all 3 models -> 1 / 5 = 20% Beta
    metrics = compute_beta_metrics(eval_matrix, gain_threshold=0.05)

    assert metrics.total_queries == 5
    assert len(metrics.models) == 3
    assert metrics.p_best == 0.60
    assert metrics.beta == 0.20
    assert metrics.ceiling == 0.80
    assert metrics.potential_gain == pytest.approx(0.20, abs=1e-3)
    assert "Ensemble PROMISES improvement" in metrics.verdict
    assert metrics.all_failed_query_indices == [3]


def test_phase_h_beta_benchmark_script():
    """Verify run_beta_benchmark outputs report and writes valid JSON."""
    spec = importlib.util.spec_from_file_location(
        "measure_beta", "scripts/measure_beta.py"
    )
    assert spec is not None and spec.loader is not None
    mb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mb)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_json = Path(tmpdir) / "beta.json"
        data = mb.run_beta_benchmark(output_path=out_json, target_queries=30)
        assert out_json.exists()
        assert data["total_queries"] == 30
        assert data["models_count"] > 0
        assert "p_best" in data
        assert "beta" in data
        assert "ceiling" in data


def test_phase_i_vision_and_audio_hard_filtering():
    """Verify Router._passes performs hard filtering on vision, audio, tools, and JSON schema."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "r.db"
        registry = Registry()
        ledger = Ledger(db_path=db_path)
        breaker = CircuitBreaker()
        router = Router(registry, ledger, breaker, free_only=False)

        o_text = Offering(
            provider="test",
            model_id="text-only",
            base_url="http://t/v1",
            caps=[Capability.STREAM],
        )
        o_vision = Offering(
            provider="test",
            model_id="vision-model",
            base_url="http://v/v1",
            caps=[Capability.VISION, Capability.STREAM],
        )
        o_gemma_26b = Offering(
            provider="test",
            model_id="gemma-4-26b-a4b-it",
            base_url="http://g/v1",
            caps=[Capability.VISION, Capability.STREAM],
        )
        o_tools = Offering(
            provider="test",
            model_id="tool-model",
            base_url="http://tool/v1",
            caps=[Capability.TOOLS, Capability.JSON],
        )

        for o in (o_text, o_vision, o_gemma_26b, o_tools):
            ledger.register(o)

        profile = RequestProfile(
            task_probs={"chat": 1.0},
            est_input_tokens=10,
            est_output_tokens=10,
            tier=Tier.M,
        )

        # 1. Vision Request
        req_vision = ChatRequest(
            messages=[
                ChatMessage(
                    role="user",
                    content=[
                        {"type": "text", "text": "describe image"},
                        {"type": "image_url", "image_url": {"url": "http://img/1.png"}},
                    ],
                )
            ]
        )
        assert router._passes(o_text, req_vision, profile) is False
        assert router._passes(o_vision, req_vision, profile) is True

        # 2. Audio Request
        req_audio = ChatRequest(
            messages=[
                ChatMessage(
                    role="user",
                    content=[{"type": "input_audio", "input_audio": {"data": "abc"}}],
                )
            ]
        )
        # Gemma 4 26b-a4b cannot accept audio
        assert router._passes(o_gemma_26b, req_audio, profile) is False

        # 3. Tools Request
        req_tools = ChatRequest(
            messages=[ChatMessage(role="user", content="run tool")],
            tools=[{"type": "function", "function": {"name": "calc"}}],
        )
        assert router._passes(o_text, req_tools, profile) is False
        assert router._passes(o_tools, req_tools, profile) is True

        # 4. JSON Schema Request
        req_json = ChatRequest(
            messages=[ChatMessage(role="user", content="return json")],
            response_format={"type": "json_object"},
        )
        assert router._passes(o_text, req_json, profile) is False
        assert router._passes(o_tools, req_json, profile) is True


def test_phase_i_decomposition_analysis():
    """Verify separable vs deictic multimodal task decomposition."""
    # 1. Separable OCR task
    req_sep = ChatRequest(
        messages=[
            ChatMessage(
                role="user",
                content=[
                    {
                        "type": "text",
                        "text": "Extract all text and summarize the report.",
                    },
                    {"type": "image_url", "image_url": {"url": "http://img/doc.png"}},
                ],
            )
        ]
    )
    res_sep = analyze_multimodal_decomposition(req_sep)
    assert res_sep.is_separable is True
    assert res_sep.stage == "extraction_first"

    # 2. Deictic prompt (non-separable)
    assert (
        is_deictic_prompt("What is circled in red on the left side of the chart?")
        is True
    )
    req_deictic = ChatRequest(
        messages=[
            ChatMessage(
                role="user",
                content=[
                    {"type": "text", "text": "What is circled in red on the diagram?"},
                    {"type": "image_url", "image_url": {"url": "http://img/chart.png"}},
                ],
            )
        ]
    )
    res_deictic = analyze_multimodal_decomposition(req_deictic)
    assert res_deictic.is_separable is False
    assert res_deictic.stage == "single_shot"

    # 3. Vision Token Budgets (Spec I.2)
    assert get_vision_token_budget("ocr_document") == 1120
    assert get_vision_token_budget("diagram_analysis") == 560
    assert get_vision_token_budget("scene_description") == 280
    assert get_vision_token_budget("video_frame") == 70
