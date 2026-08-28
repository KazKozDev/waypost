"""Check the auto-discovery of free models."""
import pytest

from waypost.discovery import _caps, _tier, discover_provider, is_free_openrouter
from waypost.providers.openai_compat import ProviderError, Verdict
from waypost.registry import Offering, Registry
from waypost.schemas import Capability, Tier


def test_manifest_loads_all_providers():
    """The real manifest parses: all providers and models are valid."""
    reg = Registry.from_manifest("config/providers.yaml")
    providers = {o.provider for o in reg.all()}
    assert {
        "mlx",
        "openrouter",
        "groq",
        "nvidia",
        "cerebras",
        "mistral",
        "gemini",
    } <= providers
    assert any(o.is_local for o in reg.all())
    assert all(o.ctx_window > 0 for o in reg.all())
    assert all(o.tier in Tier for o in reg.all())


class FakeAdapter:
    def __init__(self, models):
        self.models = models

    async def list_models(self, o, timeout=30.0):
        return self.models


class FailingLocalAdapter:
    async def list_models(self, o, timeout=30.0):
        raise ProviderError(Verdict.RETRY, 408, "connection refused")


def test_is_free_openrouter():
    assert is_free_openrouter({"id": "x:free", "pricing": {"prompt": "0"}})
    assert is_free_openrouter({"id": "y", "pricing": {"prompt": "0"}})
    assert not is_free_openrouter({"id": "z", "pricing": {"prompt": "0.5"}})


def test_caps_from_params():
    m = {
        "supported_parameters": ["tools", "response_format"],
        "architecture": {"modality": "image+text->text"},
    }
    caps = _caps(m)
    assert Capability.TOOLS in caps
    assert Capability.JSON in caps
    assert Capability.VISION in caps
    assert Capability.STREAM in caps


def test_tier_heuristic():
    # Tier S: tiny 'малютка' models (<= 4b)
    assert _tier("x-mini") is Tier.S
    assert _tier("qwen3:4b-instruct") is Tier.S
    assert _tier("llama3.2:3b") is Tier.S
    assert _tier("gemma3:1b-it-qat") is Tier.S
    assert _tier("lfm2.5-thinking:1.2b") is Tier.S
    assert _tier("gemma4:e2b") is Tier.S
    assert _tier("smollm:135m") is Tier.S

    # Tier M: 7B-16B models
    assert _tier("gemma-4-31b") is Tier.M
    assert _tier("qwen2.5:7b") is Tier.M
    assert _tier("deepseek-r1:8b") is Tier.M
    assert _tier("translategemma:12b") is Tier.M

    # Tier L: heavy 20B+ models
    assert _tier("nemotron-3-ultra-550b") is Tier.L
    assert _tier("qwen3.6:27b") is Tier.L
    assert _tier("qwen3.6:35b") is Tier.L
    assert _tier("granite4.1:30b") is Tier.L
    assert _tier("gpt-oss:20b") is Tier.L


@pytest.mark.asyncio
async def test_discover_adds_free_models():
    provider = Offering(
        provider="openrouter", model_id="seed", base_url="http://x/v1", api_key_env="K"
    )
    registry = Registry([provider])
    adapter = FakeAdapter(
        [
            {
                "id": "new/free-model:free",
                "pricing": {"prompt": "0"},
                "context_length": 100000,
                "supported_parameters": ["tools", "response_format"],
                "architecture": {"modality": "text->text"},
                "top_provider": {"max_completion_tokens": 4096},
            },
            {
                "id": "paid/model",
                "pricing": {"prompt": "0.5"},
                "context_length": 100000,
            },
        ]
    )

    r = await discover_provider(adapter, registry, provider)
    assert r["status"] == "ok"
    assert r["added"] == ["openrouter/new/free-model:free"]
    assert registry.get("openrouter/new/free-model:free") is not None
    # the paid model is not added
    assert registry.get("openrouter/paid/model") is None


@pytest.mark.asyncio
async def test_discover_does_not_duplicate():
    provider = Offering(
        provider="openrouter", model_id="seed", base_url="http://x/v1", api_key_env="K"
    )
    registry = Registry([provider])
    adapter = FakeAdapter(
        [
            {
                "id": "m:free",
                "pricing": {"prompt": "0"},
                "context_length": 100000,
                "supported_parameters": [],
                "architecture": {"modality": "text->text"},
            },
        ]
    )
    await discover_provider(adapter, registry, provider)
    r = await discover_provider(adapter, registry, provider)
    assert r["added"] == []  # a repeat run duplicates nothing


@pytest.mark.asyncio
async def test_non_openrouter_reconciles_only():
    provider = Offering(
        provider="groq", model_id="known-model", base_url="http://x/v1", api_key_env="K"
    )
    registry = Registry([provider])
    adapter = FakeAdapter([{"id": "known-model"}, {"id": "other-model"}])
    r = await discover_provider(adapter, registry, provider)
    assert r["status"] == "ok"
    # other-model not added: freeness is not visible in the API
    assert registry.get("groq/other-model") is None


@pytest.mark.asyncio
async def test_discover_local_backend():
    provider = Offering(
        provider="ollama",
        model_id="seed",
        base_url="http://127.0.0.1:11434/v1",
        is_local=True,
    )
    registry = Registry([])
    adapter = FakeAdapter(
        [
            {"id": "qwen3:4b-instruct"},
            {"id": "llama3.2:3b"},
            {"id": "qwen3.6:27b"},
            {"id": "cloud/deepseek-r1:70b"},
            {
                "id": "nomic-embed-text:latest"
            },  # embedding model - should be filtered out
        ]
    )
    r = await discover_provider(adapter, registry, provider)
    assert r["status"] == "ok"
    assert "ollama/qwen3:4b-instruct" in r["added"]
    assert "ollama/llama3.2:3b" in r["added"]
    assert "ollama/qwen3.6:27b" in r["added"]
    assert "ollama/cloud/deepseek-r1:70b" in r["added"]
    assert "ollama/nomic-embed-text:latest" not in r["added"]

    # Verify tiers and local vs cloud distribution
    qwen4b = registry.get("ollama/qwen3:4b-instruct")
    assert qwen4b is not None
    assert qwen4b.tier == Tier.S
    assert qwen4b.is_local is True
    assert qwen4b.trains_on_data is False

    llama3b = registry.get("ollama/llama3.2:3b")
    assert llama3b.tier == Tier.S
    assert llama3b.is_local is True

    qwen27b = registry.get("ollama/qwen3.6:27b")
    assert qwen27b.tier == Tier.L
    assert qwen27b.is_local is True

    # Cloud-prefixed Ollama model must be marked as is_local = False
    cloud_model = registry.get("ollama/cloud/deepseek-r1:70b")
    assert cloud_model is not None
    assert cloud_model.is_local is False
    assert cloud_model.tier == Tier.L


@pytest.mark.asyncio
async def test_local_runtime_is_removed_from_routing_when_health_check_fails():
    local = Offering(
        provider="mlx",
        model_id="qwen",
        base_url="http://127.0.0.1:8081/v1",
        is_local=True,
        runtime_available=True,
    )
    registry = Registry([local])

    result = await discover_provider(FailingLocalAdapter(), registry, local)

    assert result["status"] == "error"
    assert local.runtime_available is False
    assert local.usable is False
    assert registry.usable() == []


@pytest.mark.asyncio
async def test_local_runtime_requires_the_configured_model_and_recovers():
    local = Offering(
        provider="mlx",
        model_id="qwen",
        base_url="http://127.0.0.1:8081/v1",
        is_local=True,
        runtime_available=False,
    )
    registry = Registry([local])

    missing = await discover_provider(FakeAdapter([{"id": "other"}]), registry, local)
    assert missing["status"] == "ok"
    assert local.usable is False

    found = await discover_provider(FakeAdapter([{"id": "qwen"}]), registry, local)
    assert found["status"] == "ok"
    assert local.runtime_available is True
    assert local.usable is True


@pytest.mark.asyncio
async def test_discover_ollama_cloud_provider():
    provider = Offering(
        provider="ollama",
        model_id="seed",
        base_url="https://ollama.com/v1",
        is_local=False,
    )
    registry = Registry([])
    adapter = FakeAdapter(
        [
            {"id": "qwen3.5:397b"},
            {"id": "cloud/gpt-oss:120b"},
        ]
    )
    r = await discover_provider(adapter, registry, provider)
    assert r["status"] == "ok"
    assert "ollama/cloud/qwen3.5:397b" in r["added"]
    assert "ollama/cloud/gpt-oss:120b" in r["added"]

    m1 = registry.get("ollama/cloud/qwen3.5:397b")
    assert m1 is not None
    assert m1.is_local is False

    m2 = registry.get("ollama/cloud/gpt-oss:120b")
    assert m2 is not None
    assert m2.is_local is False
