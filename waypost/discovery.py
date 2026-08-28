"""Auto-discovery of free models (control plane).

At startup the server itself polls providers and learns what is free.
The decision about paidness is not made by this module but by
waypost/pricing.py — here we only gather evidence from GET /models.

For OpenRouter, freeness is visible in the API (pricing.prompt == "0" or
the ":free" suffix) — discovery there is fully automatable: new :free
models enter the registry without editing the manifest.

Other providers usually have no prices in /models. What is there is
checked by the same classifier: if a host did return a non-zero price,
the model is demoted to paid right in the registry, and the free_only
filter stops selecting it. At the same time, known models are checked
for liveness and disappeared ones are flagged.

Runs as a background job in the server lifespan and via the script
scripts/free_models.py (for CI — OpenRouter works there without a key).
"""
from __future__ import annotations

import logging
import os
from typing import Any

from . import pricing
from .providers.openai_compat import OpenAICompatAdapter, ProviderError
from .registry import Offering, Registry, is_cloud_ollama_model
from .schemas import Capability, Tier

log = logging.getLogger("waypost.discovery")

LOCAL_DEFAULTS = [
    (
        "ollama",
        lambda: os.environ.get("OLLAMA_BASE_URL")
        or os.environ.get("LOCAL_OLLAMA_URL")
        or "http://127.0.0.1:11434/v1",
    ),
    (
        "lmstudio",
        lambda: os.environ.get("LMSTUDIO_BASE_URL")
        or os.environ.get("LOCAL_LMSTUDIO_URL")
        or "http://127.0.0.1:1234/v1",
    ),
    (
        "llamacpp",
        lambda: os.environ.get("LLAMACPP_BASE_URL")
        or os.environ.get("LOCAL_LLAMACPP_URL")
        or "http://127.0.0.1:8082/v1",
    ),
    (
        "mlx",
        lambda: os.environ.get("MLX_BASE_URL")
        or os.environ.get("LOCAL_MLX_URL")
        or "http://127.0.0.1:8081/v1",
    ),
]


def local_ollama_allowed() -> bool:
    """Local Ollama stays out of the routing pool by default.

    Ollama keeps one model resident at a time: a plan that mixes several
    of its models pays a full unload/load cycle on every switch, which
    costs more than the cloud hop it was meant to save. MLX serves the
    local tier — it is pinned to a single manifest model.

    WAYPOST_LOCAL_OLLAMA=1 opts back in.
    """
    return os.environ.get("WAYPOST_LOCAL_OLLAMA", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def is_free_openrouter(model: dict[str, Any]) -> bool:
    """OpenRouter: free if prices are zero or the :free suffix is present."""
    return pricing.classify(model.get("id", ""), api_model=model).is_free


def _caps(model: dict[str, Any]) -> set[Capability]:
    params = model.get("supported_parameters") or []
    params = params if isinstance(params, list) else []
    caps: set[Capability] = {Capability.STREAM}
    if "tools" in params or "tool_choice" in params:
        caps.add(Capability.TOOLS)
    if "response_format" in params:
        caps.add(Capability.JSON)
    modality = (model.get("architecture") or {}).get("modality", "")
    if "image" in modality:
        caps.add(Capability.VISION)
    return caps


def _tier(model_id: str) -> Tier:
    """Assign S/M/L tier. S-tier is for tiny 'малютка' models (<=4b)."""
    low = model_id.lower()
    # 1. Tier S: tiny / lightweight models <= 4B
    if any(
        k in low
        for k in (
            ":0.5b",
            "-0.5b",
            "0.5b",
            ":1b",
            "-1b",
            "1b-",
            ":1.2b",
            ":1.5b",
            "1.5b",
            ":2b",
            "-2b",
            "2b-",
            "e2b",
            ":3b",
            "-3b",
            "3b-",
            ":4b",
            "-4b",
            "4b-",
            "e4b",
            "mini",
            "nano",
            "small",
            "tiny",
            "smollm",
            "pico",
        )
    ):
        return Tier.S
    # 2. Tier M: standard mid-size 7B-16B models
    if any(
        k in low
        for k in (
            ":7b",
            "-7b",
            "7b-",
            ":8b",
            "-8b",
            "8b-",
            ":9b",
            "-9b",
            "9b-",
            ":11b",
            "-11b",
            ":12b",
            "-12b",
            "12b-",
            ":13b",
            "-13b",
            ":14b",
            "-14b",
            ":15b",
            ":16b",
        )
    ):
        return Tier.M
    # 3. Tier L: heavy / reasoning models >= 20B
    if any(
        k in low
        for k in (
            "ultra",
            "550b",
            "397b",
            "120b",
            "70b",
            "35b",
            "34b",
            "32b",
            "30b",
            "27b",
            "26b",
            "24b",
            "20b",
            "large",
            "super",
            "r1",
            "k2",
            "k3",
            "plus",
            "pro",
            "675b",
        )
    ):
        return Tier.L
    return Tier.M


def _is_chat_model(model_id: str) -> bool:
    """Filter out pure embedding / rerank / utility models from chat offerings."""
    low = model_id.lower()
    if any(
        k in low for k in ("embed", "embedding", "bge-", "rerank", "clip", "deplot")
    ):
        return False
    return True


def _offering_from_local(
    provider_name: str,
    base_url: str,
    model_id: str,
    is_provider_cloud: bool = False,
) -> Offering:
    tier = _tier(model_id)
    quality = 0.50 if tier is Tier.S else (0.72 if tier is Tier.M else 0.82)
    ttft = 200.0 if tier is Tier.S else (500.0 if tier is Tier.M else 900.0)

    is_local = True
    if provider_name.lower() in ("ollama", "local"):
        if (
            is_cloud_ollama_model(model_id)
            or "ollama.com" in base_url.lower()
            or is_provider_cloud
        ):
            is_local = False

    return Offering(
        provider=provider_name,
        model_id=model_id,
        base_url=base_url,
        api_key_env=None if is_local else "OLLAMA_API_KEY",
        tier=tier,
        ctx_window=32768,
        max_output=4096,
        caps={Capability.STREAM, Capability.JSON, Capability.TOOLS},
        free=True,
        free_source=pricing.Source.MANIFEST.value,
        free_detail="local model" if is_local else "cloud model via ollama",
        trains_on_data=False,
        is_local=is_local,
        concurrency=1 if is_local else None,
        quality_score=quality,
        ttft_p50_ms=ttft,
        weight=1.0,
    )


def _offering_from_openrouter(
    provider: Offering, model: dict[str, Any], verdict: "pricing.Verdict"
) -> Offering:
    tp = model.get("top_provider") or {}
    return Offering(
        provider=provider.provider,
        model_id=model["id"],
        base_url=provider.base_url,
        api_key_env=provider.api_key_env,
        tier=_tier(model["id"]),
        ctx_window=model.get("context_length") or 8192,
        max_output=tp.get("max_completion_tokens") or 4096,
        caps=_caps(model),
        free=verdict.is_free,
        free_source=verdict.source.value,
        free_detail=verdict.detail,
        trains_on_data=provider.trains_on_data,
        is_local=False,
        limit_rpm=provider.limit_rpm,
        limit_rpd=provider.limit_rpd,
        limit_tpm=provider.limit_tpm,
        quality_score=0.5,  # starting point; probe will refine
        ttft_p50_ms=2000.0,
        weight=0.8,  # auto-discovered — slightly below curated
    )


def _recheck_prices(
    registry: Registry, provider: Offering, remote: list[dict[str, Any]]
) -> list[dict[str, str]]:
    """Reconcile declared paidness with the prices the provider returned."""
    by_id = {m.get("id", ""): m for m in remote}
    changed: list[dict[str, str]] = []
    for o in registry.all():
        if o.provider != provider.provider:
            continue
        m = by_id.get(o.model_id)
        if m is None:
            continue
        verdict = pricing.classify(o.model_id, api_model=m, declared=o.free)
        if verdict.is_free == o.free:
            continue
        was = "free" if o.free else "paid"
        if o.set_pricing(verdict):
            changed.append({"key": o.key, "verdict": str(verdict)})
            log.warning("pricing %s: %s → %s", o.key, was, verdict)
    return changed


async def discover_provider(
    adapter: OpenAICompatAdapter, registry: Registry, provider: Offering
) -> dict[str, Any]:
    """Polls one provider's /models. Returns a summary."""
    known = {o.model_id for o in registry.all() if o.provider == provider.provider}
    try:
        remote = await adapter.list_models(
            provider, timeout=2.0 if provider.is_local else 30.0
        )
    except ProviderError as exc:
        if provider.is_local:
            registry.update_local_runtime(provider.provider, provider.base_url, None)
        return {
            "provider": provider.provider,
            "status": "error",
            "detail": str(exc)[:200],
        }
    except Exception as exc:  # noqa: BLE001
        if provider.is_local:
            registry.update_local_runtime(provider.provider, provider.base_url, None)
        return {
            "provider": provider.provider,
            "status": "unsupported",
            "detail": type(exc).__name__,
        }

    added: list[str] = []
    demoted = _recheck_prices(registry, provider, remote)

    is_remote_cloud_ollama = provider.provider.lower() == "ollama" and (
        "ollama.com" in provider.base_url.lower() or not provider.is_local
    )

    if provider.is_local or provider.provider in (
        "local",
        "ollama",
        "lmstudio",
        "llamacpp",
        "mlx",
    ):
        local_model_ids = {
            str(m.get("id", ""))
            for m in remote
            if m.get("id") and _is_chat_model(str(m.get("id", "")))
        }
        if provider.is_local:
            registry.update_local_runtime(
                provider.provider, provider.base_url, local_model_ids
            )

        for m in remote:
            mid = m.get("id", "")
            if not mid or not _is_chat_model(mid):
                continue
            # For MLX, avoid auto-registering un-loaded HF cache models if manifest already configured MLX
            if provider.provider == "mlx" and known and mid not in known:
                continue

            if is_remote_cloud_ollama:
                if not is_cloud_ollama_model(mid):
                    mid = f"cloud/{mid}"
                o = _offering_from_local(
                    provider.provider,
                    provider.base_url,
                    mid,
                    is_provider_cloud=True,
                )
            else:
                o = _offering_from_local(provider.provider, provider.base_url, mid)

            if registry.add(o):
                added.append(o.key)
        return {
            "provider": provider.provider,
            "status": "ok",
            "known": len(known),
            "added": added,
            "demoted": demoted,
        }

    if provider.provider == "openrouter":
        for m in remote:
            verdict = pricing.classify(m.get("id", ""), api_model=m)
            if not verdict.is_free:
                continue
            o = _offering_from_openrouter(provider, m, verdict)
            if registry.add(o):
                added.append(o.key)
        return {
            "provider": provider.provider,
            "status": "ok",
            "known": len(known),
            "added": added,
            "demoted": demoted,
        }

    # Other cloud providers: liveness check + price demotion
    remote_ids = {m.get("id", "") for m in remote}
    gone = sorted(known - remote_ids)
    return {
        "provider": provider.provider,
        "status": "ok",
        "known": len(known),
        "gone": gone,
        "demoted": demoted,
    }


async def discover_local_backends(
    adapter: OpenAICompatAdapter, registry: Registry
) -> list[dict[str, Any]]:
    """Auto-detect and discover models from running local engines (Ollama, LM Studio, llama.cpp, MLX)."""
    endpoints: dict[str, tuple[str, str]] = {}
    for o in registry.all():
        if o.is_local:
            endpoints[o.provider] = (o.provider, o.base_url)

    if not endpoints:
        for name, url_getter in LOCAL_DEFAULTS:
            if name not in endpoints:
                endpoints[name] = (name, url_getter())

    # Deliberate, not accidental: see local_ollama_allowed().
    if not local_ollama_allowed():
        endpoints.pop("ollama", None)

    results = []
    for name, (prov_name, base_url) in endpoints.items():
        dummy = Offering(
            provider=prov_name, model_id="_probe", base_url=base_url, is_local=True
        )
        try:
            res = await discover_provider(adapter, registry, dummy)
            # Offline results are useful control-plane state too: omitting
            # them left the UI showing the last successful discovery forever.
            results.append(res)
        except Exception as exc:  # noqa: BLE001
            log.debug(
                "local backend %s (%s) check failed: %s", prov_name, base_url, exc
            )
    return results


async def run_discovery(
    adapter: OpenAICompatAdapter,
    registry: Registry,
) -> list[dict[str, Any]]:
    """Polls all local runtimes and cloud providers with keys. Returns a summary."""
    results = []
    # 1. Local engines auto-discovery
    try:
        local_results = await discover_local_backends(adapter, registry)
        results.extend(local_results)
    except Exception as exc:  # noqa: BLE001
        log.warning("local discovery failed: %s", exc)

    # 2. Cloud providers
    cloud_providers = [o for o in registry.usable() if not o.is_local]
    seen_prov = set()
    for p in cloud_providers:
        if p.provider in seen_prov:
            continue
        seen_prov.add(p.provider)
        try:
            results.append(await discover_provider(adapter, registry, p))
        except Exception as exc:  # noqa: BLE001
            log.warning("discovery %s failed: %s", p.provider, exc)
    return results
