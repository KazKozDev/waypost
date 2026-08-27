#!/usr/bin/env python3
"""Discovery of free models (source B).

Walks GET /models of each provider and reconciles with the manifest: what
appeared, what disappeared. On OpenRouter, free models are marked with
the ':free' suffix in the id — discovery there is genuinely automatable;
on the rest it is a check against the static manifest.

    python -m scripts.discover
    python -m scripts.discover --free-only

The script adds NOTHING to the manifest itself. The decision to enable a
model is human: it has a license and a data policy that /models does not
carry.
"""
from __future__ import annotations

import argparse
import asyncio

import httpx

from waypost import pricing
from waypost.config import Settings, load_env
from waypost.providers.openai_compat import OpenAICompatAdapter, ProviderError
from waypost.registry import Registry


async def discover(
    adapter: OpenAICompatAdapter, registry: Registry, provider: str, free_only: bool
) -> None:
    offerings = [o for o in registry.usable() if o.provider == provider]
    if not offerings:
        return
    probe = offerings[0]
    known = {o.model_id for o in offerings}

    try:
        remote = await adapter.list_models(probe)
    except ProviderError as exc:
        print(f"\n{provider}: unavailable — {exc}")
        return
    except Exception as exc:  # noqa: BLE001
        print(f"\n{provider}: /models not supported ({type(exc).__name__})")
        return

    # Paidness is decided by the common classifier, not by a substring in
    # the id: for some hosts the price comes in /models itself.
    remote_ids = {m.get("id", "") for m in remote}
    if free_only:
        remote_ids = {
            m.get("id", "")
            for m in remote
            if pricing.classify(m.get("id", ""), api_model=m).is_free
        }

    new = sorted(remote_ids - known)
    gone = sorted(known - remote_ids)

    print(
        f"\n{provider}: {len(remote_ids)} models at the provider, "
        f"{len(known)} in the manifest"
    )
    for i in new[:25]:
        print(f"  + {i}")
    if len(new) > 25:
        print(f"  … {len(new) - 25} more")
    for i in gone:
        print(f"  - {i}  (in the manifest, not at the provider)")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--free-only",
        action="store_true",
        help="only models with a freeness marker in the id",
    )
    args = ap.parse_args()

    settings = Settings()
    load_env()
    registry = Registry.from_manifest(settings.manifest_path)
    providers = sorted({o.provider for o in registry.usable() if not o.is_local})
    if not providers:
        print("no providers with keys: fill in .env")
        return

    async with httpx.AsyncClient() as client:
        adapter = OpenAICompatAdapter(client)
        for p in providers:
            await discover(adapter, registry, p, args.free_only)

    print(
        "\nNothing was added automatically: /models does not carry a "
        "license or data policy — the decision is yours."
    )


if __name__ == "__main__":
    asyncio.run(main())
