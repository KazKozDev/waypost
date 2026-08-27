#!/usr/bin/env python3
"""Show the free models of providers.

For CI: OpenRouter returns /models without a key, so the list of :free
models can be shown in a pipeline. Other providers need a key — they are
skipped if the variable is missing.

    python -m scripts.free_models
    python -m scripts.free_models --provider openrouter
"""
from __future__ import annotations

import argparse
import asyncio

import httpx

from waypost.config import Settings, load_env
from waypost.discovery import discover_provider
from waypost.providers.openai_compat import OpenAICompatAdapter
from waypost.registry import Registry


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", help="only this provider")
    args = ap.parse_args()

    settings = Settings()
    load_env()
    registry = Registry.from_manifest(settings.manifest_path)

    # One offering per provider. OpenRouter is polled even without a key
    # (for CI); the rest — only if a key is set.
    seen: set[str] = set()
    providers = []
    for o in registry.all():
        if o.is_local or o.provider in seen:
            continue
        seen.add(o.provider)
        if args.provider and o.provider != args.provider:
            continue
        if o.provider != "openrouter" and not o.api_key:
            continue
        providers.append(o)

    if not providers:
        print("no providers to poll (need keys in .env)")
        return

    async with httpx.AsyncClient() as client:
        adapter = OpenAICompatAdapter(client)
        for p in providers:
            r = await discover_provider(adapter, registry, p)
            print(f"\n== {r['provider']} ({r.get('status')}) ==")
            if r.get("detail"):
                print(f"   {r['detail']}")
                continue
            if r.get("gone"):
                print(f"   gone: {', '.join(r['gone'])}")
            if r.get("added"):
                print(f"   added by auto-discovery: {', '.join(r['added'])}")
            for d in r.get("demoted", []):
                print(f"   ! became paid: {d['key']} — {d['verdict']}")
            offerings = [
                o for o in registry.all() if o.provider == p.provider and not o.is_local
            ]
            for o in sorted(offerings, key=lambda x: x.model_id):
                if not o.free:
                    print(f"   x {o.model_id}  PAID ({o.free_source})")
                    continue
                mark = " (auto)" if o.weight < 1.0 else ""
                print(
                    f"   - {o.model_id}  [{o.tier.value}] "
                    f"ctx={o.ctx_window} free/{o.free_source}{mark}"
                )


if __name__ == "__main__":
    asyncio.run(main())
