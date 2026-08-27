#!/usr/bin/env python3
"""Probing (source C) — a CLI over waypost.probe.

    python -m scripts.probe                 # all available offerings
    python -m scripts.probe --provider groq
    python -m scripts.probe --apply         # merge into the registry and show

The logic lives in waypost/probe.py: the same code runs as a control-plane
background job once a day. The script exists to run the measurement by
hand and see the result with your own eyes.
"""
from __future__ import annotations

import argparse
import asyncio

import httpx

import json

from waypost.config import Settings, load_env
from waypost.probe import apply_to_registry, probe_all, store, summarize_by_tier
from waypost.providers.openai_compat import OpenAICompatAdapter
from waypost.registry import Registry


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", help="probe only this provider")
    ap.add_argument(
        "-t",
        "--tier",
        choices=["S", "M", "L", "s", "m", "l"],
        help="probe only this tier (S, M, L)",
    )
    ap.add_argument(
        "--all", action="store_true", help="include offerings with missing API keys"
    )
    ap.add_argument("--json", action="store_true", help="output results as JSON")
    ap.add_argument("--manifest", default=None)
    ap.add_argument(
        "--apply",
        action="store_true",
        help="merge the results into the registry and show the diff",
    )
    args = ap.parse_args()

    load_env()
    settings = Settings()
    registry = Registry.from_manifest(args.manifest or settings.manifest_path)
    tier_filter = args.tier.upper() if args.tier else None

    candidates = registry.all() if args.all else registry.usable()
    targets = [
        o
        for o in candidates
        if (not args.provider or o.provider == args.provider)
        and (not tier_filter or o.tier.value == tier_filter)
    ]
    if not targets:
        print("no matching offerings: check keys in .env or tier/provider filters")
        return

    async with httpx.AsyncClient() as client:
        results = await probe_all(OpenAICompatAdapter(client), targets)

    store(settings.db_path, results)
    summary = summarize_by_tier(results)

    if args.json:
        print(
            json.dumps(
                {"summary": summary, "probes": results}, ensure_ascii=False, indent=2
            )
        )
        return

    width = max(len(str(r["key"])) for r in results)
    for r in sorted(results, key=lambda x: (str(x.get("tier", "")), str(x["status"]))):
        ttft = r.get("ttft_p50_ms")
        tier_val = r.get("tier", "?")
        mark = "~" if r.get("ttft_estimated") else " "
        print(
            f"[{tier_val}] {str(r['key']):<{width}}  {r['status']:<9} "
            f"ttft={mark}{ttft if ttft else '—':>8}  "
            f"tools={'yes' if r.get('supports_tools') else 'no':<4} "
            f"json={'yes' if r.get('supports_json') else 'no'}"
            + (f"  {r.get('error', '')}" if r["status"] != "healthy" else "")
        )

    print("\nSummary by Tier:")
    for tier, s in summary["by_tier"].items():
        if s["total"] == 0:
            continue
        avg = f", avg ttft={s['avg_ttft_ms']}ms" if s["avg_ttft_ms"] else ""
        print(
            f"  Tier {tier}: {s['healthy']}/{s['total']} healthy"
            f" (degraded={s['degraded']}, dead={s['dead']}, no_key={s['no_key']}{avg})"
        )
    print("\n~ — TTFT estimated from the full answer: the provider did not stream.")

    if args.apply:
        applied = apply_to_registry(registry, {r["key"]: r for r in results})
        print(
            f"merged into the registry: {applied} offerings "
            f"(the server does the same at startup)"
        )


if __name__ == "__main__":
    asyncio.run(main())
