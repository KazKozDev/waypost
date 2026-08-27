#!/usr/bin/env python3
"""Offline quality scoring: a dense "task × model" matrix.

Online signals for the bandit are sparse and noisy: an ok verdict says the
provider answered, not that the answer is good. Once a day you can do
better — run the accumulated requests through several models and compare
the answers with a judge. The judge here is a local model (it is free and
private), ideally a reward model like ArmoRM or Skywork-Reward.

    python -m scripts.reward_score --sample 50
    python -m scripts.reward_score --judge local/qwen3-27b-4bit --apply

--apply merges the result into the bandit: (task × model) pairs get a
prior based on comparison on YOUR traffic, not on an arena.

Requires a running server and ROUTER_ENABLE_PROMPT_LOG=true.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
from collections import defaultdict

import httpx

from waypost.bandit import Bandit
from waypost.config import Settings, load_env
from waypost.telemetry import Telemetry

JUDGE_PROMPT = """You are scoring the quality of an answer to a request.

Request:
{query}

Answer:
{answer}

Score the answer with a single number from 0 to 10: how useful, accurate
and to the point it is. Return only JSON: {{"score": <number>}}"""


async def ask(
    client: httpx.AsyncClient, base: str, model: str, content: str, **kw
) -> dict:
    r = await client.post(
        f"{base}/v1/chat/completions",
        json={
            "model": model,
            "temperature": 0.0,
            "no_cache": True,
            "messages": [{"role": "user", "content": content}],
            **kw,
        },
        timeout=180.0,
    )
    r.raise_for_status()
    return r.json()


def _content(body: dict) -> str:
    try:
        return body["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        return ""


async def judge_score(
    client: httpx.AsyncClient, base: str, judge: str, query: str, answer: str
) -> float | None:
    try:
        body = await ask(
            client,
            base,
            judge,
            JUDGE_PROMPT.format(query=query[:2000], answer=answer[:2000]),
            response_format={"type": "json_object"},
        )
        parsed = json.loads(_content(body))
        return max(0.0, min(1.0, float(parsed["score"]) / 10.0))
    except Exception:  # noqa: BLE001
        return None


async def main_async(args) -> None:
    load_env()
    settings = Settings()
    base = f"http://{settings.host}:{settings.port}"
    telemetry = Telemetry(settings.db_path)

    rows = telemetry.prompts(window_s=args.days * 86_400, limit=5000)
    if not rows:
        print("no request logs: enable ROUTER_ENABLE_PROMPT_LOG=true")
        return
    random.seed(args.seed)
    sample = random.sample(rows, min(args.sample, len(rows)))

    async with httpx.AsyncClient() as client:
        models = args.models.split(",") if args.models else None
        if not models:
            listing = (await client.get(f"{base}/v1/models", timeout=10.0)).json()
            models = [m["id"] for m in listing.get("data", [])][: args.top_models]
        print(f"comparing {len(models)} models on {len(sample)} requests")

        scores: dict[tuple[str, str], list[float]] = defaultdict(list)
        for i, row in enumerate(sample, 1):
            for model in models:
                try:
                    body = await ask(client, base, model, row["text"])
                except Exception as exc:  # noqa: BLE001
                    print(f"  {model}: {type(exc).__name__}")
                    continue
                answer = _content(body)
                score = await judge_score(client, base, args.judge, row["text"], answer)
                if score is not None:
                    scores[(row["task_class"], model)].append(score)
            print(f"[{i}/{len(sample)}] {row['text'][:60]}…")

    if not scores:
        print("failed to get any score")
        return

    print("\naverage judge score (task × model):")
    bandit = Bandit(settings.db_path) if args.apply else None
    for (task, model), values in sorted(scores.items()):
        mean = sum(values) / len(values)
        print(f"  {task:<12} {model:<44} {mean:.2f}  (n={len(values)})")
        if bandit is not None:
            # The judge score is the reward: the same interface as the
            # online signal, only denser and without retry noise.
            for v in values:
                bandit.update(task, model, v)
    if bandit is not None:
        print(
            "\nmerged into the bandit: scoring now relies on comparison "
            "on your traffic, not on the manifest"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=25)
    ap.add_argument("--days", type=float, default=30.0)
    ap.add_argument(
        "--models",
        help="comma-separated list; by default — " "the first ones from /v1/models",
    )
    ap.add_argument("--top-models", type=int, default=3)
    ap.add_argument(
        "--judge",
        default="auto",
        help="judge model (a local one is best: free and private)",
    )
    ap.add_argument(
        "--apply", action="store_true", help="merge the scores into the bandit"
    )
    ap.add_argument("--seed", type=int, default=0)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
