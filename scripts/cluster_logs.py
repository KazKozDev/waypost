#!/usr/bin/env python3
"""Log clustering: what your traffic actually consists of.

Eleven task classes invented in advance almost always diverge from
reality. The honest path is the reverse: take your own requests, compute
embeddings, group them and look at which groups come out. That is the
empirical task_class — and a drift detector at the same time: a shift
in the distribution means the classifier needs retraining.

    python -m scripts.cluster_logs --clusters 8
    python -m scripts.cluster_logs --days 30 --out var/clusters.json

Requires ROUTER_ENABLE_PROMPT_LOG=true — without texts there is nothing
to cluster. HDBSCAN is used if installed (it finds the cluster count
itself and honestly marks noise); otherwise k-means on numpy, with no
dependencies.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np

from waypost.config import Settings, load_env
from waypost.embeddings import EmbeddingService
from waypost.telemetry import Telemetry


def kmeans(X: np.ndarray, k: int, iters: int = 60, seed: int = 0) -> np.ndarray:
    """k-means++ initialization and ordinary Lloyd iterations.

    No claim to elegance: the goal is to see the traffic structure, not
    to win a clustering benchmark.
    """
    rng = np.random.default_rng(seed)
    centers = [X[rng.integers(len(X))]]
    for _ in range(k - 1):
        d = np.min(np.stack([np.sum((X - c) ** 2, axis=1) for c in centers]), axis=0)
        probs = d / d.sum() if d.sum() else None
        centers.append(X[rng.choice(len(X), p=probs)])
    C = np.stack(centers)

    labels = np.zeros(len(X), dtype=int)
    for _ in range(iters):
        dist = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
        new = dist.argmin(axis=1)
        if (new == labels).all():
            break
        labels = new
        for j in range(k):
            members = X[labels == j]
            if len(members):
                C[j] = members.mean(axis=0)
    return labels


def cluster(X: np.ndarray, k: int) -> tuple[np.ndarray, str]:
    try:
        import hdbscan  # noqa: PLC0415

        model = hdbscan.HDBSCAN(min_cluster_size=max(5, len(X) // 50))
        return model.fit_predict(X), "hdbscan"
    except Exception:  # noqa: BLE001
        return kmeans(X, k), "kmeans"


def keywords(texts: list[str], top: int = 6) -> list[str]:
    """The most frequent words of a cluster — crude, but enough to tell
    what it is about."""
    words = Counter()
    for t in texts:
        for w in t.lower().split():
            w = w.strip(".,!?:;()[]{}\"'`")
            if len(w) > 4:
                words[w] += 1
    return [w for w, _ in words.most_common(top)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clusters", type=int, default=8)
    ap.add_argument("--days", type=float, default=30.0)
    ap.add_argument("--limit", type=int, default=5000)
    ap.add_argument("--out", default=None, help="where to save the JSON")
    args = ap.parse_args()

    load_env()
    settings = Settings()
    rows = Telemetry(settings.db_path).prompts(
        window_s=args.days * 86_400, limit=args.limit
    )
    if len(rows) < args.clusters * 3:
        print(
            f"too little data: {len(rows)} records. Enable "
            f"ROUTER_ENABLE_PROMPT_LOG=true and let the router run."
        )
        return

    svc = EmbeddingService(model_name=settings.embed_model)
    if svc.is_fallback:
        print(
            "WARNING: embeddings on the lexical fallback — clusters "
            "will be by words, not by meaning. pip install -e '.[ml]'"
        )
    texts = [r["text"] for r in rows]
    X = np.asarray(svc.encode(texts))

    labels, backend = cluster(X, args.clusters)
    print(f"{len(rows)} requests, algorithm {backend}\n")

    report = []
    for label in sorted(set(labels.tolist())):
        idx = [i for i, lbl in enumerate(labels) if lbl == label]
        subset = [rows[i] for i in idx]
        tiers = Counter(r["tier"] for r in subset)
        declared = Counter(r["task_class"] for r in subset)
        esc = sum(r["escalated"] for r in subset) / len(subset)
        name = "noise" if label == -1 else f"cluster {label}"
        entry = {
            "cluster": int(label),
            "size": len(subset),
            "keywords": keywords([r["text"] for r in subset]),
            "tiers": dict(tiers),
            "declared_task": dict(declared),
            "escalation_rate": round(esc, 3),
            "sample": subset[0]["text"][:120],
        }
        report.append(entry)
        print(f"{name}: {len(subset)} requests, escalations {esc:.0%}")
        print(f"  words: {', '.join(entry['keywords'])}")
        print(f"  rules say: {dict(declared)}")
        print(f"  tiers: {dict(tiers)}")
        print(f"  sample: {entry['sample']}\n")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"saved to {args.out}")
    print(
        "A cluster where the rules give one class but escalations are "
        "many is\nthe first candidate for relabeling: that is where the "
        "classifier is wrong."
    )


if __name__ == "__main__":
    main()
