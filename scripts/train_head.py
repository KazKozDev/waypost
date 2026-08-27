#!/usr/bin/env python3
"""Train the L1 classifier head.

The teacher is NVIDIA prompt-task-and-complexity-classifier (English) +
a local model (Russian). Labels accumulate in a JSONL file:

    {"text": "...", "task_class": "code", "complexity": 0.7}

    python -m scripts.train_head --data var/labels.jsonl
    python -m scripts.train_head --from-telemetry   # weak labels from logs

The head is written to var/head.json and picked up by the classifier at
server startup (ROUTER_ENABLE_L1=true). Retrain on a cron schedule.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from waypost.config import Settings, load_env
from waypost.head import TaskHead


def _encode(texts: list[str]):
    """Static embeddings potion-multilingual-128M. Without the package —
    an honest error: there is nothing to train the head on."""
    from model2vec import StaticModel

    model = StaticModel.from_pretrained("minishlab/potion-multilingual-128M")
    return model.encode(texts)


def load_labels(path: str | Path) -> tuple[list[str], list[str], list[float]]:
    texts, tasks, comps = [], [], []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        texts.append(rec["text"])
        tasks.append(rec["task_class"])
        comps.append(float(rec.get("complexity", 0.5)))
    return texts, tasks, comps


def labels_from_telemetry(_db: str) -> tuple[list[str], list[str], list[float]]:
    """Weak labels: task_class is already recorded in the attempt logs.
    Complexity is taken from the tier as a proxy (S=0.2, M=0.5, L=0.8)."""
    # Without prompt texts in the logs the labels are useless — require a file.
    raise SystemExit("the logs have no prompt texts; use --data with JSONL labels")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="JSONL with labels (text, task_class, complexity)")
    ap.add_argument("--from-telemetry", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    settings = Settings()
    load_env()
    out = Path(args.out or settings.head_path)

    if args.data:
        texts, tasks, comps = load_labels(args.data)
    elif args.from_telemetry:
        texts, tasks, comps = labels_from_telemetry(settings.db_path)
    else:
        ap.error("specify --data or --from-telemetry")

    if len(texts) < 20:
        raise SystemExit(f"too few examples ({len(texts)}): need >= 20")

    X = _encode(texts)
    head = TaskHead(dim=X.shape[1])
    head.train(X, tasks, comps)
    head.save(out)
    print(f"trained on {len(texts)} examples, saved to {out}")


if __name__ == "__main__":
    main()
