"""Offline before/after smoke evaluation of request classification and representation.

No provider calls, training, or changes to the user's routing database.
The fixed baseline is the pre-request-v2 implementation.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from scripts.baselines.classify_v1 import classify_l0 as classify_baseline
from scripts.baselines.classify_v1 import _flatten as flatten_baseline
from waypost.classify import classify_l0
from waypost.request_view import build_request_view
from waypost.schemas import ChatRequest

DEFAULT_CASES = Path(__file__).resolve().parents[1] / "data/evals/request_routing_v2.json"


def evaluate(path: str | Path = DEFAULT_CASES, repeats: int = 20) -> dict:
    data = json.loads(Path(path).read_text())
    rows, times = [], {"before": [], "after": []}
    for case in data["cases"]:
        req = ChatRequest(**case["request"])
        view = build_request_view(req)
        row = {"id": case["id"], "expected": case["expected"]}
        for name, classifier in (("before", classify_baseline), ("after", classify_l0)):
            profile = classifier(req)
            samples = []
            for _ in range(repeats):
                start = time.perf_counter()
                classifier(req)
                samples.append((time.perf_counter() - start) * 1000)
            times[name].append(statistics.median(samples))
            expected = case["expected"]
            text = flatten_baseline(req)[0][:4000] if name == "before" else profile.routing_text
            matches = {
                "task": profile.task_class == expected["task_class"],
                "tier": profile.tier.value in expected["tiers"],
                "language": not expected.get("language") or profile.language == expected["language"],
            }
            row[name] = {
                "task_class": profile.task_class, "task_subtype": profile.task_subtype,
                "tier": profile.tier.value, "language": profile.language,
                "matches": matches, "passed": all(matches.values()),
                "latest_head_visible": view.latest[:80] in text,
                "latest_tail_visible": view.latest[-80:] in text,
            }
        rows.append(row)
    summary = {}
    for name in ("before", "after"):
        summary[name] = {
            "passed": sum(row[name]["passed"] for row in rows),
            "total": len(rows),
            "task_correct": sum(row[name]["matches"]["task"] for row in rows),
            "tier_correct": sum(row[name]["matches"]["tier"] for row in rows),
            "language_correct": sum(row[name]["matches"]["language"] for row in rows),
            "latest_head_visible": sum(row[name]["latest_head_visible"] for row in rows),
            "latest_tail_visible": sum(row[name]["latest_tail_visible"] for row in rows),
            "classification_median_ms": round(statistics.median(times[name]), 4),
        }
    return {
        "description": data["description"],
        "limitations": "Hand-authored smoke set, not held-out production traffic. Measures rules and input retention, not answer quality, embedding semantics or provider cost.",
        "summary": summary, "cases": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = evaluate(args.cases)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))
    for case in report["cases"]:
        if not case["after"]["passed"]:
            print(f"MISS {case['id']}: expected={case['expected']} actual={case['after']}")


if __name__ == "__main__":
    main()
