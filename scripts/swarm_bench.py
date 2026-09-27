"""Run the swarm benchmark: every task in benchmarks/swarm_tasks.json, one
after another, through the running Waypost. Records status, model calls,
wall time and the result, so a change to the swarm is judged on all kinds
of work, not on one coding task.

    python scripts/swarm_bench.py LABEL [--only id1,id2] [--timeout 1200]

Writes var/bench/LABEL/report.json and var/bench/LABEL/results.md.
Quality is not scored here: each task carries a `check` line for a human
(or a later judge) to compare the result against.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import httpx

ROOT = Path(__file__).resolve().parents[1]
BASE = "http://127.0.0.1:8080"


def run_task(client: httpx.Client, task: dict, timeout: float) -> dict:
    created = client.post(f"{BASE}/v1/swarm/runs", json={"task": task["task"]}).json()
    run_id = created["id"]
    started = time.monotonic()
    status = {}
    while time.monotonic() - started < timeout:
        time.sleep(10)
        status = client.get(f"{BASE}/v1/swarm/runs/{run_id}").json()
        if status.get("status") not in ("starting", "running") or not status.get("running", True):
            break
    else:
        client.post(f"{BASE}/v1/swarm/runs/{run_id}/interrupt")
        status = client.get(f"{BASE}/v1/swarm/runs/{run_id}").json()
        status["status"] = "timeout"
    result_path = ROOT / "var" / "swarm-ui" / run_id / "result.md"
    return {"id": task["id"], "kind": task["kind"], "run": run_id, "status": status.get("status"),
            "calls": status.get("calls"), "seconds": round(time.monotonic() - started),
            "error": status.get("error"), "check": task["check"],
            "result": result_path.read_text() if result_path.exists() else status.get("draft", "")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("label")
    parser.add_argument("--only", default="")
    parser.add_argument("--timeout", type=float, default=1200)
    args = parser.parse_args()
    tasks = json.loads((ROOT / "benchmarks" / "swarm_tasks.json").read_text())
    if args.only:
        wanted = set(args.only.split(","))
        tasks = [t for t in tasks if t["id"] in wanted]
    out = ROOT / "var" / "bench" / args.label
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    with httpx.Client(timeout=30, trust_env=False) as client:
        for task in tasks:
            row = run_task(client, task, args.timeout)
            rows.append(row)
            print(f"{row['id']:<20} {row['status']:<12} calls={row['calls']:<4} {row['seconds']}s", flush=True)
            (out / "report.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    md = [f"# Swarm benchmark: {args.label}\n",
          "| task | kind | status | calls | seconds |", "|---|---|---|---|---|"]
    md += [f"| {r['id']} | {r['kind']} | {r['status']} | {r['calls']} | {r['seconds']} |" for r in rows]
    for r in rows:
        md.append(f"\n## {r['id']} ({r['status']}, {r['calls']} calls, {r['seconds']} s)\n\n"
                  f"**Check:** {r['check']}\n\n{(r['result'] or '')[:6000]}")
    (out / "results.md").write_text("\n".join(md))
    print(f"calls total {sum(r['calls'] or 0 for r in rows)}, seconds total {sum(r['seconds'] for r in rows)}")


if __name__ == "__main__":
    main()
