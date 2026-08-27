"""Telemetry report.

Reads the DB (var/router.db) and builds a summary: per-provider usage,
success rate, latency, tokens, cache, bandit, quotas. This is what
decisions need — not raw logs.

    waypost report                 # human-readable
    waypost report --json          # machine
    waypost report --csv           # per-provider table
"""
from __future__ import annotations

import csv
import json
import sqlite3
import sys
import time
from pathlib import Path


def _conn(db_path: str | Path) -> sqlite3.Connection:
    return sqlite3.connect(db_path)


def _has_table(conn: sqlite3.Connection, name: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone()
        is not None
    )


def generate(db_path: str | Path, window_s: float = 7 * 86_400) -> dict:
    since = time.time() - window_s
    conn = _conn(db_path)

    def q(sql, *args):
        return conn.execute(sql, args).fetchall()

    total = q(
        "SELECT COUNT(*), AVG(ok), SUM(tokens), AVG(latency_ms) "
        "FROM attempts WHERE ts > ?",
        since,
    )[0]

    by_offering = [
        {
            "offering": r[0],
            "attempts": r[1],
            "success_rate": round(r[2] or 0, 3),
            "avg_latency_ms": round(r[3] or 0, 1),
            "tokens": r[4] or 0,
        }
        for r in q(
            "SELECT offering, COUNT(*), AVG(ok), AVG(latency_ms), "
            "SUM(tokens) FROM attempts WHERE ts > ? "
            "GROUP BY offering ORDER BY 2 DESC",
            since,
        )
    ]

    by_task = [
        {"task": r[0], "attempts": r[1], "success_rate": round(r[2] or 0, 3)}
        for r in q(
            "SELECT task_class, COUNT(*), AVG(ok) FROM attempts "
            "WHERE ts > ? GROUP BY task_class ORDER BY 2 DESC",
            since,
        )
    ]

    verdicts = {
        r[0]: r[1]
        for r in q(
            "SELECT verdict, COUNT(*) FROM attempts WHERE ts > ? " "GROUP BY verdict",
            since,
        )
    }

    cache_hits = 0
    cache_entries = 0
    if _has_table(conn, "cache"):
        cache = q("SELECT hits FROM cache")
        cache_entries = len(cache)
        cache_hits = sum(r[0] for r in cache)

    bandit = {}
    if _has_table(conn, "bandit"):
        bandit = {
            r[0]: {r[1]: round(r[2] / (r[2] + r[3]), 3)}
            for r in q("SELECT offering, task, alpha, beta FROM bandit")
        }

    quota = {}
    if _has_table(conn, "quota"):
        quota = {
            r[0]: {
                r[1]: r[2]
                for r in q("SELECT key, bucket, used FROM quota WHERE key=?", r[0])
            }
            for r in q("SELECT DISTINCT key FROM quota")
        }

    from . import pricing

    tier_rows = q(
        "SELECT tier, SUM(tokens), COUNT(*) FROM attempts WHERE ts > ? AND ok = 1 GROUP BY tier",
        since,
    )
    total_saved = 0.0
    for tr in tier_rows:
        t_name = tr[0] or "M"
        toks = tr[1] or 0
        p_tok = int(toks * 0.75)
        c_tok = toks - p_tok
        total_saved += pricing.calculate_savings(p_tok, c_tok, t_name)

    conn.close()
    return {
        "window_s": window_s,
        "overview": {
            "attempts": total[0] or 0,
            "success_rate": round(total[1] or 0, 3),
            "tokens": total[2] or 0,
            "avg_latency_ms": round(total[3] or 0, 1),
            "savings_usd": round(total_saved, 2),
            "cache_entries": cache_entries,
            "cache_hits": cache_hits,
            "verdicts": verdicts,
        },
        "by_offering": by_offering,
        "by_task": by_task,
        "bandit": bandit,
        "quota": quota,
    }


def print_report(r: dict) -> None:
    o = r["overview"]
    print(f"Report for {r['window_s'] / 86_400:.0f} days")
    print(
        f"  requests: {o['attempts']}  success: {o['success_rate']:.0%}  "
        f"tokens: {o['tokens']}  saved: ${o.get('savings_usd', 0.0):.2f}  "
        f"avg latency: {o['avg_latency_ms']}ms"
    )
    print(
        f"  cache: {o['cache_entries']} entries, {o['cache_hits']} hits  "
        f"verdicts: {o['verdicts']}"
    )

    print("\nBy provider:")
    print(f"  {'offering':<45} {'n':>4} {'ok':>6} {'lat':>7} {'tokens':>8}")
    for row in r["by_offering"]:
        print(
            f"  {row['offering']:<45} {row['attempts']:>4} "
            f"{row['success_rate']:>6.0%} {row['avg_latency_ms']:>7.1f} "
            f"{row['tokens']:>8}"
        )

    if r["by_task"]:
        print("\nBy task class:")
        for row in r["by_task"]:
            print(
                f"  {row['task']:<20} n={row['attempts']:>4} "
                f"ok={row['success_rate']:.0%}"
            )

    if r["bandit"]:
        print("\nBandit (quality per task×model pair):")
        for offering, tasks in r["bandit"].items():
            print(f"  {offering:<45} " + " ".join(f"{t}={v}" for t, v in tasks.items()))


def main() -> None:
    import argparse
    from .config import Settings

    ap = argparse.ArgumentParser(prog="waypost report")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--csv", action="store_true")
    ap.add_argument("--days", type=float, default=7.0)
    args = ap.parse_args()

    r = generate(Settings().db_path, window_s=args.days * 86_400)

    if args.json:
        print(json.dumps(r, ensure_ascii=False, indent=2))
    elif args.csv:
        w = csv.writer(sys.stdout)
        w.writerow(["offering", "attempts", "success_rate", "avg_latency_ms", "tokens"])
        for row in r["by_offering"]:
            w.writerow(
                [
                    row["offering"],
                    row["attempts"],
                    row["success_rate"],
                    row["avg_latency_ms"],
                    row["tokens"],
                ]
            )
    else:
        print_report(r)


if __name__ == "__main__":
    main()
