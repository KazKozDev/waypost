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


# Feedback kinds that say the delivered answer was wrong, not followed up.
NEGATIVE_FEEDBACK = ("corrected", "regenerated", "rated_bad")


def _percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    return sorted_vals[min(len(sorted_vals) - 1, int(len(sorted_vals) * pct))]


def routing_quality(db_path: str | Path, window_s: float = 7 * 86_400) -> dict:
    """Per-request routing outcomes over the attempt log.

    The legacy `attempts` table counts deliveries; this answers the
    routing questions instead: solved on the first attempt, escalated,
    corrected by the user, attempts and latency to an answer, quota
    spent — overall and per policy arm. Shadows and explorations are
    excluded: they are measurements, not served requests.
    """
    since = time.time() - window_s
    conn = _conn(db_path)
    empty = {
        "requests": 0,
        "first_attempt_pass_rate": 0.0,
        "escalation_rate": 0.0,
        "correction_rate": 0.0,
        "mean_attempts": 0.0,
        "p50_latency_ms": 0,
        "p95_latency_ms": 0,
        "total_tokens": 0,
        "by_arm": {},
    }
    try:
        if not _has_table(conn, "attempt_log"):
            return empty
        finals = conn.execute(
            "SELECT request_id, attempt_no, outcome, latency_ms, arm, "
            "COALESCE(input_tokens,0)+COALESCE(output_tokens,0) "
            "FROM attempt_log WHERE ts > ? AND is_final = 1 AND is_exploration = 0",
            (since,),
        ).fetchall()
        if not finals:
            return empty
        per_req = {
            r[0]: (r[1], r[2])
            for r in conn.execute(
                "SELECT request_id, SUM(latency_ms), "
                "SUM(COALESCE(input_tokens,0)+COALESCE(output_tokens,0)) "
                "FROM attempt_log WHERE ts > ? AND is_exploration = 0 "
                "GROUP BY request_id",
                (since,),
            ).fetchall()
        }
        corrected = {
            r[0]
            for r in conn.execute(
                "SELECT DISTINCT request_id FROM feedback WHERE ts > ? AND kind IN (?,?,?)",
                (since, *NEGATIVE_FEEDBACK),
            ).fetchall()
        } if _has_table(conn, "feedback") else set()

        def summarize(rows: list) -> dict:
            n = len(rows)
            first_pass = sum(
                1 for r in rows if r[2] == "pass" and (r[1] or 1) == 1
            )
            escalated = sum(1 for r in rows if (r[1] or 1) > 1)
            corrected_n = sum(1 for r in rows if r[0] in corrected)
            costs = sorted(per_req.get(r[0], (r[3] or 0, r[5] or 0)) for r in rows)
            return {
                "requests": n,
                "first_attempt_pass_rate": round(first_pass / n, 3),
                "escalation_rate": round(escalated / n, 3),
                "correction_rate": round(corrected_n / n, 3),
                "mean_attempts": round(sum(r[1] or 1 for r in rows) / n, 2),
                "p50_latency_ms": int(_percentile([c[0] for c in costs], 0.50)),
                "p95_latency_ms": int(_percentile([c[0] for c in costs], 0.95)),
                "total_tokens": int(sum(c[1] for c in costs)),
            }

        out = summarize(finals)
        by_arm: dict[str, list] = {}
        for r in finals:
            by_arm.setdefault(r[4] or "control", []).append(r)
        out["by_arm"] = {arm: summarize(rows) for arm, rows in sorted(by_arm.items())}
        return out
    finally:
        conn.close()


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
        "routing": routing_quality(db_path, window_s),
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

    q = r.get("routing") or {}
    if q.get("requests"):
        print(
            f"\nRouting quality ({q['requests']} requests): "
            f"first-attempt pass {q['first_attempt_pass_rate']:.0%}, "
            f"escalated {q['escalation_rate']:.0%}, "
            f"corrected {q['correction_rate']:.0%}, "
            f"attempts {q['mean_attempts']:.2f}, "
            f"p50 {q['p50_latency_ms']}ms p95 {q['p95_latency_ms']}ms, "
            f"tokens {q['total_tokens']}"
        )
        for arm, a in (q.get("by_arm") or {}).items():
            print(
                f"  [{arm}] n={a['requests']} "
                f"first-pass {a['first_attempt_pass_rate']:.0%} "
                f"escalated {a['escalation_rate']:.0%} "
                f"corrected {a['correction_rate']:.0%} "
                f"p95 {a['p95_latency_ms']}ms"
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
