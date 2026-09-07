from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from .schemas import RequestProfile


@dataclass
class AttemptLogEntry:
    """A single attempt record for attempt_log per Waypost Spec v5."""

    request_id: str
    attempt_no: int
    ts: float = field(default_factory=time.time)

    # вход
    embedding: list[float] | np.ndarray | bytes | None = None
    embedder_version: str = "minishlab/potion-multilingual-128M"
    l0_labels: dict[str, Any] = field(default_factory=dict)
    l1_prediction: dict[str, Any] | None = None
    input_tokens: int = 0
    modality: str = "text"
    image_count: int = 0
    audio_duration_s: float = 0.0
    vision_token_budget: int | None = None

    # решение
    provider: str = ""
    backend: str = "cloud"  # cloud | ollama | mlx | lmstudio | llamacpp
    model: str = ""
    model_version: str = ""
    tier: str = "M"  # S | M | L
    thinking_mode: bool = False
    routing_source: str = "l0"  # l0 | l1 | cache | explore | forced | fallback | escalation_thinking | escalation_cloud | hedge
    is_exploration: bool = False
    quota_remaining_pct: float = 100.0
    quota_window_reset_in_s: int = 0
    quota_binding_limit: str = "requests"  # requests | tokens

    # результат
    status: str = "ok"  # ok | timeout | rate_limited | error | refused | truncated
    error_class: str = ""
    latency_ms: int = 0
    ttft_ms: int = 0
    output_tokens: int = 0
    reasoning_tokens: int | None = None
    peak_memory_mb: int | None = None
    is_final: bool = True

    # исход
    outcome: str = "pass"  # pass | fail | unknown
    outcome_source: str = "hard_check"  # hard_check | user_signal | judge | manual
    outcome_detail: dict[str, Any] = field(default_factory=dict)


def _serialize_embedding(emb: list[float] | np.ndarray | bytes | None) -> bytes | None:
    if emb is None:
        return None
    if isinstance(emb, bytes):
        return emb
    if isinstance(emb, np.ndarray):
        return emb.astype(np.float32).tobytes()
    if isinstance(emb, (list, tuple)):
        return np.asarray(emb, dtype=np.float32).tobytes()
    return None


def _deserialize_embedding(blob: bytes | None) -> list[float] | None:
    if blob is None:
        return None
    arr = np.frombuffer(blob, dtype=np.float32)
    return arr.tolist()


class Telemetry:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._conn() as c:
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS attempts (
                    ts REAL, offering TEXT, task_class TEXT, tier TEXT,
                    complexity REAL, confidence REAL, classifier_source TEXT,
                    est_tokens INTEGER, tokens INTEGER, ok INTEGER,
                    verdict TEXT, latency_ms REAL)
            """
            )
            c.execute("CREATE INDEX IF NOT EXISTS idx_ts ON attempts(ts)")
            # The prompt log is the training set for the classifier head
            # and for traffic clustering. NOT written by default: request
            # texts are the most sensitive data in the system.
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS prompts (
                    ts REAL, hash TEXT, text TEXT, task_class TEXT,
                    tier TEXT, language TEXT, offering TEXT, tokens INTEGER,
                    ok INTEGER, escalated INTEGER)
            """
            )
            c.execute("CREATE INDEX IF NOT EXISTS idx_prompts_ts ON prompts(ts)")

            # Waypost v5: attempt_log table for attempt-level outcomes
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS attempt_log (
                    request_id TEXT,
                    attempt_no INTEGER,
                    ts REAL,
                    embedding BLOB,
                    embedder_version TEXT,
                    l0_labels TEXT,
                    l1_prediction TEXT,
                    input_tokens INTEGER,
                    modality TEXT,
                    image_count INTEGER,
                    audio_duration_s REAL,
                    vision_token_budget INTEGER,
                    provider TEXT,
                    backend TEXT,
                    model TEXT,
                    model_version TEXT,
                    tier TEXT,
                    thinking_mode INTEGER,
                    routing_source TEXT,
                    is_exploration INTEGER,
                    quota_remaining_pct REAL,
                    quota_window_reset_in_s INTEGER,
                    quota_binding_limit TEXT,
                    status TEXT,
                    error_class TEXT,
                    latency_ms INTEGER,
                    ttft_ms INTEGER,
                    output_tokens INTEGER,
                    reasoning_tokens INTEGER,
                    peak_memory_mb INTEGER,
                    is_final INTEGER,
                    outcome TEXT,
                    outcome_source TEXT,
                    outcome_detail TEXT,
                    PRIMARY KEY (request_id, attempt_no)
                )
            """
            )
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_attempt_log_ts ON attempt_log(ts)"
            )
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_attempt_log_req ON attempt_log(request_id)"
            )
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_attempt_log_model ON attempt_log(model)"
            )
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_attempt_log_outcome ON attempt_log(outcome)"
            )

            self._migrate(c)

    # The schema is extended in place: dropping accumulated logs for a
    # new column is not allowed — this is a training set, not debug
    # output.
    _EXTRA_COLUMNS = (
        ("hedged", "INTEGER DEFAULT 0"),
        ("key_index", "INTEGER DEFAULT 0"),
        ("escalated", "INTEGER DEFAULT 0"),
        ("verify_reason", "TEXT"),
        ("prompt_tokens", "INTEGER"),
        ("completion_tokens", "INTEGER"),
        ("actual_cost_usd", "REAL"),
        ("usage_measured", "INTEGER DEFAULT 0"),
    )

    def _migrate(self, c: sqlite3.Connection) -> None:
        have = {r[1] for r in c.execute("PRAGMA table_info(attempts)")}
        for name, decl in self._EXTRA_COLUMNS:
            if name not in have:
                c.execute(f"ALTER TABLE attempts ADD COLUMN {name} {decl}")
        # Older non-streaming rows already contain provider-reported total
        # tokens. Preserve that useful coverage without pretending that the
        # unavailable input/output split can be reconstructed exactly.
        c.execute(
            "UPDATE attempts SET usage_measured = 1 "
            "WHERE usage_measured = 0 AND ok = 1 AND tokens > 0"
        )

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def log_attempt(
        self,
        offering: str,
        profile: RequestProfile,
        *,
        ok: bool,
        verdict: str,
        latency_ms: float = 0.0,
        tokens: int = 0,
        hedged: bool = False,
        key_index: int = 0,
        escalated: bool = False,
        verify_reason: str | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        actual_cost_usd: float | None = None,
        usage_measured: bool = False,
    ) -> None:
        row = (
            time.time(),
            offering,
            profile.task_class,
            profile.tier.value,
            profile.complexity,
            profile.confidence,
            profile.classifier_source,
            profile.est_total_tokens,
            tokens,
            int(ok),
            verdict,
            latency_ms,
            int(hedged),
            key_index,
            int(escalated),
            verify_reason,
            prompt_tokens,
            completion_tokens,
            actual_cost_usd,
            int(usage_measured),
        )
        with self._lock, self._conn() as c:
            c.execute(
                "INSERT INTO attempts (ts, offering, task_class, tier, "
                "complexity, confidence, classifier_source, est_tokens, "
                "tokens, ok, verdict, latency_ms, hedged, key_index, "
                "escalated, verify_reason, prompt_tokens, completion_tokens, "
                "actual_cost_usd, usage_measured) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                row,
            )

    def log_prompt(
        self,
        text: str,
        profile: RequestProfile,
        *,
        offering: str,
        tokens: int = 0,
        ok: bool = True,
        escalated: bool = False,
        max_chars: int = 8000,
    ) -> None:
        """Written only when ROUTER_ENABLE_PROMPT_LOG=true.

        Without request texts you can neither train the classifier head,
        nor cluster traffic, nor compute offline reward. But these are
        your data, and the decision to store them is made explicitly, not
        by default.
        """
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        row = (
            time.time(),
            digest,
            text[:max_chars],
            profile.task_class,
            profile.tier.value,
            profile.language,
            offering,
            tokens,
            int(ok),
            int(escalated),
        )
        with self._lock, self._conn() as c:
            c.execute("INSERT INTO prompts VALUES (?,?,?,?,?,?,?,?,?,?)", row)

    def prompts(
        self, window_s: float = 30 * 86_400, limit: int = 100_000
    ) -> list[dict]:
        since = time.time() - window_s
        with self._conn() as c:
            rows = c.execute(
                "SELECT ts, hash, text, task_class, tier, language, offering, "
                "tokens, ok, escalated FROM prompts WHERE ts > ? "
                "ORDER BY ts DESC LIMIT ?",
                (since, limit),
            ).fetchall()
        keys = (
            "ts",
            "hash",
            "text",
            "task_class",
            "tier",
            "language",
            "offering",
            "tokens",
            "ok",
            "escalated",
        )
        return [dict(zip(keys, r)) for r in rows]

    def success_rates(self, window_s: float = 7 * 86_400) -> dict[str, float]:
        """Fed back into the registry: scoring must rely on measured
        reliability, not declared reliability."""
        since = time.time() - window_s
        with self._conn() as c:
            rows = c.execute(
                "SELECT offering, AVG(ok), COUNT(*) FROM attempts "
                "WHERE ts > ? GROUP BY offering",
                (since,),
            ).fetchall()
        return {r[0]: r[1] for r in rows if r[2] >= 5}

    def attempt_counts(self, window_s: float = 86_400) -> dict[str, tuple[int, float]]:
        """(attempts, success rate) per offering.

        success_rates() drops anything with fewer than five attempts, which
        is right for scoring but wrong for lifecycle decisions: promoting a
        shadow model needs to know it carried enough traffic to be judged
        at all.
        """
        since = time.time() - window_s
        with self._conn() as c:
            rows = c.execute(
                "SELECT offering, COUNT(*), AVG(ok) FROM attempts "
                "WHERE ts > ? GROUP BY offering",
                (since,),
            ).fetchall()
        return {r[0]: (int(r[1]), float(r[2] or 0.0)) for r in rows}

    def cascade_stats(self, window_s: float = 7 * 86_400) -> dict:
        """Share of verifier failures on the lower tier.

        The cascade pays off while this stays below ~30-40%: above that
        it is cheaper to call the strong model right away. The number
        must be measured, not assumed, which is why it lives in /v1/stats.
        """
        since = time.time() - window_s
        with self._conn() as c:
            total = c.execute(
                "SELECT COUNT(*) FROM attempts WHERE ts > ? AND ok = 1", (since,)
            ).fetchone()[0]
            escalated = c.execute(
                "SELECT COUNT(*) FROM attempts WHERE ts > ? AND escalated = 1", (since,)
            ).fetchone()[0]
            reasons = c.execute(
                "SELECT verify_reason, COUNT(*) FROM attempts "
                "WHERE ts > ? AND verify_reason IS NOT NULL "
                "GROUP BY verify_reason ORDER BY 2 DESC",
                (since,),
            ).fetchall()
        rate = escalated / total if total else 0.0
        return {
            "answers": total,
            "escalations": escalated,
            "escalation_rate": round(rate, 3),
            "profitable": rate < 0.35,
            "reasons": {r[0]: r[1] for r in reasons},
        }

    def hedge_stats(self, window_s: float = 86_400) -> dict:
        since = time.time() - window_s
        with self._conn() as c:
            total, hedged = c.execute(
                "SELECT COUNT(*), COALESCE(SUM(hedged), 0) FROM attempts "
                "WHERE ts > ?",
                (since,),
            ).fetchone()
        return {
            "attempts": total,
            "hedged": hedged,
            "rate": round(hedged / total, 3) if total else 0.0,
        }

    def summary(self, window_s: float = 86_400) -> list[dict]:
        since = time.time() - window_s
        with self._conn() as c:
            rows = c.execute(
                "SELECT offering, COUNT(*), AVG(ok), AVG(latency_ms), "
                "SUM(tokens) FROM attempts WHERE ts > ? "
                "GROUP BY offering ORDER BY 2 DESC",
                (since,),
            ).fetchall()
        return [
            {
                "offering": r[0],
                "attempts": r[1],
                "success_rate": round(r[2], 3),
                "avg_latency_ms": round(r[3] or 0, 1),
                "tokens": r[4] or 0,
            }
            for r in rows
        ]

    def latency_percentiles(
        self,
        window_s: float = 86_400,
        local_offerings: set[str] | None = None,
    ) -> dict[str, Any]:
        """Measured p50/p95, per offering and split local vs cloud.

        A percentile cannot be derived from an average: mean * 1.3 is a
        guess with a percentile's name on it. Only successful attempts
        count — a 200 ms connection refusal is not a fast answer.
        """
        since = time.time() - window_s
        with self._conn() as c:
            rows = c.execute(
                "SELECT offering, latency_ms FROM attempts "
                "WHERE ts > ? AND ok = 1 AND latency_ms > 0",
                (since,),
            ).fetchall()

        per_offering: dict[str, list[float]] = {}
        local_pool: list[float] = []
        cloud_pool: list[float] = []
        local_offerings = local_offerings or set()
        for off, lat in rows:
            per_offering.setdefault(off, []).append(float(lat))
            (local_pool if off in local_offerings else cloud_pool).append(float(lat))

        def pcts(values: list[float]) -> dict[str, int]:
            if not values:
                return {"p50": 0, "p95": 0, "n": 0}
            return {
                "p50": int(np.percentile(values, 50)),
                "p95": int(np.percentile(values, 95)),
                "n": len(values),
            }

        return {
            "per_offering": {k: pcts(v) for k, v in per_offering.items()},
            "local": pcts(local_pool),
            "cloud": pcts(cloud_pool),
        }

    def savings_stats(self, window_s: float = 30 * 86_400) -> dict:
        """Savings and token totals with explicit measurement coverage.

        Legacy rows can have an exact provider-reported total but no input/output
        split. They remain in the estimate using the historical 75/25 split and
        are reported separately from fully measured rows.
        """
        from . import pricing

        since = time.time() - window_s
        with self._conn() as c:
            rows = c.execute(
                "SELECT tier, tokens, prompt_tokens, completion_tokens, "
                "actual_cost_usd, usage_measured FROM attempts "
                "WHERE ts > ? AND ok = 1",
                (since,),
            ).fetchall()

        total_saved = 0.0
        total_tokens = 0
        measured_requests = 0
        exact_savings_requests = 0
        estimated_savings_requests = 0
        by_tier: dict[str, dict] = {}
        for r in rows:
            tier_name = r[0] or "M"
            tokens = int(r[1] or 0)
            prompt_tok, comp_tok = r[2], r[3]
            actual_cost = r[4]
            measured = bool(r[5])
            total_tokens += tokens
            measured_requests += int(measured)

            exact = prompt_tok is not None and comp_tok is not None
            if exact:
                exact_savings_requests += 1
                prompt_tok, comp_tok = int(prompt_tok), int(comp_tok)
            elif tokens > 0:
                estimated_savings_requests += 1
                prompt_tok = int(tokens * 0.75)
                comp_tok = tokens - prompt_tok
            else:
                prompt_tok = comp_tok = 0

            # Waypost routes only offerings proven free. New rows persist a
            # zero here explicitly; legacy rows predate cost accounting and
            # retain the old zero-cost estimate.
            saved = pricing.calculate_savings(
                prompt_tok, comp_tok, tier_name, actual_cost or 0.0
            )
            total_saved += saved
            bucket = by_tier.setdefault(
                tier_name, {"tokens": 0, "saved_usd": 0.0, "requests": 0}
            )
            bucket["tokens"] += tokens
            bucket["saved_usd"] += saved
            bucket["requests"] += 1

        for bucket in by_tier.values():
            bucket["saved_usd"] = round(bucket["saved_usd"], 4)

        total_attempts = len(rows)
        coverage = measured_requests / total_attempts if total_attempts else 1.0

        return {
            "total_saved_usd": round(total_saved, 2),
            "total_tokens": total_tokens,
            "total_requests": total_attempts,
            "measured_requests": measured_requests,
            "unmeasured_requests": total_attempts - measured_requests,
            "usage_coverage_pct": round(coverage * 100, 1),
            "exact_savings_requests": exact_savings_requests,
            "estimated_savings_requests": estimated_savings_requests,
            "baseline": pricing.BASELINE_NAME,
            "by_tier": by_tier,
        }

    # ------------------------------------------------------------- attempt_log (v5)
    def log_attempt_row(self, entry: AttemptLogEntry) -> None:
        """Write an attempt record to attempt_log table per Waypost Spec v5 Section B.1."""
        row = (
            entry.request_id,
            entry.attempt_no,
            entry.ts,
            _serialize_embedding(entry.embedding),
            entry.embedder_version,
            json.dumps(entry.l0_labels, ensure_ascii=False),
            json.dumps(entry.l1_prediction, ensure_ascii=False)
            if entry.l1_prediction is not None
            else None,
            entry.input_tokens,
            entry.modality,
            entry.image_count,
            entry.audio_duration_s,
            entry.vision_token_budget,
            entry.provider,
            entry.backend,
            entry.model,
            entry.model_version,
            entry.tier,
            int(entry.thinking_mode),
            entry.routing_source,
            int(entry.is_exploration),
            entry.quota_remaining_pct,
            entry.quota_window_reset_in_s,
            entry.quota_binding_limit,
            entry.status,
            entry.error_class,
            entry.latency_ms,
            entry.ttft_ms,
            entry.output_tokens,
            entry.reasoning_tokens,
            entry.peak_memory_mb,
            int(entry.is_final),
            entry.outcome,
            entry.outcome_source,
            json.dumps(entry.outcome_detail, ensure_ascii=False),
        )
        with self._lock, self._conn() as c:
            c.execute(
                """
                INSERT OR REPLACE INTO attempt_log (
                    request_id, attempt_no, ts, embedding, embedder_version,
                    l0_labels, l1_prediction, input_tokens, modality, image_count,
                    audio_duration_s, vision_token_budget, provider, backend, model,
                    model_version, tier, thinking_mode, routing_source, is_exploration,
                    quota_remaining_pct, quota_window_reset_in_s, quota_binding_limit,
                    status, error_class, latency_ms, ttft_ms, output_tokens,
                    reasoning_tokens, peak_memory_mb, is_final, outcome, outcome_source,
                    outcome_detail
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
                row,
            )

    def attempt_logs(
        self,
        *,
        request_id: str | None = None,
        model: str | None = None,
        outcome: str | None = None,
        window_s: float = 7 * 86_400,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Retrieve rows from attempt_log for dashboard traces and analysis."""
        since = time.time() - window_s
        query = "SELECT * FROM attempt_log WHERE ts > ?"
        params: list[Any] = [since]
        if request_id:
            query += " AND request_id = ?"
            params.append(request_id)
        if model:
            query += " AND model = ?"
            params.append(model)
        if outcome:
            query += " AND outcome = ?"
            params.append(outcome)
        query += " ORDER BY ts DESC LIMIT ?"
        params.append(limit)

        with self._conn() as c:
            c.row_factory = sqlite3.Row
            rows = c.execute(query, params).fetchall()

        results = []
        for r in rows:
            d = dict(r)
            d["thinking_mode"] = bool(d.get("thinking_mode"))
            d["is_exploration"] = bool(d.get("is_exploration"))
            d["is_final"] = bool(d.get("is_final"))
            if d.get("l0_labels"):
                try:
                    d["l0_labels"] = json.loads(d["l0_labels"])
                except Exception:
                    pass
            if d.get("l1_prediction"):
                try:
                    d["l1_prediction"] = json.loads(d["l1_prediction"])
                except Exception:
                    pass
            if d.get("outcome_detail"):
                try:
                    d["outcome_detail"] = json.loads(d["outcome_detail"])
                except Exception:
                    pass
            if d.get("embedding"):
                d["embedding"] = _deserialize_embedding(d["embedding"])
            results.append(d)
        return results

    def outcome_coverage(self, window_s: float = 30 * 86_400) -> dict[str, Any]:
        """Coverage and distribution of outcomes in attempt_log."""
        since = time.time() - window_s
        with self._conn() as c:
            total = c.execute(
                "SELECT COUNT(*) FROM attempt_log WHERE ts > ?", (since,)
            ).fetchone()[0]
            if not total:
                return {
                    "total": 0,
                    "coverage_rate": 0.0,
                    "by_outcome": {},
                    "by_source": {},
                    "weak_pass_rate": 0.0,
                }
            by_outcome_rows = c.execute(
                "SELECT outcome, COUNT(*) FROM attempt_log WHERE ts > ? GROUP BY outcome",
                (since,),
            ).fetchall()
            by_source_rows = c.execute(
                "SELECT outcome_source, COUNT(*) FROM attempt_log WHERE ts > ? GROUP BY outcome_source",
                (since,),
            ).fetchall()
            weak_count = c.execute(
                "SELECT COUNT(*) FROM attempt_log WHERE ts > ? AND outcome = 'pass' AND outcome_detail LIKE '%\"weak\": true%'",
                (since,),
            ).fetchone()[0]

        by_outcome = {r[0]: r[1] for r in by_outcome_rows}
        by_source = {r[0]: r[1] for r in by_source_rows}
        known_outcomes = sum(v for k, v in by_outcome.items() if k != "unknown")
        return {
            "total": total,
            "coverage_rate": round(known_outcomes / total, 3) if total else 0.0,
            "by_outcome": by_outcome,
            "by_source": by_source,
            "weak_pass_rate": round(weak_count / total, 3) if total else 0.0,
        }
