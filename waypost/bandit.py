"""Bandit scoring (Thompson sampling) over the (task_class × model) pair.

The quality_score from the manifest is a starting point, not the truth.
The bandit keeps a success/failure count per pair and updates the
estimate from online signals: an ok verdict, the absence of a retry,
schema validity. Under a week of load this gives a more honest picture
than any benchmark.

The estimate is the posterior mean of Beta(alpha, beta), deterministic
(stable for scoring). sample() is a random draw — the router uses it for
selection, which gives exploration for free instead of paying for it with
duplicate calls.
State survives restarts via SQLite.

Counts decay. Without it alpha and beta only grow: after a few thousand
requests the posterior is frozen and new evidence cannot move it, so a
model that got worse last week still scores on last month's record. The
world of free tiers is not stationary — models are swapped behind the
same id, quantization changes, hosts get overloaded. A half-life of a few
days keeps the estimate about the *current* model, and the cap on
alpha+beta bounds how confident it is allowed to get.
"""
from __future__ import annotations

import logging
import math
import sqlite3
import threading
import time
from pathlib import Path

import numpy as np

log = logging.getLogger("waypost.bandit")

DECAY_PER_HOUR = 0.995  # ≈ half-life of 5.8 days
MAX_EVIDENCE = 200.0  # cap on alpha + beta

# Evidence semantics version. v1 mixed delivery, format checks and user
# dislike into one number (a completed stream was a success, an
# escalation punished the model that finally answered, a "bad" rating
# was averaged with automatic checks). v2 records per-attempt quality
# rewards with evidence weights. The schema is shared with other tables
# in the same file, so the stamp lives in its own meta table, not in
# PRAGMA user_version.
EVIDENCE_VERSION = 2
# Stale evidence keeps its direction at a quarter of the confidence:
# informative, but no longer allowed to outshout new observations.
STALE_EVIDENCE_KEEP = 0.25


class Bandit:
    def __init__(
        self,
        db_path: str | Path | None = None,
        *,
        decay_per_hour: float = DECAY_PER_HOUR,
        max_evidence: float = MAX_EVIDENCE,
    ):
        self._alpha: dict[tuple[str, str], float] = {}
        self._beta: dict[tuple[str, str], float] = {}
        self._ts: dict[tuple[str, str], float] = {}
        self.decay_per_hour = decay_per_hour
        self.max_evidence = max_evidence
        self._lock = threading.Lock()
        self._db_path = Path(db_path) if db_path else None
        self._conn_cache: sqlite3.Connection | None = None
        if self._db_path:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_db()
            self._load()
            self._migrate_evidence()

    # ---------------------------------------------------------------- db
    def _conn(self) -> sqlite3.Connection:
        """Reused, like the ledger's: update() runs once per answer."""
        conn = self._conn_cache
        if conn is None:
            conn = sqlite3.connect(self._db_path, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._conn_cache = conn
        return conn

    def close(self) -> None:
        conn, self._conn_cache = self._conn_cache, None
        if conn is not None:
            conn.close()

    def _init_db(self) -> None:
        with self._conn() as c:
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS bandit (
                    task TEXT, offering TEXT, alpha REAL, beta REAL,
                    PRIMARY KEY (task, offering))
            """
            )
            # Migration: databases written before decay existed have no ts.
            cols = {r[1] for r in c.execute("PRAGMA table_info(bandit)")}
            if "ts" not in cols:
                c.execute("ALTER TABLE bandit ADD COLUMN ts REAL")
            c.execute(
                "CREATE TABLE IF NOT EXISTS bandit_meta (key TEXT PRIMARY KEY, value TEXT)"
            )

    def _load(self) -> None:
        with self._conn() as c:
            rows = c.execute(
                "SELECT task, offering, alpha, beta, ts FROM bandit"
            ).fetchall()
        now = time.time()
        for task, offering, a, b, ts in rows:
            self._alpha[(task, offering)] = a
            self._beta[(task, offering)] = b
            self._ts[(task, offering)] = ts or now

    def _evidence_version(self) -> int:
        with self._conn() as c:
            row = c.execute(
                "SELECT value FROM bandit_meta WHERE key='evidence_version'"
            ).fetchone()
        try:
            return int(row[0]) if row else 1
        except (TypeError, ValueError):
            return 1

    def _migrate_evidence(self) -> None:
        """Slash the confidence of pre-v2 evidence, once.

        Old counts learned under different semantics (delivery counted
        as quality, escalations punished the rescuer). They are not
        deleted — the direction is still informative — but quartered,
        so fresh observations outshout them within days, not months.
        """
        if not self._db_path or self._evidence_version() >= EVIDENCE_VERSION:
            return
        now = time.time()
        with self._lock:
            for key in list(self._alpha):
                a, b = self._decayed(key, now)
                self._alpha[key] = 1.0 + (a - 1.0) * STALE_EVIDENCE_KEEP
                self._beta[key] = 1.0 + (b - 1.0) * STALE_EVIDENCE_KEEP
                self._persist(*key)
            count = len(self._alpha)
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO bandit_meta VALUES (?,?)",
                ("evidence_version", str(EVIDENCE_VERSION)),
            )
        log.info("bandit: quartered %d stale evidence pair(s) to v%d", count, EVIDENCE_VERSION)

    def _persist(self, task: str, offering: str) -> None:
        if not self._db_path:
            return
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO bandit VALUES (?,?,?,?,?)",
                (
                    task,
                    offering,
                    self._alpha[(task, offering)],
                    self._beta[(task, offering)],
                    self._ts[(task, offering)],
                ),
            )

    # ------------------------------------------------------------- decay
    def _decayed(self, key: tuple[str, str], now: float) -> tuple[float, float]:
        """alpha/beta pulled back toward the uninformative prior by age.

        Pure: reading a stale pair must not rewrite it, or a busy task
        class would decay faster than a quiet one purely by being read.
        """
        a = self._alpha.get(key, 1.0)
        b = self._beta.get(key, 1.0)
        if self.decay_per_hour >= 1.0:
            return a, b
        hours = max(0.0, (now - self._ts.get(key, now)) / 3600.0)
        if hours <= 0.0:
            return a, b
        f = math.pow(self.decay_per_hour, hours)
        # Decay the *evidence* (counts above the prior), never the prior.
        return 1.0 + (a - 1.0) * f, 1.0 + (b - 1.0) * f

    # ------------------------------------------------------------- api
    def quality(self, task_class: str, offering_key: str, prior: float = 0.5) -> float:
        """Posterior mean. Without data returns the manifest prior."""
        a, b = self._decayed((task_class, offering_key), time.time())
        if a == 1.0 and b == 1.0:
            return prior
        return a / (a + b)

    def sample(self, task_class: str, offering_key: str, prior: float = 0.5) -> float:
        """Thompson draw. Used by the router for selection: the spread of
        the draw *is* the exploration, and it costs nothing, unlike a
        duplicate call to a second provider."""
        a, b = self._decayed((task_class, offering_key), time.time())
        if a == 1.0 and b == 1.0:
            # No evidence yet — explore around the prior rather than
            # returning it flat, or an unmeasured model can never win.
            return float(np.random.beta(max(0.1, prior * 2), max(0.1, (1 - prior) * 2)))
        return float(np.random.beta(a, b))

    def evidence(self, task_class: str, offering_key: str) -> float:
        """How much the estimate is worth: alpha + beta above the prior."""
        a, b = self._decayed((task_class, offering_key), time.time())
        return (a - 1.0) + (b - 1.0)

    def update(
        self, task_class: str, offering_key: str, reward: float, *, weight: float = 1.0
    ) -> None:
        """reward in [0,1]: 1 = success, 0 = failure.

        Only call this for evidence about ANSWER QUALITY — a verifier
        verdict. A 429 or a connection reset says nothing about how good
        the model is; routing those here taught the bandit to avoid
        whichever provider happened to be rate limited.
        """
        reward = max(0.0, min(1.0, reward))
        weight = max(0.0, min(1.0, weight))
        if weight == 0.0:
            return
        with self._lock:
            key = (task_class, offering_key)
            now = time.time()
            a, b = self._decayed(key, now)
            a += reward * weight
            b += (1.0 - reward) * weight
            total = a + b
            if total > self.max_evidence:
                # Bound the confidence: with unbounded counts the
                # posterior collapses to a point and stops exploring.
                scale = self.max_evidence / total
                a, b = 1.0 + (a - 1.0) * scale, 1.0 + (b - 1.0) * scale
            self._alpha[key], self._beta[key], self._ts[key] = a, b, now
            self._persist(*key)

    def snapshot(self) -> dict[str, dict[str, float]]:
        with self._lock:
            now = time.time()
            out: dict[str, dict[str, float]] = {}
            for key in list(self._alpha):
                a, b = self._decayed(key, now)
                task, offering = key
                out.setdefault(offering, {})[task] = round(a / (a + b), 3)
            return out
