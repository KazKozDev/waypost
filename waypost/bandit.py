"""Bandit scoring (Thompson sampling) over the (task_class × model) pair.

The quality_score from the manifest is a starting point, not the truth.
The bandit keeps a success/failure count per pair and updates the
estimate from online signals: an ok verdict, the absence of a retry,
schema validity. Under a week of load this gives a more honest picture
than any benchmark.

The estimate is the posterior mean of Beta(alpha, beta), deterministic
(stable for scoring). sample() is a random draw for exploration.
State survives restarts via SQLite.
"""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import numpy as np


class Bandit:
    def __init__(self, db_path: str | Path | None = None):
        self._alpha: dict[tuple[str, str], float] = {}
        self._beta: dict[tuple[str, str], float] = {}
        self._lock = threading.Lock()
        self._db_path = Path(db_path) if db_path else None
        if self._db_path:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_db()
            self._load()

    # ---------------------------------------------------------------- db
    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self) -> None:
        with self._conn() as c:
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS bandit (
                    task TEXT, offering TEXT, alpha REAL, beta REAL,
                    PRIMARY KEY (task, offering))
            """
            )

    def _load(self) -> None:
        with self._conn() as c:
            rows = c.execute(
                "SELECT task, offering, alpha, beta " "FROM bandit"
            ).fetchall()
        for task, offering, a, b in rows:
            self._alpha[(task, offering)] = a
            self._beta[(task, offering)] = b

    def _persist(self, task: str, offering: str) -> None:
        if not self._db_path:
            return
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO bandit VALUES (?,?,?,?)",
                (
                    task,
                    offering,
                    self._alpha[(task, offering)],
                    self._beta[(task, offering)],
                ),
            )

    # ------------------------------------------------------------- api
    def quality(self, task_class: str, offering_key: str, prior: float = 0.5) -> float:
        """Posterior mean. Without data returns the manifest prior."""
        a = self._alpha.get((task_class, offering_key), 1.0)
        b = self._beta.get((task_class, offering_key), 1.0)
        if a == 1.0 and b == 1.0:
            return prior
        return a / (a + b)

    def sample(self, task_class: str, offering_key: str, prior: float = 0.5) -> float:
        """Thompson draw — for exploration, not for scoring."""
        a = self._alpha.get((task_class, offering_key), 1.0)
        b = self._beta.get((task_class, offering_key), 1.0)
        if a == 1.0 and b == 1.0:
            return prior
        return float(np.random.beta(a, b))

    def update(self, task_class: str, offering_key: str, reward: float) -> None:
        """reward in [0,1]: 1 = success, 0 = failure."""
        reward = max(0.0, min(1.0, reward))
        with self._lock:
            key = (task_class, offering_key)
            self._alpha[key] = self._alpha.get(key, 1.0) + reward
            self._beta[key] = self._beta.get(key, 1.0) + (1.0 - reward)
            self._persist(*key)

    def snapshot(self) -> dict[str, dict[str, float]]:
        with self._lock:
            out: dict[str, dict[str, float]] = {}
            for (task, offering), a in self._alpha.items():
                b = self._beta.get((task, offering), 1.0)
                out.setdefault(offering, {})[task] = round(a / (a + b), 3)
            return out
