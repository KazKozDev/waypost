"""Quota accounting — the central object of the system.

Free tiers are limited not by money but by RPM/TPM/RPD. A router that
does not know the remainder is useless: it will confidently pick the
"best" model and get a 429.

A reserve → commit scheme: before the call we reserve an estimate, after
the answer we correct it by the actual usage (the preliminary estimate
always lies). State survives a restart — otherwise daily limits are
lost.

Buckets are keyed by KEY, not by offering: two free accounts of one
provider are two independent quotas, and the second step of the
degradation ladder ("another key of the same provider") does not work
without such accounting. The internal id is "provider/model#index".
"""
from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from .registry import Offering


@dataclass
class Bucket:
    limit: int
    window_s: int
    used: int = 0
    window_start: float = 0.0
    blocked_until: float = 0.0  # set from the provider's Retry-After

    def _roll(self, now: float) -> None:
        if now - self.window_start >= self.window_s:
            self.window_start = now
            self.used = 0

    def remaining(self, now: float | None = None) -> int:
        now = now or time.time()
        self._roll(now)
        return max(0, self.limit - self.used)

    def available(self, cost: int, now: float | None = None) -> bool:
        now = now or time.time()
        if now < self.blocked_until:
            return False
        return self.remaining(now) >= cost

    def take(self, cost: int, now: float | None = None) -> None:
        now = now or time.time()
        self._roll(now)
        self.used += cost


class Ledger:
    """Three buckets per offering: requests/min, requests/day, tokens/min."""

    def __init__(self, db_path: str | Path | None = None):
        self._buckets: dict[str, dict[str, Bucket]] = {}
        self._key_counts: dict[str, int] = {}
        self._lock = threading.Lock()
        self._db_path = Path(db_path) if db_path else None
        if self._db_path:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_db()

    # ---------------------------------------------------------------- db
    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self) -> None:
        with self._conn() as c:
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS quota (
                    key TEXT, bucket TEXT, used INTEGER, window_start REAL,
                    blocked_until REAL, PRIMARY KEY (key, bucket))
            """
            )

    def _persist(self, key: str) -> None:
        if not self._db_path:
            return
        rows = [
            (key, name, b.used, b.window_start, b.blocked_until)
            for name, b in self._buckets.get(key, {}).items()
        ]
        with self._conn() as c:
            c.executemany("INSERT OR REPLACE INTO quota VALUES (?,?,?,?,?)", rows)

    def _restore(self, key: str) -> dict[str, tuple]:
        if not self._db_path:
            return {}
        with self._conn() as c:
            rows = c.execute(
                "SELECT bucket, used, window_start, blocked_until "
                "FROM quota WHERE key=?",
                (key,),
            ).fetchall()
        return {r[0]: r[1:] for r in rows}

    # ------------------------------------------------------------ buckets
    @staticmethod
    def bucket_key(o: Offering, key_index: int = 0) -> str:
        return f"{o.key}#{key_index}"

    def register(self, o: Offering) -> None:
        """Local models get no buckets — they have no limits. The rest get
        a set of buckets per key."""
        if o.is_local:
            return
        count = o.key_count
        with self._lock:
            self._key_counts[o.key] = count
            spec = {
                "rpm": (o.limit_rpm, 60),
                "rpd": (o.limit_rpd, 86_400),
                "tpm": (o.limit_tpm, 60),
            }
            now = time.time()
            for idx in range(count):
                bk = self.bucket_key(o, idx)
                if bk in self._buckets:
                    continue
                saved = self._restore(bk)
                buckets: dict[str, Bucket] = {}
                for name, (limit, window) in spec.items():
                    if limit is None:
                        continue
                    b = Bucket(limit=limit, window_s=window, window_start=now)
                    if name in saved:
                        b.used, b.window_start, b.blocked_until = saved[name]
                        b._roll(now)
                    buckets[name] = b
                self._buckets[bk] = buckets

    def _affordable(self, bk: str, est_tokens: int, now: float) -> bool:
        for name, b in self._buckets.get(bk, {}).items():
            cost = est_tokens if name == "tpm" else 1
            if not b.available(cost, now):
                return False
        return True

    def can_afford(
        self, o: Offering, est_tokens: int, key_index: int | None = None
    ) -> bool:
        """key_index=None — does at least one key have enough."""
        if o.is_local:
            return True
        with self._lock:
            now = time.time()
            if key_index is not None:
                return self._affordable(self.bucket_key(o, key_index), est_tokens, now)
            count = self._key_counts.get(o.key, 1)
            return any(
                self._affordable(self.bucket_key(o, i), est_tokens, now)
                for i in range(count)
            )

    def pick_key(self, o: Offering, est_tokens: int) -> int | None:
        """The index of the first key with enough quota. None — all
        exhausted, the candidate must be skipped entirely."""
        if o.is_local:
            return 0
        with self._lock:
            now = time.time()
            count = self._key_counts.get(o.key, 1)
            for i in range(count):
                if self._affordable(self.bucket_key(o, i), est_tokens, now):
                    return i
        return None

    def reserve(self, o: Offering, est_tokens: int, key_index: int = 0) -> None:
        if o.is_local:
            return
        bk = self.bucket_key(o, key_index)
        with self._lock:
            now = time.time()
            for name, b in self._buckets.get(bk, {}).items():
                b.take(est_tokens if name == "tpm" else 1, now)
            self._persist(bk)

    def commit(
        self, o: Offering, est_tokens: int, actual_tokens: int, key_index: int = 0
    ) -> None:
        """Correction after the answer: write off the difference between
        the estimate and the fact."""
        if o.is_local:
            return
        bk = self.bucket_key(o, key_index)
        with self._lock:
            if (b := self._buckets.get(bk, {}).get("tpm")) is not None:
                b.used = max(0, b.used - est_tokens + actual_tokens)
            self._persist(bk)

    def penalize(
        self, o: Offering, retry_after_s: float = 60.0, key_index: int = 0
    ) -> None:
        """The provider returned 429 — trust it more than our own
        counters.

        Only the key that got the refusal is blocked: a neighboring
        account has its own quota, and silencing it too means losing a
        ladder step for nothing.
        """
        if o.is_local:
            return
        bk = self.bucket_key(o, key_index)
        with self._lock:
            until = time.time() + retry_after_s
            for b in self._buckets.get(bk, {}).values():
                b.blocked_until = max(b.blocked_until, until)
            self._persist(bk)

    def burn_ratio(self, o: Offering, est_tokens: int) -> float:
        """The share of the daily quota the request will eat. Used in
        scoring: a background task must not burn the quota needed by
        interactive traffic."""
        if o.is_local:
            return 0.0
        b = self._buckets.get(self.bucket_key(o, 0), {}).get("rpd")
        if b is None or b.limit <= 0:
            return 0.0
        return min(1.0, 1.0 / b.limit)

    def snapshot(self) -> dict[str, dict[str, int]]:
        """Bucket remainders. The #0 suffix for single-key offerings is
        hidden: it carries no information and clutters the output."""
        with self._lock:
            now = time.time()
            out: dict[str, dict[str, int]] = {}
            for bk, buckets in self._buckets.items():
                offering_key, _, idx = bk.rpartition("#")
                label = bk
                if self._key_counts.get(offering_key, 1) <= 1:
                    label = offering_key
                out[label] = {name: b.remaining(now) for name, b in buckets.items()}
            return out
