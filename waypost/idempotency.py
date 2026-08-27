"""Request idempotency.

The scenario this exists for: a client's network dropped after sending a
request. It does not know whether the request arrived and retries it.
Without deduplication the retry charges the quota a second time — and
here quota is a scarce resource, not money.

The key comes from the client (`idempotency_key` in the body or the
`Idempotency-Key` header). The result is stored, not the fact of the
request: a replay must get the same answer, not an "already processed"
error.

A concurrent replay (the client sent twice, no response yet) is held by
a per-key lock: the second waits for the first and gets its result.
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any


class IdempotencyStore:
    def __init__(self, db_path: str | Path, ttl_s: int = 86_400):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.ttl_s = ttl_s
        self._locks: dict[str, asyncio.Lock] = {}
        self._guard = threading.Lock()
        with self._conn() as c:
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS idempotency (
                    key TEXT PRIMARY KEY, body TEXT, created_at REAL)
            """
            )
        self.stats = {"replays": 0, "stored": 0}

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def lock(self, key: str) -> asyncio.Lock:
        with self._guard:
            lock = self._locks.get(key)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[key] = lock
            return lock

    def get(self, key: str) -> dict[str, Any] | None:
        with self._conn() as c:
            row = c.execute(
                "SELECT body, created_at FROM idempotency WHERE key=?", (key,)
            ).fetchone()
        if row is None:
            return None
        body, created = row
        if time.time() - created > self.ttl_s:
            with self._conn() as c:
                c.execute("DELETE FROM idempotency WHERE key=?", (key,))
            return None
        with self._guard:
            self.stats["replays"] += 1
        return json.loads(body)

    def put(self, key: str, body: dict[str, Any]) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO idempotency VALUES (?,?,?)",
                (key, json.dumps(body, ensure_ascii=False), time.time()),
            )
        with self._guard:
            self.stats["stored"] += 1

    def purge(self) -> int:
        cutoff = time.time() - self.ttl_s
        with self._conn() as c:
            cur = c.execute("DELETE FROM idempotency WHERE created_at < ?", (cutoff,))
            removed = cur.rowcount
        with self._guard:
            self._locks = {k: v for k, v in self._locks.items() if v.locked()}
        return max(0, removed)

    def snapshot(self) -> dict[str, int]:
        with self._guard:
            return dict(self.stats)
