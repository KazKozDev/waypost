"""Batch API: offline queue.

Three different things share the word "batching", and confusing them is
expensive:

  1. Provider Batch API — asynchronous, a window up to 24 h, ~50% off.
     This is about offline: reindexing, mass classification, running a
     dataset. Unusable for interactive traffic at all.
  2. Embedding microbatching — a 20-50 ms window (see embeddings.py).
  3. Continuous batching — a runtime parameter, not a router concern.

Here it is the first. A job is accepted wholesale, split into items and
run in the background with latency_class="batch": speed does not matter,
what matters is not burning the quota needed by interactive traffic. An
item that hit a quota wall is not lost — it goes back to the queue until
the end of the window.

State lives in SQLite: a job of ten thousand requests survives a server
restart, otherwise it has no point.
"""
from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

log = logging.getLogger("waypost.batch")

PENDING, RUNNING, DONE, FAILED, CANCELLED, EXPIRED = (
    "pending",
    "running",
    "completed",
    "failed",
    "cancelled",
    "expired",
)


@dataclass
class BatchItem:
    batch_id: str
    custom_id: str
    body: dict[str, Any]
    status: str = PENDING
    response: dict[str, Any] | None = None
    error: str | None = None


class BatchQueue:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._conn() as c:
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS batches (
                    id TEXT PRIMARY KEY, created_at REAL, status TEXT,
                    window_h REAL, expires_at REAL, completed_at REAL,
                    total INTEGER)
            """
            )
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS batch_items (
                    batch_id TEXT, seq INTEGER, custom_id TEXT,
                    request TEXT, response TEXT, status TEXT, error TEXT,
                    ts REAL, PRIMARY KEY (batch_id, seq))
            """
            )
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_items_status " "ON batch_items(status)"
            )

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    # ------------------------------------------------------------ create
    def create(
        self, requests: list[dict[str, Any]], window_h: float = 24.0
    ) -> dict[str, Any]:
        if not requests:
            raise ValueError("empty batch")
        batch_id = f"batch_{uuid.uuid4().hex[:24]}"
        now = time.time()
        with self._lock, self._conn() as c:
            c.execute(
                "INSERT INTO batches VALUES (?,?,?,?,?,?,?)",
                (
                    batch_id,
                    now,
                    PENDING,
                    window_h,
                    now + window_h * 3600,
                    None,
                    len(requests),
                ),
            )
            c.executemany(
                "INSERT INTO batch_items VALUES (?,?,?,?,?,?,?,?)",
                [
                    (
                        batch_id,
                        i,
                        r.get("custom_id") or f"item-{i}",
                        json.dumps(r.get("body") or r, ensure_ascii=False),
                        None,
                        PENDING,
                        None,
                        now,
                    )
                    for i, r in enumerate(requests)
                ],
            )
        log.info(
            "BATCH %s accepted: %d items, window %.0fh",
            batch_id,
            len(requests),
            window_h,
        )
        return self.get(batch_id)

    @staticmethod
    def parse_jsonl(text: str) -> list[dict[str, Any]]:
        """OpenAI format: one JSON per line, fields custom_id and body."""
        out = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                out.append(json.loads(line))
        return out

    # -------------------------------------------------------------- read
    def get(self, batch_id: str) -> dict[str, Any] | None:
        with self._conn() as c:
            row = c.execute(
                "SELECT id, created_at, status, window_h, expires_at, "
                "completed_at, total FROM batches WHERE id=?",
                (batch_id,),
            ).fetchone()
            if row is None:
                return None
            counts = dict(
                c.execute(
                    "SELECT status, COUNT(*) FROM batch_items WHERE batch_id=? "
                    "GROUP BY status",
                    (batch_id,),
                ).fetchall()
            )
        return {
            "id": row[0],
            "object": "batch",
            "created_at": int(row[1]),
            "status": row[2],
            "window_h": row[3],
            "expires_at": int(row[4]),
            "completed_at": int(row[5]) if row[5] else None,
            "request_counts": {
                "total": row[6],
                "completed": counts.get(DONE, 0),
                "failed": counts.get(FAILED, 0),
                "pending": counts.get(PENDING, 0),
                "running": counts.get(RUNNING, 0),
            },
        }

    def list(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._conn() as c:
            ids = [
                r[0]
                for r in c.execute(
                    "SELECT id FROM batches ORDER BY created_at DESC LIMIT ?", (limit,)
                ).fetchall()
            ]
        return [b for i in ids if (b := self.get(i))]

    def output(self, batch_id: str) -> list[dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT custom_id, status, response, error FROM batch_items "
                "WHERE batch_id=? ORDER BY seq",
                (batch_id,),
            ).fetchall()
        return [
            {
                "custom_id": r[0],
                "status": r[1],
                "response": json.loads(r[2]) if r[2] else None,
                "error": r[3],
            }
            for r in rows
        ]

    # ------------------------------------------------------------ mutate
    def cancel(self, batch_id: str) -> dict[str, Any] | None:
        with self._lock, self._conn() as c:
            c.execute(
                "UPDATE batch_items SET status=? WHERE batch_id=? "
                "AND status IN (?,?)",
                (CANCELLED, batch_id, PENDING, RUNNING),
            )
            c.execute(
                "UPDATE batches SET status=?, completed_at=? WHERE id=?",
                (CANCELLED, time.time(), batch_id),
            )
        return self.get(batch_id)

    def claim(self, limit: int = 4) -> list[BatchItem]:
        """Take items to work on. Items of expired batches are marked
        expired: the window is over, retrying is pointless."""
        now = time.time()
        with self._lock, self._conn() as c:
            c.execute(
                "UPDATE batch_items SET status=? WHERE status=? AND batch_id "
                "IN (SELECT id FROM batches WHERE expires_at < ?)",
                (EXPIRED, PENDING, now),
            )
            c.execute(
                "UPDATE batches SET status=?, completed_at=? WHERE "
                "expires_at < ? AND status IN (?,?)",
                (EXPIRED, now, now, PENDING, RUNNING),
            )
            rows = c.execute(
                "SELECT batch_id, seq, custom_id, request FROM batch_items "
                "WHERE status=? ORDER BY ts LIMIT ?",
                (PENDING, limit),
            ).fetchall()
            for r in rows:
                c.execute(
                    "UPDATE batch_items SET status=? WHERE batch_id=? " "AND seq=?",
                    (RUNNING, r[0], r[1]),
                )
                c.execute(
                    "UPDATE batches SET status=? WHERE id=? AND " "status=?",
                    (RUNNING, r[0], PENDING),
                )
        return [
            BatchItem(batch_id=r[0], custom_id=r[2], body=json.loads(r[3]))
            for r in rows
        ]

    def _seq_of(self, batch_id: str, custom_id: str) -> int | None:
        with self._conn() as c:
            row = c.execute(
                "SELECT seq FROM batch_items WHERE batch_id=? " "AND custom_id=?",
                (batch_id, custom_id),
            ).fetchone()
        return row[0] if row else None

    def finish(
        self,
        item: BatchItem,
        response: dict[str, Any] | None,
        error: str | None = None,
        *,
        requeue: bool = False,
    ) -> None:
        """requeue=True — the item did not run due to quota, put it back
        in the queue. This is not an error: there is still time until the
        end of the window."""
        status = PENDING if requeue else (DONE if response else FAILED)
        seq = self._seq_of(item.batch_id, item.custom_id)
        with self._lock, self._conn() as c:
            c.execute(
                "UPDATE batch_items SET status=?, response=?, error=?, ts=? "
                "WHERE batch_id=? AND seq=?",
                (
                    status,
                    json.dumps(response, ensure_ascii=False) if response else None,
                    error,
                    time.time(),
                    item.batch_id,
                    seq,
                ),
            )
            left = c.execute(
                "SELECT COUNT(*) FROM batch_items WHERE batch_id=? AND "
                "status IN (?,?)",
                (item.batch_id, PENDING, RUNNING),
            ).fetchone()[0]
            if left == 0:
                c.execute(
                    "UPDATE batches SET status=?, completed_at=? WHERE id=?",
                    (DONE, time.time(), item.batch_id),
                )

    def stats(self) -> dict[str, Any]:
        with self._conn() as c:
            batches = dict(
                c.execute(
                    "SELECT status, COUNT(*) FROM batches GROUP BY status"
                ).fetchall()
            )
            items = dict(
                c.execute(
                    "SELECT status, COUNT(*) FROM batch_items GROUP BY status"
                ).fetchall()
            )
        return {"batches": batches, "items": items}


class BatchWorker:
    """Background executor. Works slowly and deliberately: a pause
    between batches, bounded parallelism. The background must not bother
    the interactive path — neither on latency nor on quota."""

    def __init__(
        self,
        queue: BatchQueue,
        runner: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]],
        *,
        concurrency: int = 2,
        idle_s: float = 5.0,
        requeue_delay_s: float = 60.0,
    ):
        self.queue = queue
        self.runner = runner
        self.concurrency = concurrency
        self.idle_s = idle_s
        self.requeue_delay_s = requeue_delay_s
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._stop = asyncio.Event()
            self._task = asyncio.create_task(self.run())

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._task = None

    async def run(self) -> None:
        while not self._stop.is_set():
            items = await asyncio.to_thread(self.queue.claim, self.concurrency)
            if not items:
                await asyncio.sleep(self.idle_s)
                continue
            await asyncio.gather(*(self._one(i) for i in items))

    async def _one(self, item: BatchItem) -> None:
        try:
            body = await self.runner(item.body)
        except QuotaExhausted:
            # Quota exhausted — this is not a job failure, it is a reason
            # to wait.
            await asyncio.sleep(self.requeue_delay_s)
            await asyncio.to_thread(self.queue.finish, item, None, None, requeue=True)
        except Exception as exc:  # noqa: BLE001
            await asyncio.to_thread(
                self.queue.finish, item, None, f"{type(exc).__name__}: {exc}"[:400]
            )
        else:
            await asyncio.to_thread(self.queue.finish, item, body)


class QuotaExhausted(Exception):
    """All candidates are out of quota. For a batch this is temporary."""
