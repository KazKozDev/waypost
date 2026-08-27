"""Response cache.

L0 — exact: SHA-256 of the canonicalized request. Zero false-hit risk,
near-zero cost. Done first.
L1 — prefix caching on the provider side. Needs no code, only
discipline: a stable block order and sticky routing
(see router.Router._sticky and normalize_order below).
L2 — semantic. An extension point: a bi-encoder fetches candidates, a
cross-encoder confirms. Do not enable without a verifier — "compare A
and B" and "compare B and A" are close in vector space, but the answers
differ.

Only temperature == 0 is cached: at a non-zero temperature the user asks
for variety, and returning the same answer is not optimization but a bug.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from .schemas import ChatRequest


def canonical_key(req: ChatRequest, model_class: str = "auto") -> str:
    """Canonicalization: only what affects the answer. session_id,
    no_cache and other router extensions are not in the key."""
    payload = {
        "model_class": model_class,
        "messages": [
            {
                "role": m.role,
                "content": m.content,
                "name": m.name,
                "tool_calls": m.tool_calls,
                "tool_call_id": m.tool_call_id,
                "function_call": m.function_call,
            }
            for m in req.messages
        ],
        "tools": req.tools or req.functions,
        "tool_choice": req.tool_choice or req.function_call,
        "response_format": req.response_format,
        "temperature": req.temperature,
        "top_p": req.top_p,
        "max_tokens": req.max_tokens,
        "seed": req.seed,
        "stop": req.stop,
    }
    blob = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def cacheable(req: ChatRequest) -> bool:
    if req.no_cache or req.stream:
        return False
    return req.temperature == 0.0


def normalize_order(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Stabilize the prefix for the provider-side cache: system blocks
    first, order within preserved. Do not reorder anything beyond this —
    any permutation breaks the prefix entirely."""
    system = [m for m in messages if m.get("role") == "system"]
    rest = [m for m in messages if m.get("role") != "system"]
    return system + rest


class ExactCache:
    def __init__(self, db_path: str | Path, ttl_s: int = 86_400):
        self.ttl_s = ttl_s
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._conn() as c:
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY, body TEXT,
                    created_at REAL, hits INTEGER DEFAULT 0)
            """
            )
        self.stats = {"hit": 0, "miss": 0}

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def get(self, key: str) -> dict[str, Any] | None:
        with self._lock, self._conn() as c:
            row = c.execute(
                "SELECT body, created_at FROM cache WHERE key=?", (key,)
            ).fetchone()
            if row is None:
                self.stats["miss"] += 1
                return None
            body, created = row
            if time.time() - created > self.ttl_s:
                c.execute("DELETE FROM cache WHERE key=?", (key,))
                self.stats["miss"] += 1
                return None
            c.execute("UPDATE cache SET hits=hits+1 WHERE key=?", (key,))
            self.stats["hit"] += 1
            return json.loads(body)

    def put(self, key: str, body: dict[str, Any]) -> None:
        with self._lock, self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO cache (key, body, created_at, hits) "
                "VALUES (?,?,?,0)",
                (key, json.dumps(body, ensure_ascii=False), time.time()),
            )

    def hit_rate(self) -> float:
        total = self.stats["hit"] + self.stats["miss"]
        return self.stats["hit"] / total if total else 0.0


class SemanticCache:
    """L2 — semantic cache.

    A bi-encoder fetches candidates by cosine, a cross-encoder confirms
    (if set). Without a cross-encoder the threshold must be high —
    otherwise "compare A and B" and "compare B and A" give a false hit.

    Protection against false hits:
      - namespace by (system_hash, tools_hash, model_class, language)
      - a cross-encoder (bge-reranker-v2-m3) confirms the candidate
      - numbers and proper nouns — exact match on top of similarity
      - requests with a recency marker are not cached at all
      - TTL and explicit invalidation
    """

    def __init__(
        self,
        *,
        enabled: bool = False,
        threshold: float = 0.93,
        db_path: str | Path | None = None,
        ttl_s: int = 86_400,
        reranker=None,
        rerank_threshold: float = 0.5,
    ):
        self.enabled = enabled
        self.threshold = threshold
        self.ttl_s = ttl_s
        self.reranker = reranker  # cross-encoder: (q, cand) -> score
        self.rerank_threshold = rerank_threshold
        self._items: list[dict] = []
        self._lock = threading.Lock()
        self._db_path = Path(db_path) if db_path else None
        if self._db_path:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_db()
            self._load()
        self.stats = {"hit": 0, "miss": 0}

    def _init_db(self) -> None:
        with self._conn() as c:
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS semantic_cache (
                    namespace TEXT, query TEXT, embedding TEXT, body TEXT,
                    created_at REAL, hits INTEGER DEFAULT 0)
            """
            )

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _load(self) -> None:
        with self._conn() as c:
            rows = c.execute(
                "SELECT namespace, query, embedding, body, created_at "
                "FROM semantic_cache"
            ).fetchall()
        for ns, q, emb, body, created in rows:
            # Rows written before the shape normalization are stored as
            # [[...]]. Normalize on read: otherwise an old row breaks the
            # search even after a server restart.
            self._items.append(
                {
                    "namespace": ns,
                    "query": q,
                    "embedding": _vector(json.loads(emb)),
                    "body": json.loads(body),
                    "created_at": created,
                }
            )

    def _persist(self, item: dict) -> None:
        if not self._db_path:
            return
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO semantic_cache VALUES (?,?,?,?,?,0)",
                (
                    item["namespace"],
                    item["query"],
                    json.dumps(item["embedding"].tolist()),
                    json.dumps(item["body"], ensure_ascii=False),
                    item["created_at"],
                ),
            )

    # ------------------------------------------------------------ public
    def put(self, query: str, embedding, namespace: str, body: dict[str, Any]) -> None:
        item = {
            "namespace": namespace,
            "query": query,
            "embedding": _vector(embedding),
            "body": body,
            "created_at": time.time(),
        }
        with self._lock:
            self._items.append(item)
            self._persist(item)

    def _candidates(
        self, embedding, namespace: str, query: str, limit: int = 5
    ) -> list[tuple[float, dict]]:
        q_seq = _exact_sequence(query)
        emb = _vector(embedding)
        now = time.time()
        scored: list[tuple[float, dict]] = []
        with self._lock:
            for it in self._items:
                if it["namespace"] != namespace:
                    continue
                if now - it["created_at"] > self.ttl_s:
                    continue
                if not _exact_guard(q_seq, _exact_sequence(it["query"])):
                    continue
                scored.append((_cosine(emb, it["embedding"]), it))
        scored.sort(key=lambda p: p[0], reverse=True)
        return scored[:limit]

    def lookup(
        self,
        query: str,
        embedding,
        namespace: str,
        *,
        threshold: float | None = None,
        allow_recency: bool = False,
    ) -> dict[str, Any] | None:
        """The default threshold is from the constructor. An explicit
        threshold is needed for the degraded mode (see lookup_degraded)."""
        if not self.enabled:
            return None
        if not allow_recency and _has_recency_marker(query):
            # "now", "latest" — the answer goes stale faster than any TTL.
            return None
        bar = self.threshold if threshold is None else threshold
        pool = self._candidates(embedding, namespace, query)
        pool = [(sim, it) for sim, it in pool if sim >= bar]
        if not pool:
            self.stats["miss"] += 1
            return None

        if self.reranker is not None:
            # The cross-encoder sees both texts at once and catches the
            # inversions where cosine on static embeddings fails.
            rescored = [(self.reranker(query, it["query"]), it) for _, it in pool]
            rescored.sort(key=lambda p: p[0], reverse=True)
            best_score, best = rescored[0]
            if best_score < self.rerank_threshold:
                self.stats["miss"] += 1
                return None
        else:
            best = pool[0][1]

        self.stats["hit"] += 1
        return best["body"]

    def lookup_degraded(
        self, query: str, embedding, namespace: str, slack: float = 0.08
    ) -> dict[str, Any] | None:
        """The last step of the degradation ladder: all providers
        refused.

        The threshold is lowered, but the answer must be marked
        approximate (RouterMeta.approximate) — silently serving a similar
        answer as fresh is worse than honestly returning an error.
        """
        if not self.enabled:
            return None
        hit = self.lookup(
            query, embedding, namespace, threshold=max(0.0, self.threshold - slack)
        )
        if hit is not None:
            self.stats["degraded"] = self.stats.get("degraded", 0) + 1
        return hit

    def hit_rate(self) -> float:
        total = self.stats["hit"] + self.stats["miss"]
        return self.stats["hit"] / total if total else 0.0


# ------------------------------------------------------------- helpers

RECENCY_MARKERS = (
    "сейчас",
    "последний",
    "текущий",
    "сегодня",
    "now",
    "latest",
    "current",
    "today",
)


def _has_recency_marker(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in RECENCY_MARKERS)


_NUM_RE = re.compile(r"\d+")
_PROP_NOUN_RE = re.compile(r"\b[A-ZА-ЯЁ][a-zа-яё]+\b")


def _exact_sequence(text: str) -> tuple[str, ...]:
    """Numbers and proper nouns IN ORDER OF APPEARANCE.

    The order matters here. "Compare Moscow and Paris" and "compare
    Paris and Moscow" give the same entity set and nearly the same
    vector — but different answers. A set comparison lets that case
    through, a sequence comparison does not.
    """
    return tuple(m.group(0) for m in re.finditer(r"\d+|\b[A-ZА-ЯЁ][a-zа-яё]+\b", text))


def _exact_tokens(text: str) -> set[str]:
    """The set of the same entities — a cheap prefilter."""
    return set(_exact_sequence(text))


def _exact_guard(a, b) -> bool:
    """Pass a candidate only with a full match of entities and their
    order."""
    if isinstance(a, set) or isinstance(b, set):
        return set(a) == set(b)
    return tuple(a) == tuple(b)


def _vector(embedding) -> np.ndarray:
    """An embedding as a one-dimensional vector, whoever computed it.

    The cache accepts a vector from outside, and outside they are
    computed differently: (D,) from the classifier and (1, D) from a
    batch service. Normalize on entry — otherwise a shape mismatch
    surfaces not here but in np.dot deep inside the search, already as a
    500 on a live request.
    """
    return np.asarray(embedding, dtype=np.float64).reshape(-1)


def _cosine(a, b) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)
