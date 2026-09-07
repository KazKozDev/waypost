"""Routing by resemblance: who handled requests like this one?

The bandit reasons about five buckets. Two requests inside "code" can be
renaming a variable and writing a parser, and it averages them together.
The query embedding tells them apart, and it is computed for every
request anyway.

So: keep the recent attempt log in memory as vectors, and for a new query
find the nearest past requests and ask how each offering did on *those*.
It is a k-nearest-neighbour estimate — no training step, no weights file,
and it starts being useful after a few hundred rows rather than a few
thousand.

This complements the predictor rather than replacing it. The regression
generalises and needs enough data per model to be fitted at all; the
neighbourhood is local and works immediately, but says nothing about a
region of the space it has never seen. Both are folded into the prior by
how much evidence stands behind them, and where neither has anything to
say the manifest number is used — which is the honest answer.

Deliberately bounded: a fixed-size ring of recent rows, brute-force
cosine over a few thousand vectors. That is well under a millisecond in
numpy, and an approximate index would be machinery in exchange for
nothing at this scale.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any

import numpy as np

log = logging.getLogger("waypost.neighbors")


class NeighborIndex:
    def __init__(
        self,
        *,
        capacity: int = 5000,
        k: int = 24,
        min_neighbors: int = 4,
        full_trust_n: int = 12,
        half_life_days: float = 14.0,
    ):
        self.capacity = capacity
        self.k = k
        # Fewer than this among the neighbours and the estimate is one or
        # two anecdotes; the prior is a better answer than a coin flip
        # dressed as evidence.
        self.min_neighbors = min_neighbors
        self.full_trust_n = full_trust_n
        self.half_life_days = half_life_days
        self._vectors: np.ndarray | None = None  # (n, dim), L2-normalised
        self._rewards: np.ndarray = np.zeros(0)
        self._ts: np.ndarray = np.zeros(0)
        self._offerings: list[str] = []
        self._dim: int | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------- build
    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v, axis=-1, keepdims=True)
        return v / np.maximum(n, 1e-9)

    def add(self, embedding: Any, offering: str, reward: float, ts: float = 0.0) -> bool:
        vec = np.asarray(embedding, dtype=np.float32).ravel()
        if vec.size == 0:
            return False
        with self._lock:
            if self._dim is None:
                self._dim = vec.size
            elif vec.size != self._dim:
                # A different embedder. Mixing vector spaces produces
                # neighbours that are not neighbours of anything.
                return False
            row = self._normalize(vec)[None, :]
            if self._vectors is None:
                self._vectors = row
            else:
                self._vectors = np.vstack([self._vectors, row])
            self._rewards = np.append(self._rewards, float(reward))
            self._ts = np.append(self._ts, ts or time.time())
            self._offerings.append(offering)
            if len(self._offerings) > self.capacity:
                drop = len(self._offerings) - self.capacity
                self._vectors = self._vectors[drop:]
                self._rewards = self._rewards[drop:]
                self._ts = self._ts[drop:]
                del self._offerings[:drop]
        return True

    def load(self, rows: list[dict[str, Any]]) -> int:
        """Fill from telemetry.training_rows() at startup."""
        added = 0
        for r in sorted(rows, key=lambda x: x.get("ts", 0.0)):
            if self.add(r["embedding"], r["offering"], r["reward"], r.get("ts", 0.0)):
                added += 1
        if added:
            log.info("neighbors: %d past attempts indexed", added)
        return added

    # ------------------------------------------------------------ lookup
    def estimate(self, embedding: Any) -> dict[str, tuple[float, float]]:
        """offering → (mean reward among neighbours, evidence weight).

        One pass for all offerings: the neighbourhood is found once and
        then split by who answered, which is both faster and fairer than
        asking per candidate.
        """
        with self._lock:
            if self._vectors is None or len(self._offerings) < self.min_neighbors:
                return {}
            vectors = self._vectors
            offerings = list(self._offerings)
            rewards = self._rewards
            ts = self._ts

        query = np.asarray(embedding, dtype=np.float32).ravel()
        if self._dim is None or query.size != self._dim:
            return {}
        query = self._normalize(query)

        sims = vectors @ query
        k = min(self.k, sims.size)
        idx = np.argpartition(-sims, k - 1)[:k]

        now = time.time()
        out: dict[str, list[tuple[float, float]]] = {}
        for i in idx:
            sim = float(sims[i])
            if sim <= 0.0:
                continue
            # Weight by both resemblance and age: an old answer from a
            # model that has since been swapped behind the same id is
            # weak evidence about today.
            age_days = max(0.0, (now - float(ts[i])) / 86_400.0)
            weight = sim * (0.5 ** (age_days / self.half_life_days))
            out.setdefault(offerings[i], []).append((float(rewards[i]), weight))

        estimates: dict[str, tuple[float, float]] = {}
        for offering, points in out.items():
            total_w = sum(w for _, w in points)
            if total_w <= 0:
                continue
            mean = sum(r * w for r, w in points) / total_w
            # Trust grows with the number of neighbours, not their weight:
            # ten near-identical rows from one burst are one observation
            # wearing ten hats.
            trust = min(1.0, len(points) / self.full_trust_n)
            estimates[offering] = (mean, trust)
        return estimates

    def escalation_risk(self, embedding: Any) -> tuple[float, float]:
        """(share of similar past requests that needed escalating, trust).

        The cascade is reactive: try the cheap model, verify, escalate.
        Every escalation pays for the cheap attempt twice — once in
        latency, once in quota. When queries like this one have needed
        escalating before, the cheap rung is not a saving, it is a tax,
        and the plan should start higher.

        This reuses the neighbourhood rather than training a second
        model: the question "did requests like this one work out" is the
        same question, read the other way round.
        """
        est = self.estimate(embedding)
        if not est:
            return 0.0, 0.0
        # Weight each offering's local success by how much of the
        # neighbourhood it accounts for, then read the failure share.
        total_trust = sum(trust for _, trust in est.values())
        if total_trust <= 0:
            return 0.0, 0.0
        mean_reward = sum(m * t for m, t in est.values()) / total_trust
        return max(0.0, min(1.0, 1.0 - mean_reward)), min(1.0, total_trust)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "rows": len(self._offerings),
                "capacity": self.capacity,
                "dim": self._dim,
                "offerings": len(set(self._offerings)),
            }
