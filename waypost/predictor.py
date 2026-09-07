"""Per-model quality predictor: P(pass | query, model).

The bandit knows how a model does on a *class* of tasks — five buckets.
Two requests inside "code" can be renaming a variable and writing a
parser, and the bandit averages them together. The query embedding is the
thing that tells them apart, and it is already computed for every
request, logged with every attempt, and was until now used by nobody.

This is a logistic regression per offering over that embedding, trained
offline on the attempt log by scripts/train_predictor.py. Deliberately
small: a few hundred coefficients per model, trained in seconds, honest
about how little data it has.

Confidence matters more than the estimate here. With forty observations
a per-model regression is mostly noise, so a prediction is blended into
the manifest/bandit prior in proportion to how much data stands behind
it, rather than replacing it outright. A predictor that is trusted before
it has earned it is worse than no predictor: it routes confidently on
nothing.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .registry import Offering
from .schemas import RequestProfile


class ModelQualityPredictor:
    # Observations at which a per-model regression is trusted on its own.
    # Below it the prediction is mixed with the prior in proportion to
    # the data behind it.
    FULL_TRUST_N = 200

    def __init__(self, weights_path: Path | str | None = None):
        self.weights_path = Path(weights_path) if weights_path else None
        self.weights: dict[str, Any] = {}
        self.is_trained = False
        self.trained_at: float = 0.0
        self.hits = 0
        self.misses = 0
        if self.weights_path and self.weights_path.exists():
            self.load(self.weights_path)

    def load(self, path: Path | str) -> None:
        """Loads trained weights from disk."""
        p = Path(path)
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                # The file holds models plus metadata; older files were a
                # bare mapping of offering → weights.
                self.weights = payload.get("models", payload)
                self.trained_at = float(payload.get("trained_at", 0.0))
                self.is_trained = bool(self.weights)
            except Exception:
                self.weights = {}
                self.is_trained = False

    def reload(self) -> bool:
        """Pick up weights retrained since startup, without a restart."""
        if not self.weights_path or not self.weights_path.exists():
            return False
        before = self.trained_at
        self.load(self.weights_path)
        return self.trained_at > before

    def save(self, path: Path | str) -> None:
        """Saves weights to disk."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.weights, f, indent=2)

    def predict_p_pass(
        self,
        offering: Offering,
        profile: RequestProfile,
        embedding: list[float] | None = None,
    ) -> float:
        """P(pass | this query, this model), blended with the prior."""
        prior = offering.quality_for(profile.task_class)
        model_w = self.weights.get(offering.key) if self.is_trained else None
        if not model_w or not embedding:
            self.misses += 1
            return prior

        coefs = model_w.get("coefs") or []
        if len(coefs) != len(embedding):
            # The embedder changed under the weights. Stale coefficients
            # applied to a different vector space are not a degraded
            # prediction, they are a random one.
            self.misses += 1
            return prior

        raw = model_w.get("bias", 0.0) + sum(
            c * x for c, x in zip(coefs, embedding)
        )
        p = 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, raw))))

        n = float(model_w.get("n", 0))
        trust = min(1.0, n / self.FULL_TRUST_N)
        self.hits += 1
        return trust * p + (1.0 - trust) * prior

    def snapshot(self) -> dict[str, Any]:
        total = self.hits + self.misses
        return {
            "trained": self.is_trained,
            "models": len(self.weights),
            "trained_at": int(self.trained_at),
            "coverage": round(self.hits / total, 3) if total else 0.0,
        }
