"""Per-Model Quality Predictor and Prior Score Fallback for Waypost v5 (Spec Section K.1).

Estimates P(pass | query, model). When trained weights are present, uses linear/MLP
projection over query embeddings; otherwise gracefully falls back to the Bayesian
prior quality_score from the manifest.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .registry import Offering
from .schemas import RequestProfile


class ModelQualityPredictor:
    def __init__(self, weights_path: Path | str | None = None):
        self.weights_path = Path(weights_path) if weights_path else None
        self.weights: dict[str, Any] = {}
        self.is_trained = False
        if self.weights_path and self.weights_path.exists():
            self.load(self.weights_path)

    def load(self, path: Path | str) -> None:
        """Loads trained weights from disk."""
        p = Path(path)
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    self.weights = json.load(f)
                self.is_trained = True
            except Exception:
                self.weights = {}
                self.is_trained = False

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
        """Predicts probability of success P(pass | query, model).

        If model has trained weights and embedding is available, uses prediction.
        Otherwise falls back to task-specific quality_for(task_class).
        """
        model_key = offering.key
        if self.is_trained and model_key in self.weights and embedding:
            model_w = self.weights[model_key]
            bias = model_w.get("bias", 0.0)
            coefs = model_w.get("coefs", [])
            if len(coefs) == len(embedding):
                # Sigmoid of dot product
                import math

                raw = bias + sum(c * x for c, x in zip(coefs, embedding))
                raw_clamped = max(-20.0, min(20.0, raw))
                return 1.0 / (1.0 + math.exp(-raw_clamped))

        # Fallback to Bayesian prior from manifest
        return offering.quality_for(profile.task_class)
