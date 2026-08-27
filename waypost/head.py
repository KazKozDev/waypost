"""L1 classifier head.

On top of static embeddings (potion-multilingual-128M, 256-dim) a small
head is trained: softmax regression on task class + linear regression on
complexity. Trains in seconds, weighs kilobytes, retrains on a cron
schedule.

The teacher is NVIDIA prompt-task-and-complexity-classifier (English) +
a local model (Russian), see scripts/train_head.py. Before training the
head is absent — and that is a normal state for the first weeks: the
classifier runs on rules.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

TASK_CLASSES = ["chat", "extraction", "reasoning", "code", "creative"]


def _softmax(z, axis: int = -1) -> np.ndarray:
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def _onehot(y: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros((y.size, n))
    out[np.arange(y.size), y] = 1.0
    return out


class TaskHead:
    """Pure numpy: no sklearn, no torch. Trains in seconds."""

    def __init__(self, dim: int = 256, classes: list[str] | None = None):
        self.dim = dim
        self.classes = classes or TASK_CLASSES
        self.W = np.zeros((len(self.classes), dim))
        self.b = np.zeros(len(self.classes))
        self.cw = np.zeros(dim)  # complexity regression weights
        self.cb = 0.5
        self._mean = np.zeros(dim)
        self._std = np.ones(dim)

    # ------------------------------------------------------------ predict
    def predict(self, embedding) -> tuple[dict[str, float], float, float]:
        x = np.asarray(embedding, dtype=np.float64).reshape(-1)
        if x.shape[0] != self.dim:
            x = x[: self.dim]
        xs = (x - self._mean) / self._std
        probs = _softmax(self.W @ xs + self.b)
        complexity = float(np.clip(self.cw @ xs + self.cb, 0.0, 1.0))
        confidence = float(probs.max())
        return dict(zip(self.classes, probs.tolist())), complexity, confidence

    # ------------------------------------------------------------- train
    def train(
        self,
        X,
        y_task,
        y_complexity,
        *,
        epochs: int = 200,
        lr: float = 0.1,
        l2: float = 1e-4,
    ) -> None:
        X = np.asarray(X, dtype=np.float64)
        y = np.array([self.classes.index(t) for t in y_task])
        c = np.asarray(y_complexity, dtype=np.float64)
        n = X.shape[0]
        self._mean = X.mean(0)
        self._std = X.std(0) + 1e-9
        Xs = (X - self._mean) / self._std

        for _ in range(epochs):
            logits = Xs @ self.W.T + self.b
            p = _softmax(logits, axis=1)
            grad = (p - _onehot(y, len(self.classes))) / n
            self.W -= lr * (grad.T @ Xs + l2 * self.W)
            self.b -= lr * (grad.sum(0) + l2 * self.b)

            pred = Xs @ self.cw + self.cb
            self.cw -= lr * (Xs.T @ (pred - c) / n + l2 * self.cw)
            self.cb -= lr * ((pred - c).mean() + l2 * self.cb)

    # ---------------------------------------------------------------- io
    def save(self, path: str | Path) -> None:
        data = {
            "dim": self.dim,
            "classes": self.classes,
            "W": self.W.tolist(),
            "b": self.b.tolist(),
            "cw": self.cw.tolist(),
            "cb": self.cb,
            "mean": self._mean.tolist(),
            "std": self._std.tolist(),
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(data))

    @classmethod
    def load(cls, path: str | Path) -> "TaskHead":
        data = json.loads(Path(path).read_text())
        h = cls(dim=data["dim"], classes=data["classes"])
        h.W = np.array(data["W"])
        h.b = np.array(data["b"])
        h.cw = np.array(data["cw"])
        h.cb = data["cb"]
        h._mean = np.array(data["mean"])
        h._std = np.array(data["std"])
        return h
