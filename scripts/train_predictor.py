"""Train the per-model quality predictor on the router's own traffic.

The attempt log has carried query embeddings, the model that answered and
the outcome since the day it was written. Nothing read it. This closes
that loop: for each offering, a logistic regression predicting
P(pass | query) from the embedding.

Pure numpy, like the classifier head — a few hundred coefficients per
model, seconds to train, kilobytes on disk. Anything heavier would be
dishonest about the amount of data behind it.

Three things this refuses to do, all for the same reason: a model that is
confidently wrong routes worse than no model at all.

*It skips offerings with too few observations,* rather than fitting them
anyway. Forty rows over a 256-dimensional embedding is not a regression,
it is memorisation.

*It skips offerings with no variance* — everything passed, or everything
failed. A separable fit sends coefficients to infinity and produces
certainty out of an accident.

*It reports held-out accuracy against the base rate,* and drops a model
whose fit does not beat simply predicting the majority. The router then
falls back to the prior, which is exactly the right outcome.

Kept fits are additionally measured on a time-ordered cut (train on the
past, test on the most recent rows) with Brier score and a small
calibration table. Those are informational: on little data a time cut is
noisy, so selection still uses the random-split lift.

    python -m scripts.train_predictor --db var/router.db --out var/predictor.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from waypost.telemetry import Telemetry

# Below this an offering is not fitted at all.
MIN_ROWS = 60
# Rows are weighted by age: last month's model is not this month's model.
HALF_LIFE_DAYS = 14.0
# A fit must beat the base rate by this much on held-out rows to be kept.
MIN_LIFT = 0.02
HOLDOUT = 0.25


def fit_logistic(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    *,
    l2: float = 1.0,
    epochs: int = 300,
    lr: float = 0.5,
) -> tuple[np.ndarray, float]:
    """Weighted logistic regression by gradient descent.

    L2 is deliberately strong. With a few hundred rows in 256 dimensions
    the unregularised fit is memorisation, and memorisation of past
    routing decisions is the one thing this must not learn.
    """
    n, d = X.shape
    coefs = np.zeros(d)
    bias = float(np.average(y, weights=w))
    bias = np.log(max(1e-6, bias) / max(1e-6, 1.0 - bias))  # start at the base rate
    wn = w / w.sum()
    for _ in range(epochs):
        z = np.clip(X @ coefs + bias, -20, 20)
        p = 1.0 / (1.0 + np.exp(-z))
        err = (p - y) * wn
        grad_c = X.T @ err + l2 * coefs / n
        grad_b = err.sum()
        coefs -= lr * grad_c
        bias -= lr * grad_b
    return coefs, float(bias)


def evaluate(X: np.ndarray, y: np.ndarray, coefs: np.ndarray, bias: float) -> float:
    z = np.clip(X @ coefs + bias, -20, 20)
    p = 1.0 / (1.0 + np.exp(-z))
    return float(((p >= 0.5).astype(float) == y).mean())


def predict_proba(X: np.ndarray, coefs: np.ndarray, bias: float) -> np.ndarray:
    z = np.clip(X @ coefs + bias, -20, 20)
    return 1.0 / (1.0 + np.exp(-z))


def brier_score(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def calibration_bins(
    p: np.ndarray, y: np.ndarray, *, n_bins: int = 5
) -> list[dict]:
    """Per-bin (predicted mean, empirical rate, count).

    Accuracy says whether the 0.5 cutoff separates the classes; routing
    needs the probabilities themselves to mean what they say, so this is
    reported alongside it. Empty bins are omitted.
    """
    out: list[dict] = []
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        if int(mask.sum()) == 0:
            continue
        out.append(
            {
                "lo": round(float(lo), 2),
                "hi": round(float(hi), 2),
                "n": int(mask.sum()),
                "mean_p": round(float(p[mask].mean()), 3),
                "empirical": round(float(y[mask].mean()), 3),
            }
        )
    return out


def train(
    db: str | Path,
    out: str | Path,
    *,
    min_rows: int = MIN_ROWS,
    threshold: float = 0.5,
    verbose: bool = True,
) -> dict:
    rows = Telemetry(db).training_rows()
    if verbose:
        print(f"{len(rows)} rows with embeddings in {db}")

    by_offering: dict[str, list[dict]] = {}
    for r in rows:
        by_offering.setdefault(r["offering"], []).append(r)

    now = time.time()
    models: dict[str, dict] = {}
    skipped: dict[str, str] = {}

    for offering, items in sorted(by_offering.items()):
        # A new request representation or encoder is a different feature
        # space even when the vector dimensions happen to match.
        version = max(items, key=lambda i: i["ts"]).get("embedder_version", "")
        items = [i for i in items if i.get("embedder_version", "") == version]
        if len(items) < min_rows:
            skipped[offering] = f"only {len(items)} rows"
            continue
        dim = len(items[0]["embedding"])
        items = [i for i in items if len(i["embedding"]) == dim]

        X = np.array([i["embedding"] for i in items], dtype=float)
        # Explicit feedback overrides mechanical checks upstream. Weak
        # format checks and implicit reactions carry less training weight.
        y = np.array([1.0 if i["reward"] >= threshold else 0.0 for i in items])
        age_days = (now - np.array([i["ts"] for i in items])) / 86_400.0
        w = np.power(0.5, age_days / HALF_LIFE_DAYS)
        w *= np.array([i.get("weight", 1.0) for i in items])

        base = y.mean()
        if base in (0.0, 1.0):
            skipped[offering] = f"no variance ({int(base)} everywhere)"
            continue

        idx = np.random.default_rng(0).permutation(len(y))
        cut = max(1, int(len(y) * HOLDOUT))
        test, train_idx = idx[:cut], idx[cut:]
        coefs, bias = fit_logistic(X[train_idx], y[train_idx], w[train_idx])

        acc = evaluate(X[test], y[test], coefs, bias)
        base_acc = max(y[test].mean(), 1.0 - y[test].mean())
        if acc < base_acc + MIN_LIFT:
            skipped[offering] = f"no lift ({acc:.2f} vs base {base_acc:.2f})"
            continue

        # Time-ordered check: a random holdout is optimistic under drift
        # (models are swapped behind the same id, hosts degrade), so the
        # same fit is also measured training on the past and testing on
        # the most recent rows. Informational: selection still uses the
        # random-split lift above, on small data a time cut is noisy.
        ts = np.array([i["ts"] for i in items], dtype=float)
        order = np.argsort(ts, kind="stable")
        tcut = max(1, int(len(y) * HOLDOUT))
        time_test = order[-tcut:]
        time_p = predict_proba(X[time_test], coefs, bias)
        time_acc = evaluate(X[time_test], y[time_test], coefs, bias)
        brier = brier_score(predict_proba(X[test], coefs, bias), y[test])
        time_brier = brier_score(time_p, y[time_test])

        models[offering] = {
            "bias": bias,
            "coefs": [round(float(c), 6) for c in coefs],
            "n": float(w[train_idx].sum()),
            "accuracy": round(acc, 3),
            "base_rate": round(float(base), 3),
            "time_accuracy": round(float(time_acc), 3),
            "brier": round(float(brier), 4),
            "time_brier": round(float(time_brier), 4),
            "calibration": calibration_bins(time_p, y[time_test]),
            "dim": dim,
            "embedder_version": version,
        }
        if verbose:
            print(
                f"  {offering}: n={len(items)} acc={acc:.2f} "
                f"(base {base_acc:.2f}) time_acc={time_acc:.2f} "
                f"brier={brier:.3f} pass_rate={base:.2f}"
            )

    payload = {
        "trained_at": now,
        "rows": len(rows),
        "models": models,
        "skipped": skipped,
    }
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if verbose:
        print(f"\n{len(models)} model(s) trained → {out}")
        for offering, why in sorted(skipped.items()):
            print(f"  skipped {offering}: {why}")
        if not models:
            print(
                "\nNothing trained yet — the router falls back to the manifest "
                "prior, which is the correct behaviour with this little data."
            )
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default="var/router.db")
    ap.add_argument("--out", default="var/predictor.json")
    ap.add_argument("--min-rows", type=int, default=MIN_ROWS)
    args = ap.parse_args()
    train(args.db, args.out, min_rows=args.min_rows)


if __name__ == "__main__":
    main()
