"""Beta measurement and error overlap analytics for Waypost v5 (Spec Section H).

Calculates:
- p_best: Accuracy of the single best model across benchmark queries.
- beta: Fraction of queries where ALL models failed simultaneously (joint failure floor).
- 1 - beta: Theoretical maximum upper ceiling for any ensemble or selection policy.
- Delta = (1 - beta) - p_best: Maximum potential benefit of ensembling.
- Error correlation matrix between all model pairs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BetaMetrics:
    total_queries: int
    models: list[str]
    model_accuracies: dict[str, float]
    p_best_model: str
    p_best: float
    beta: float
    ceiling: float
    potential_gain: float
    verdict: str
    error_correlation: dict[str, dict[str, float]] = field(default_factory=dict)
    all_failed_query_indices: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_queries": self.total_queries,
            "models_count": len(self.models),
            "p_best_model": self.p_best_model,
            "p_best": round(self.p_best, 4),
            "beta": round(self.beta, 4),
            "ceiling": round(self.ceiling, 4),
            "potential_gain": round(self.potential_gain, 4),
            "verdict": self.verdict,
            "model_accuracies": {
                k: round(v, 4) for k, v in self.model_accuracies.items()
            },
            "all_failed_count": len(self.all_failed_query_indices),
        }


def compute_beta_metrics(
    eval_matrix: dict[str, list[bool]],
    gain_threshold: float = 0.05,
) -> BetaMetrics:
    """Computes beta metrics from a matrix of model execution outcomes.

    Args:
        eval_matrix: Dictionary mapping model_id to a list of booleans (True = pass, False = fail).
                     All lists must have the same length N (number of queries).
        gain_threshold: Minimum required gain (default 5%) to justify Phase L ensemble.
    """
    if not eval_matrix:
        return BetaMetrics(
            total_queries=0,
            models=[],
            model_accuracies={},
            p_best_model="",
            p_best=0.0,
            beta=1.0,
            ceiling=0.0,
            potential_gain=0.0,
            verdict="No data provided for beta measurement.",
        )

    models = sorted(eval_matrix.keys())
    n_queries = len(eval_matrix[models[0]])

    if n_queries == 0:
        return BetaMetrics(
            total_queries=0,
            models=models,
            model_accuracies={m: 0.0 for m in models},
            p_best_model="",
            p_best=0.0,
            beta=1.0,
            ceiling=0.0,
            potential_gain=0.0,
            verdict="Query set is empty.",
        )

    # 1. Model Accuracies
    accuracies: dict[str, float] = {}
    for m in models:
        passes = sum(1 for res in eval_matrix[m] if res)
        accuracies[m] = passes / n_queries

    p_best_model, p_best = max(accuracies.items(), key=lambda kv: kv[1])

    # 2. Joint failure rate (Beta) & all-failed queries
    all_failed_indices: list[int] = []
    for q_idx in range(n_queries):
        all_failed = all(not eval_matrix[m][q_idx] for m in models)
        if all_failed:
            all_failed_indices.append(q_idx)

    beta = len(all_failed_indices) / n_queries
    ceiling = 1.0 - beta
    potential_gain = max(0.0, ceiling - p_best)

    # 3. Error Correlation Matrix
    corr_matrix: dict[str, dict[str, float]] = {m: {} for m in models}
    for i, m1 in enumerate(models):
        err1 = [not r for r in eval_matrix[m1]]
        for m2 in models:
            err2 = [not r for r in eval_matrix[m2]]
            # Jaccard index of errors or joint failure overlap
            both_err = sum(1 for e1, e2 in zip(err1, err2) if e1 and e2)
            total_err = sum(1 for e1, e2 in zip(err1, err2) if e1 or e2)
            corr_matrix[m1][m2] = round(both_err / total_err if total_err else 0.0, 4)

    # 4. Verdict
    if potential_gain < gain_threshold:
        verdict = (
            f"Ensemble NOT justified: Potential gain {potential_gain * 100:.1f}% < {gain_threshold * 100:.0f}%. "
            f"Single best model ({p_best_model} @ {p_best * 100:.1f}%) is optimal."
        )
    else:
        verdict = (
            f"Ensemble PROMISES improvement: Ceiling is {ceiling * 100:.1f}% "
            f"(+{potential_gain * 100:.1f}% over {p_best_model} @ {p_best * 100:.1f}%). Proceed to Phase L."
        )

    return BetaMetrics(
        total_queries=n_queries,
        models=models,
        model_accuracies=accuracies,
        p_best_model=p_best_model,
        p_best=p_best,
        beta=beta,
        ceiling=ceiling,
        potential_gain=potential_gain,
        verdict=verdict,
        error_correlation=corr_matrix,
        all_failed_query_indices=all_failed_indices,
    )
