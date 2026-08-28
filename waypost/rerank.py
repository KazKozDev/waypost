"""Cross-encoder reranker for the semantic cache and /v1/rerank.

bge-reranker-v2-m3 confirms a bi-encoder hit: it sees both texts at once
and catches inversions and negations where cosine similarity of static
embeddings fails. This fixes the main risk of §4.2 — false hits
("compare A and B" vs "compare B and A").

Lazy loading: the model loads on first call, not at startup — the
semantic cache and /v1/rerank are not always on, and holding ~1GB in
memory for nothing is wasteful. The first call after idle gives a
latency spike.

Backends in descending quality:
  sentence-transformers (bge-reranker-v2-m3) — a real cross-encoder;
  lexical — a deterministic dependency-free fallback (token Jaccard).
"""
from __future__ import annotations

import math
import re

DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


class LexicalReranker:
    """Jaccard fallback. Honest caveat: lexical, not semantic. Good for
    development, but do not tune the semantic-cache threshold on it — a
    real cross-encoder is needed for that."""

    name = "lexical"

    def score(self, query: str, candidate: str) -> float:
        a, b = _tokens(query), _tokens(candidate)
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    def rank(self, query: str, docs: list[str], top_n: int) -> list[tuple[int, float]]:
        scored = sorted(
            ((i, self.score(query, d)) for i, d in enumerate(docs)),
            key=lambda x: x[1],
            reverse=True,
        )
        return scored[:top_n]


class CrossEncoderReranker:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        from sentence_transformers import CrossEncoder  # noqa: PLC0415

        self._model = CrossEncoder(model_name)
        self.name = f"cross-encoder:{model_name}"
        # sentence-transformers squashes inside predict() itself: a
        # single-label head gets Sigmoid by default. A second sigmoid on
        # top folds every score into [0.5, 0.73] — then rerank_threshold
        # is below the minimum possible score and the verifier confirms
        # anything at all. Squash only if the backend returns raw logits.
        activation = getattr(self._model, "activation_fn", None) or getattr(
            self._model, "default_activation_function", None
        )
        self._raw_logits = type(activation).__name__ not in ("Sigmoid", "Softmax")

    def _to_score(self, value: float) -> float:
        if not self._raw_logits:
            return value
        return 1.0 / (1.0 + math.exp(-value))  # sigmoid → 0..1

    def score(self, query: str, candidate: str) -> float:
        return self._to_score(float(self._model.predict([[query, candidate]])[0]))

    def rank(self, query: str, docs: list[str], top_n: int) -> list[tuple[int, float]]:
        pairs = [[query, d] for d in docs]
        scored = sorted(
            (
                (i, self._to_score(float(val)))
                for i, val in enumerate(self._model.predict(pairs))
            ),
            key=lambda x: x[1],
            reverse=True,
        )
        return scored[:top_n]


def load_reranker(model_name: str = DEFAULT_MODEL, *, allow_fallback: bool = True):
    """A real cross-encoder if the package is installed; otherwise lexical."""
    try:
        return CrossEncoderReranker(model_name)
    except Exception:  # noqa: BLE001
        if not allow_fallback:
            raise
        return LexicalReranker()


class Reranker:
    """Lazy wrapper: the backend loads on first use."""

    def __init__(self, model_name: str = DEFAULT_MODEL, *, enabled: bool = True):
        self.model_name = model_name
        self.enabled = enabled
        self._backend = None

    def _load(self):
        if self._backend is None:
            self._backend = load_reranker(self.model_name)
        return self._backend

    @property
    def backend(self) -> str:
        return self._load().name

    def score(self, query: str, candidate: str) -> float:
        if not self.enabled:
            return 0.0
        return self._load().score(query, candidate)

    def rank(self, query: str, docs: list[str], top_n: int) -> list[tuple[int, float]]:
        if not self.enabled or not docs:
            return []
        return self._load().rank(query, docs, top_n)

    def as_verifier(self):
        """Callable for SemanticCache.reranker: (query, candidate) -> score."""
        return self.score
