"""Local embeddings + microbatching.

Two reasons to keep embeddings local rather than at the provider:

  1. Quota. Embeddings are the most frequent and cheapest call; burning
     a free tier's RPD on them is silly when a Mac Studio computes them
     for free, forever.
  2. Privacy. A text embedding goes to the same provider as the text
     itself. For privacy: strict that is unacceptable.

Microbatching (§4.2): a 20-50 ms window, gluing single calls into one
encoder pass. For interactive chat, batching is a direct TTFT loss, but
for embeddings it is the opposite: they come in batches (RAG, the
semantic cache), and the latency of a single vector matters to no one.

Backends in descending quality:
  model2vec (potion-multilingual-128M) — static embeddings, no forward
    pass, multilingual support is mandatory for mixed ru/en;
  hashing — a deterministic char-n-gram fallback with no dependencies.
    Good for development and tests, but not for a production threshold of
    the semantic cache: it is lexical, not semantic.
"""
from __future__ import annotations

import asyncio
import hashlib
import re
import threading
import time
from dataclasses import dataclass, field

import numpy as np

DEFAULT_MODEL = "minishlab/potion-multilingual-128M"
MAX_CHARS = 8000  # longer text is truncated: the tail barely affects the vector


# --------------------------------------------------------------- backends


class HashingEncoder:
    """Char-n-gram hashing. No dependencies, deterministic.

    Honest caveat: this is lexical similarity, not semantic. "car" and
    "automobile" are different to it. So do not tune the semantic-cache
    threshold on it — a real encoder is needed.
    """

    name = "hashing"

    def __init__(self, dim: int = 256, ngrams: tuple[int, ...] = (3, 4, 5)):
        self.dim = dim
        self.ngrams = ngrams

    def _vector(self, text: str) -> np.ndarray:
        text = re.sub(r"\s+", " ", text.strip().lower())[:MAX_CHARS]
        vec = np.zeros(self.dim, dtype=np.float64)
        if not text:
            return vec
        for n in self.ngrams:
            for i in range(max(0, len(text) - n + 1)):
                gram = text[i : i + n].encode("utf-8")
                h = int.from_bytes(
                    hashlib.blake2b(gram, digest_size=8).digest(), "little"
                )
                vec[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vector(t) for t in texts])


class StaticEncoder:
    """model2vec. Loaded lazily: downloading the model must not happen on
    package import."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        from model2vec import StaticModel  # noqa: PLC0415

        self._model = StaticModel.from_pretrained(model_name)
        self.name = f"model2vec:{model_name}"
        self.dim = int(self._model.dim)

    def encode(self, texts: list[str]) -> np.ndarray:
        arr = np.asarray(
            self._model.encode([t[:MAX_CHARS] for t in texts]), dtype=np.float64
        )
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms


def load_encoder(model_name: str = DEFAULT_MODEL, *, allow_fallback: bool = True):
    """A real encoder if the package is installed; otherwise hashing.

    There must be no silent failure here: the difference between a real
    encoder and the fallback is visible in /v1/stats and in the `backend`
    field of the response.
    """
    try:
        return StaticEncoder(model_name)
    except Exception:  # noqa: BLE001
        if not allow_fallback:
            raise
        return HashingEncoder()


# ------------------------------------------------------------- microbatch


@dataclass
class _Job:
    texts: list[str]
    future: asyncio.Future = field(repr=False)


class EmbeddingService:
    """A synchronous encode for the hot path, an asynchronous one with
    microbatching.

    The hot path (the semantic cache) calls encode() directly: it has one
    text, and waiting for a batch window for it is pointless. Batches
    (the /v1/embeddings endpoint, reindexing) go through aencode().
    """

    def __init__(
        self,
        encoder=None,
        *,
        window_ms: float = 25.0,
        max_batch: int = 64,
        model_name: str = DEFAULT_MODEL,
    ):
        self._encoder = encoder if encoder is not None else load_encoder(model_name)
        self.window_s = window_ms / 1000.0
        self.max_batch = max_batch
        self._queue: asyncio.Queue[_Job] | None = None
        self._worker: asyncio.Task | None = None
        self._lock = threading.Lock()
        self.stats = {"calls": 0, "texts": 0, "batches": 0, "batched_texts": 0}

    # ------------------------------------------------------------ props
    @property
    def backend(self) -> str:
        return getattr(self._encoder, "name", type(self._encoder).__name__)

    @property
    def dim(self) -> int:
        return int(getattr(self._encoder, "dim", 256))

    @property
    def is_fallback(self) -> bool:
        return self.backend == "hashing"

    # -------------------------------------------------------------- encode
    def encode(self, texts: list[str]) -> np.ndarray:
        with self._lock:
            self.stats["calls"] += 1
            self.stats["texts"] += len(texts)
        if not texts:
            return np.zeros((0, self.dim))
        return self._encoder.encode(texts)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    async def aencode(self, texts: list[str]) -> np.ndarray:
        """Through the microbatch window. Calls that land in one window
        are glued into a single encoder pass."""
        if not texts:
            return np.zeros((0, self.dim))
        loop = asyncio.get_running_loop()
        if self._queue is None:
            self._queue = asyncio.Queue()
        if self._worker is None or self._worker.done():
            self._worker = loop.create_task(self._run())
        job = _Job(texts=texts, future=loop.create_future())
        await self._queue.put(job)
        return await job.future

    async def _run(self) -> None:
        assert self._queue is not None
        while True:
            job = await self._queue.get()
            batch = [job]
            deadline = time.perf_counter() + self.window_s
            total = len(job.texts)
            # Window: pick up everything that arrived within window_ms,
            # but no more than max_batch texts — otherwise one big call
            # delays everyone.
            while total < self.max_batch:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    nxt = await asyncio.wait_for(self._queue.get(), remaining)
                except (asyncio.TimeoutError, TimeoutError):
                    break
                batch.append(nxt)
                total += len(nxt.texts)

            flat = [t for j in batch for t in j.texts]
            try:
                vectors = await asyncio.to_thread(self.encode, flat)
            except Exception as exc:  # noqa: BLE001
                for j in batch:
                    if not j.future.done():
                        j.future.set_exception(exc)
                continue
            with self._lock:
                self.stats["batches"] += 1
                self.stats["batched_texts"] += len(flat)
            pos = 0
            for j in batch:
                chunk = vectors[pos : pos + len(j.texts)]
                pos += len(j.texts)
                if not j.future.done():
                    j.future.set_result(chunk)

    async def aclose(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            try:
                await self._worker
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._worker = None

    def snapshot(self) -> dict:
        with self._lock:
            s = dict(self.stats)
        s["backend"] = self.backend
        s["dim"] = self.dim
        # Average batch size — the only way to see whether microbatching
        # works at all, or each call goes alone.
        s["avg_batch"] = (
            round(s["batched_texts"] / s["batches"], 2) if s["batches"] else 0.0
        )
        return s
