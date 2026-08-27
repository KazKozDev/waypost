"""Configuration. Providers are added by editing the manifest, not the code."""
from __future__ import annotations

import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


def load_env(env_file: str | Path = ".env") -> None:
    """Load .env into os.environ.

    pydantic-settings reads .env into its Settings, but the registry reads
    keys straight from os.environ. Without this, keys from .env are not
    visible to the server. Existing environment variables are not
    overwritten.
    """
    p = Path(env_file)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip()
        if k and k not in os.environ:
            os.environ[k] = v


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ROUTER_", env_file=".env", extra="ignore"
    )

    host: str = "127.0.0.1"
    port: int = 8080
    manifest_path: Path = Path("config/providers.yaml")
    db_path: Path = Path("var/router.db")

    cache_ttl_s: int = 86_400
    request_timeout_s: float = 120.0
    max_attempts: int = 4

    # L1 classifier: requires `pip install model2vec` and a trained head.
    # Off by default — heuristics cover most cases.
    enable_l1_classifier: bool = False
    head_path: Path = Path("var/head.json")

    # Semantic cache (L2): requires embeddings (model2vec). The threshold
    # is tuned on your own logs, not borrowed from a paper.
    enable_semantic_cache: bool = True
    semantic_threshold: float = 0.95

    # Exploration: duplicate 5-10% of requests to a 2nd random candidate in background
    # to avoid training bias on own policy.
    enable_exploration: bool = True
    explore_rate: float = 0.10
    explore_floor: float = 0.02

    # Bandit scoring: online quality learning per (task × model) pair.
    enable_bandit: bool = True

    # PII detector: detected personal data forces privacy: strict.
    enable_pii: bool = True

    # Cascade verifier: a failed lower-tier answer → escalation.
    enable_verifier: bool = True

    # Adaptive quota threshold (Phase K): dynamic local vs cloud threshold based on remaining quota
    enable_adaptive_threshold: bool = True

    # Ensemble & Self-Consistency (Phase L):
    enable_ensemble: bool = True
    self_consistency_samples: int = 3
    fanout_timeout_ms: int = 4000
    ensemble_risk_threshold: float = 0.70

    # Auto-discovery of free models at startup (control plane).
    enable_discovery: bool = True

    # Free models only. No paid providers in the manifest, but the flag
    # stays as a line of defense in case a free tier stops being free
    # (see waypost/pricing.py).
    free_only: bool = True

    # Terminal log level: DEBUG/INFO/WARNING/ERROR.
    log_level: str = "INFO"

    # ---- guard layer ------------------------------------------------
    # Injection detection in untrusted blocks (tool outputs, RAG chunks).
    # scope=all makes sense when the router is not for you alone.
    enable_guard: bool = True
    guard_scope: str = "untrusted"  # untrusted | all
    guard_action: str = "annotate"  # annotate | block
    guard_neutralize: bool = True

    # ---- embeddings -------------------------------------------------
    # Local, so /v1/embeddings does not burn provider quota.
    embed_model: str = "minishlab/potion-multilingual-128M"
    embed_micro_batch_ms: float = 25.0
    embed_max_batch: int = 64

    # ---- semantic cache --------------------------------------------
    # The cross-encoder confirms a bi-encoder hit. With it the threshold
    # can be lowered: the verifier absorbs the false-hit risk.
    enable_semantic_rerank: bool = False
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_threshold: float = 0.5
    # Last step of the ladder: an approximate answer from the cache when
    # all providers refused. Always marked approximate.
    enable_degraded_cache: bool = True

    # ---- execution -------------------------------------------------
    enable_hedging: bool = True
    hedge_budget: float = 0.05  # share of requests, no more
    idempotency_ttl_s: int = 86_400

    # ---- batch API --------------------------------------------------
    enable_batch: bool = True
    batch_concurrency: int = 2
    batch_window_h: float = 24.0

    # ---- context compression ---------------------------------------
    # off — until you have your own measurements, this is the right mode.
    compress_mode: str = "off"  # off | safe | llmlingua
    compress_min_tokens: int = 3000
    compress_rate: float = 0.6

    # ---- control plane ----------------------------------------------
    enable_probe: bool = False  # probing spends quota
    probe_interval_h: float = 24.0
    discovery_interval_h: float = 12.0
    health_interval_min: float = 5.0
    purge_interval_h: float = 6.0

    # ---- observability ----------------------------------------------
    enable_metrics: bool = True
    # Log request texts. Needed to train the head, cluster traffic and
    # for offline reward — and for the same reason off by default: these
    # are the most sensitive data in the system and must be stored
    # deliberately.
    enable_prompt_log: bool = False
