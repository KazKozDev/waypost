"""Registry of model offerings.

Sources, as in the spec:
  A. Static YAML manifest — what cannot be learned programmatically
     (free-tier presence, data policy, license).
  B. Provider GET /models — reconciliation with the manifest
     (scripts/discover.py).
  C. Empirical probing — measured_* fields (scripts/probe.py).

Here only A + reading B/C results. The registry lives fully in memory:
it is small, and the hot path must not hit the database.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from . import keychain, pricing
from .schemas import Capability, Tier


@dataclass
class Offering:
    """One registry row: a specific model at a specific provider."""

    provider: str
    model_id: str
    base_url: str
    api_key_env: str | None = None
    # Additional keys for the same provider. The second step of the
    # degradation ladder — "another key" — is cheaper than switching
    # providers: the model is the same, the warmed prefix is not lost.
    api_key_envs: list[str] = field(default_factory=list)

    tier: Tier = Tier.M
    ctx_window: int = 8192
    max_output: int = 4096
    caps: set[Capability] = field(default_factory=set)

    # Data policy. trains_on_data=True is the price of a free tier.
    trains_on_data: bool = True
    is_local: bool = False

    # Is the model free? Paid models are filtered out by free_only so the
    # router does not spend money (and the manifest has none anyway).
    # The value is not taken on faith: waypost/pricing.py sets it from
    # the manifest, prices from /models, and the actual bill in the
    # response (fail-closed — "unknown" means "paid"). free_source stores
    # how it was proven.
    free: bool = True
    free_source: str = pricing.Source.MANIFEST.value
    free_detail: str = ""

    # Limits. None = unlimited (a local model).
    limit_rpm: int | None = None
    limit_rpd: int | None = None
    limit_tpm: int | None = None

    # Measured by the probe job, not taken from documentation.
    quality_score: float = 0.5
    ttft_p50_ms: float = 1500.0
    success_rate: float = 1.0

    # Quality per task class (code/reasoning/extraction/chat/creative).
    # Lets you prefer a code model for code, a fast one for chat.
    # Empty → quality_score is used.
    quality: dict[str, float] = field(default_factory=dict)

    def quality_for(self, task_class: str) -> float:
        if task_class in self.quality:
            return self.quality[task_class]
        return self.quality_score

    weight: float = 1.0  # manual preference multiplier
    enabled: bool = True

    # Provider prefix cache: "auto" — kicks in from ~1024 prefix tokens
    # (OpenAI, DeepSeek), "explicit" — requires block markers (Anthropic
    # cache_control), "none" — none at all.
    prompt_cache: str = "auto"

    # Parallel request cap. For a local model this is not tuning but a
    # necessity: mlx-lm holds weights in shared memory, and two heavy
    # generations at once contend for memory, not speed.
    concurrency: int | None = None

    def set_pricing(self, verdict: "pricing.Verdict") -> bool:
        """Apply a paidness verdict. True if the status changed.

        Demotion to paid is irreversible within the process lifetime: a
        billing proof is stronger than any later declaration.
        """
        was_free, was_source = self.free, self.free_source
        if not self.free and self.free_source == pricing.Source.BILLED.value:
            return False
        self.free = verdict.is_free
        self.free_source = verdict.source.value
        self.free_detail = verdict.detail
        return (was_free, was_source) != (self.free, self.free_source)

    @property
    def key(self) -> str:
        return f"{self.provider}/{self.model_id}"

    @property
    def key_names(self) -> list[str]:
        """Names of variables where a key may live. Beyond the explicit
        ones — numbered variants NAME_2, NAME_3: a convenient way to add
        a second free account without touching the manifest."""
        names: list[str] = []
        if self.api_key_env:
            names.append(self.api_key_env)
            names += [f"{self.api_key_env}_{i}" for i in range(2, 6)]
        names += [n for n in self.api_key_envs if n not in names]
        return names

    @property
    def api_keys(self) -> list[str]:
        """All available keys in preference order. Duplicates removed:
        the same string in two variables is one key and shares one
        quota."""
        out: list[str] = []
        for name in self.key_names:
            value = os.environ.get(name) or keychain.get(name)
            if value and value not in out:
                out.append(value)
        return out

    @property
    def api_key(self) -> str | None:
        keys = self.api_keys
        return keys[0] if keys else None

    def api_key_at(self, index: int) -> str | None:
        keys = self.api_keys
        return keys[index] if 0 <= index < len(keys) else None

    @property
    def key_count(self) -> int:
        """How many independent quotas this offering has. A local model
        and a provider without auth — one "quota" with no key."""
        if self.is_local or not self.api_key_env:
            return 1
        return max(1, len(self.api_keys))

    @property
    def usable(self) -> bool:
        """If the provider declared api_key_env but the variable is
        missing, the offering is not used. Silently getting a 401 on
        every request is worse than not having the candidate at all."""
        if not self.enabled:
            return False
        if self.is_local or not self.api_key_env:
            return True
        return bool(self.api_key)


_CAP_MAP = {c.value: c for c in Capability}


def _parse_offering(provider: dict, model: dict) -> Offering:
    caps = {_CAP_MAP[c] for c in model.get("caps", []) if c in _CAP_MAP}
    limits = {**provider.get("limits", {}), **model.get("limits", {})}
    declared = model.get("free", provider.get("free"))
    verdict = pricing.classify(model["id"], declared=declared)
    return Offering(
        provider=provider["name"],
        model_id=model["id"],
        base_url=provider["base_url"],
        api_key_env=provider.get("api_key_env"),
        api_key_envs=list(provider.get("api_key_envs", [])),
        tier=Tier(model.get("tier", "M")),
        ctx_window=model.get("ctx_window", 8192),
        max_output=model.get("max_output", 4096),
        caps=caps,
        trains_on_data=provider.get("trains_on_data", True),
        is_local=provider.get("is_local", False),
        free=verdict.is_free,
        free_source=verdict.source.value,
        free_detail=verdict.detail,
        limit_rpm=limits.get("rpm"),
        limit_rpd=limits.get("rpd"),
        limit_tpm=limits.get("tpm"),
        quality_score=model.get("quality_score", 0.5),
        quality=model.get("quality", {}),
        ttft_p50_ms=model.get("ttft_p50_ms", 1500.0),
        weight=model.get("weight", 1.0),
        enabled=model.get("enabled", True),
        prompt_cache=model.get("prompt_cache", provider.get("prompt_cache", "auto")),
        concurrency=model.get("concurrency", provider.get("concurrency")),
    )


class Registry:
    def __init__(self, offerings: list[Offering] | None = None):
        self._offerings: dict[str, Offering] = {o.key: o for o in (offerings or [])}

    @classmethod
    def from_manifest(cls, path: str | Path) -> "Registry":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        offerings = [
            _parse_offering(p, m)
            for p in data.get("providers", [])
            for m in p.get("models", [])
        ]
        return cls(offerings)

    def all(self) -> list[Offering]:
        return list(self._offerings.values())

    def usable(self) -> list[Offering]:
        return [o for o in self._offerings.values() if o.usable]

    def get(self, key: str) -> Offering | None:
        return self._offerings.get(key)

    def add(self, o: Offering) -> bool:
        """Add an offering if it is not there yet. Returns True if a new
        one was added (used by auto-discovery)."""
        if o.key in self._offerings:
            return False
        self._offerings[o.key] = o
        return True

    def find_by_model_id(self, model_id: str) -> Offering | None:
        """For an explicit request of a specific model bypassing the
        router."""
        if "/" in model_id and (o := self._offerings.get(model_id)):
            return o
        for o in self._offerings.values():
            if o.model_id == model_id:
                return o
        if "/" in model_id:
            prov, _, m_name = model_id.partition("/")
            for o in self._offerings.values():
                if (
                    o.provider == prov
                    or (
                        o.is_local
                        and prov in ("local", "ollama", "lmstudio", "llamacpp", "mlx")
                    )
                ) and o.model_id == m_name:
                    return o
        # If user passed a provider name like 'local' or 'ollama', pick the first usable
        prov_offerings = self.find_by_provider(model_id)
        if prov_offerings:
            return prov_offerings[0]
        return None

    def find_by_provider(self, provider: str) -> list[Offering]:
        """Find all usable offerings belonging to a provider or local group."""
        low = provider.lower()
        if low == "local":
            return [o for o in self.usable() if o.is_local]
        return [o for o in self.usable() if o.provider.lower() == low]

    def apply_probe(self, key: str, **measured) -> None:
        """Probe results flow into the registry without a reload."""
        if (o := self._offerings.get(key)) is None:
            return
        for k, v in measured.items():
            if hasattr(o, k) and v is not None:
                setattr(o, k, v)
