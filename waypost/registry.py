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

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from . import keychain, pricing
from .schemas import Capability, Tier

log = logging.getLogger("waypost.registry")


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

    # Lifecycle. An offering is not simply present or absent: a model that
    # answered 403 once may be back in an hour, and a model discovery
    # invented five minutes ago has not earned production traffic yet.
    #   active     — in the routing pool
    #   candidate  — just discovered, not probed, must not take traffic
    #   shadow     — probed and plausible, allowed a small share
    #   quarantine — failed twice / vanished from /models, retried daily
    lifecycle: str = "active"
    # How many consecutive discovery cycles this model was missing from the
    # provider's /models. Three (~36 h) means retired, one means a blip.
    miss_streak: int = 0
    # Consecutive dead probes. One is not evidence — proxies return 404 and
    # gateways return 403 for reasons that pass.
    dead_streak: int = 0
    quarantined_at: float = 0.0

    # A manifest entry says that a local model may exist; it does not prove
    # that its process is listening right now.  Production manifest entries
    # start unavailable and discovery flips this only after GET /models
    # succeeds and returns the configured model.  Directly constructed
    # offerings keep the True default for tests and embedded users.
    runtime_available: bool = True

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
        # candidate: discovered but never measured. quarantine: proven
        # broken twice. Neither may serve a user request; both are still
        # in the registry so the control plane can re-check them.
        if self.lifecycle in ("candidate", "quarantine"):
            return False
        if self.is_local and not self.runtime_available:
            return False
        if self.is_local or not self.api_key_env:
            return True
        return bool(self.api_key)


def is_cloud_ollama_model(model_id: str) -> bool:
    """Checks if an Ollama model has a 'cloud' prefix (cloud model) or is local."""
    low = model_id.lower().strip()
    return (
        low.startswith("cloud/")
        or low.startswith("cloud:")
        or low.startswith("cloud-")
        or low.startswith("cloud_")
        or low.startswith("cloud.")
        or low == "cloud"
        or "/cloud/" in low
        or ":cloud/" in low
    )


_CAP_MAP = {c.value: c for c in Capability}


def _parse_offering(provider: dict, model: dict) -> Offering:
    caps = {_CAP_MAP[c] for c in model.get("caps", []) if c in _CAP_MAP}
    limits = {**provider.get("limits", {}), **model.get("limits", {})}
    declared = model.get("free", provider.get("free"))
    verdict = pricing.classify(model["id"], declared=declared)

    prov_name = provider.get("name", "").lower()
    model_id = model["id"]
    is_local = provider.get("is_local", False)

    # For Ollama / local provider: models with 'cloud' prefix or ollama.com base_url
    # are cloud models; all other local engine models are local.
    if prov_name in ("ollama", "local"):
        if (
            is_cloud_ollama_model(model_id)
            or "ollama.com" in provider.get("base_url", "").lower()
        ):
            is_local = False
        else:
            is_local = True

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
        is_local=is_local,
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
        runtime_available=model.get("runtime_available", not is_local),
        prompt_cache=model.get("prompt_cache", provider.get("prompt_cache", "auto")),
        concurrency=model.get("concurrency", provider.get("concurrency")),
    )


# Runtime state an offering earns by serving traffic. A rebuild of the
# registry from the manifest must carry it over, or every hot reload
# would throw away everything the control plane learned and start the
# pool cold — measured TTFT, reliability, lifecycle, the record of which
# models turned out to be paid.
LEARNED_FIELDS = (
    "ttft_p50_ms",
    "success_rate",
    "quality_score",
    "limit_rpm",
    "limit_rpd",
    "limit_tpm",
    "lifecycle",
    "miss_streak",
    "dead_streak",
    "quarantined_at",
    "runtime_available",
    "enabled",
    "weight",
    "caps",
)


class Registry:
    def __init__(self, offerings: list[Offering] | None = None):
        self._offerings: dict[str, Offering] = {o.key: o for o in (offerings or [])}
        # Bumped on every hot swap. Returned in the router block of each
        # response, so a routing decision can be tied to the pool that
        # produced it.
        self.version: int = 1

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

    def update_local_runtime(
        self,
        provider: str,
        base_url: str,
        model_ids: set[str] | None,
    ) -> int:
        """Apply a live local-runtime check to matching offerings.

        ``model_ids=None`` means the endpoint itself was unreachable.  An
        empty set means it answered but did not expose a configured chat
        model.  Both states must remove the offering from the routing pool.
        """
        normalized_url = base_url.rstrip("/")
        updated = 0
        for o in self._offerings.values():
            if (
                not o.is_local
                or o.provider != provider
                or o.base_url.rstrip("/") != normalized_url
            ):
                continue
            available = model_ids is not None and o.model_id in model_ids
            if o.runtime_available != available:
                o.runtime_available = available
                updated += 1
        return updated

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

    def transition(self, key: str, to: str, reason: str = "") -> bool:
        """Move an offering between lifecycle states. Returns True on a
        real change, so callers can log only what happened."""
        o = self._offerings.get(key)
        if o is None or o.lifecycle == to:
            return False
        was = o.lifecycle
        o.lifecycle = to
        if to == "quarantine":
            o.quarantined_at = time.time()
        elif to in ("active", "shadow"):
            o.quarantined_at = 0.0
            o.dead_streak = 0
            o.miss_streak = 0
        # Weight follows the state: shadow gets a small share of traffic,
        # not a coin flip against a proven model.
        if to == "shadow":
            o.weight = min(o.weight, 0.2)
        elif to == "active" and was == "shadow":
            o.weight = 0.8
        log.info("lifecycle %s: %s → %s%s", key, was, to, f" ({reason})" if reason else "")
        return True

    def by_lifecycle(self, *states: str) -> list["Offering"]:
        return [o for o in self._offerings.values() if o.lifecycle in states]

    def quarantined(self, older_than_s: float = 86_400) -> list["Offering"]:
        """Quarantined long enough to be worth one resurrection probe.

        A model marked dead used to stay dead forever: the probe job
        targets usable() offerings, and a disabled one is not usable, so
        nothing ever re-checked it.
        """
        cutoff = time.time() - older_than_s
        return [
            o
            for o in self._offerings.values()
            if o.lifecycle == "quarantine"
            and o.quarantined_at
            and o.quarantined_at <= cutoff
            # A model proven to bill is never resurrected: a billing
            # record outranks any later claim of being free.
            and not (not o.free and o.free_source == pricing.Source.BILLED.value)
        ]

    # ------------------------------------------------------------- swap
    def offerings_map(self) -> dict[str, Offering]:
        """The live mapping. Callers must treat it as read-only."""
        return self._offerings

    def carry_over(self, fresh: list[Offering]) -> list[Offering]:
        """Copy learned runtime state from the live pool onto a rebuilt one."""
        for o in fresh:
            old = self._offerings.get(o.key)
            if old is None:
                continue
            for field in LEARNED_FIELDS:
                value = getattr(old, field, None)
                if value is not None:
                    setattr(o, field, value)
            # A billing proof outranks any later manifest claim.
            if not old.free and old.free_source == pricing.Source.BILLED.value:
                o.free = False
                o.free_source = old.free_source
                o.free_detail = old.free_detail
        return fresh

    def swap(self, fresh: list[Offering]) -> int:
        """Replace the whole pool by rebinding one reference.

        Requests already in flight hold their Offering objects directly
        (the plan is a list of them), so they finish against the pool
        they were planned on. Readers that go through the registry see
        either the old mapping or the new one, never a half-built one —
        which a piecemeal add/remove could not promise.
        """
        self._offerings = {o.key: o for o in fresh}
        self.version += 1
        return self.version

    def apply_probe(self, key: str, **measured) -> None:
        """Probe results flow into the registry without a reload."""
        if (o := self._offerings.get(key)) is None:
            return
        for k, v in measured.items():
            if hasattr(o, k) and v is not None:
                setattr(o, k, v)
