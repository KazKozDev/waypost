"""Probing (registry source C).

Provider docs say a model "supports tool calling" and do not say how long
it really waits for the first token or what happens if you ask for JSON.
The gap between declared and working is exactly what probe exists for.

Measured:
  TTFT — on a canary request, an honest measurement of the first stream chunk;
  real limits — from x-ratelimit-* headers, not from README;
  tool calling and JSON — by attempting, not by reading the spec;
  status — healthy / degraded / dead.

Results are written to the probes table and merged into the registry at
server startup, overriding the manifest: measurement beats declaration.
"""
from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

import httpx

from .providers.openai_compat import OpenAICompatAdapter, ProviderError
from .registry import Offering, Registry
from .schemas import Capability, ChatMessage, ChatRequest

log = logging.getLogger("waypost.probe")

CANARY = ChatRequest(
    messages=[ChatMessage(role="user", content="Reply with one word: test")],
    max_tokens=16,
    temperature=0.0,
)

TOOL_CANARY = ChatRequest(
    messages=[ChatMessage(role="user", content="What is the weather in Barcelona?")],
    max_tokens=64,
    temperature=0.0,
    tools=[
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Weather in a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ],
)

JSON_CANARY = ChatRequest(
    messages=[
        ChatMessage(role="user", content='Return JSON {"ok": true} and nothing else')
    ],
    max_tokens=32,
    temperature=0.0,
    response_format={"type": "json_object"},
)

RATE_HEADERS = {
    "x-ratelimit-limit-requests": "limit_rpm",
    "x-ratelimit-remaining-requests": "remaining_requests",
    "x-ratelimit-limit-tokens": "limit_tpm",
    "x-ratelimit-remaining-tokens": "remaining_tokens",
    "x-ratelimit-limit-requests-day": "limit_rpd",
}


def parse_rate_limits(headers: httpx.Headers | dict) -> dict[str, int]:
    """Real limits arrive in headers, not in documentation."""
    out: dict[str, int] = {}
    for header, field in RATE_HEADERS.items():
        value = headers.get(header)
        if value is None:
            continue
        try:
            out[field] = int(float(str(value).rstrip("s")))
        except ValueError:
            continue
    return out


async def measure_ttft(
    adapter: OpenAICompatAdapter, o: Offering, timeout: float = 60.0
) -> float | None:
    """TTFT is measured on a stream: a non-stream response has no first
    token, only the full generation time — that measures something else."""
    t0 = time.perf_counter()
    try:
        async for chunk in adapter.stream(o, CANARY, timeout):
            if chunk.strip():
                return round((time.perf_counter() - t0) * 1000, 1)
    except ProviderError:
        return None
    except Exception:  # noqa: BLE001
        return None
    return None


async def probe_one(
    adapter: OpenAICompatAdapter, o: Offering, timeout: float = 60.0
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "key": o.key,
        "provider": o.provider,
        "model_id": o.model_id,
        "tier": o.tier.value,
        "free": o.free,
        "is_local": o.is_local,
        "caps": [c.value for c in o.caps],
        "ts": time.time(),
        "status": "healthy",
    }
    if not o.is_local and o.api_key_env and not o.api_key:
        result["status"] = "no_key"
        result["error"] = f"API key '{o.api_key_env}' is not configured"
        return result

    t0 = time.perf_counter()
    try:
        body = await adapter.complete(o, CANARY, timeout)
        result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        result["sample"] = json.dumps(body.get("choices", [])[:1], ensure_ascii=False)[
            :300
        ]
    except ProviderError as exc:
        # 401/404/403/402 — offering is dead (no key, wrong key, model gone, unpaid).
        # Everything else is temporary degradation.
        status_code = exc.status
        result["status"] = "dead" if status_code in (401, 402, 403, 404) else "degraded"
        result["http_status"] = status_code
        result["error"] = str(exc)[:300]
        return result
    except Exception as exc:  # noqa: BLE001
        result["status"] = "degraded"
        result["error"] = f"{type(exc).__name__}: {exc}"[:300]
        return result

    if (ttft := await measure_ttft(adapter, o, timeout)) is not None:
        result["ttft_p50_ms"] = ttft
    else:
        # Stream not supported or silent — take the full time as an upper
        # bound, noting that this is not a real TTFT.
        result["ttft_p50_ms"] = result.get("latency_ms")
        result["ttft_estimated"] = True

    for name, canary in (
        ("supports_tools", TOOL_CANARY),
        ("supports_json", JSON_CANARY),
    ):
        try:
            await adapter.complete(o, canary, timeout)
            result[name] = True
        except Exception:  # noqa: BLE001
            result[name] = False  # declared ≠ working
    return result


async def probe_all(
    adapter: OpenAICompatAdapter,
    offerings: list[Offering],
    *,
    concurrency: int = 4,
    timeout: float = 60.0,
) -> list[dict[str, Any]]:
    sem = asyncio.Semaphore(concurrency)

    async def one(o: Offering) -> dict[str, Any]:
        async with sem:
            return await probe_one(adapter, o, timeout)

    return list(await asyncio.gather(*(one(o) for o in offerings)))


def summarize_by_tier(
    probes: list[dict[str, Any]] | dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Aggregate probe metrics grouped by model tier (S / M / L)."""
    items = list(probes.values()) if isinstance(probes, dict) else probes
    summary: dict[str, Any] = {
        "total": len(items),
        "healthy": sum(1 for r in items if r.get("status") == "healthy"),
        "degraded": sum(1 for r in items if r.get("status") == "degraded"),
        "dead": sum(1 for r in items if r.get("status") == "dead"),
        "no_key": sum(1 for r in items if r.get("status") == "no_key"),
        "by_tier": {},
    }
    for tier in ("S", "M", "L"):
        tier_items = [r for r in items if str(r.get("tier", "")).upper() == tier]
        ttfts = [
            r["ttft_p50_ms"]
            for r in tier_items
            if r.get("ttft_p50_ms") and r.get("status") == "healthy"
        ]
        summary["by_tier"][tier] = {
            "total": len(tier_items),
            "healthy": sum(1 for r in tier_items if r.get("status") == "healthy"),
            "degraded": sum(1 for r in tier_items if r.get("status") == "degraded"),
            "dead": sum(1 for r in tier_items if r.get("status") == "dead"),
            "no_key": sum(1 for r in tier_items if r.get("status") == "no_key"),
            "avg_ttft_ms": round(sum(ttfts) / len(ttfts), 1) if ttfts else None,
        }
    return summary


# ------------------------------------------------------------- storage


def init_db(db_path: str | Path) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as c:
        c.execute(
            """CREATE TABLE IF NOT EXISTS probes (
            ts REAL, key TEXT, status TEXT, ttft_p50_ms REAL,
            supports_tools INTEGER, supports_json INTEGER, payload TEXT)"""
        )


def store(db_path: str | Path, results: list[dict[str, Any]]) -> None:
    init_db(db_path)
    with sqlite3.connect(db_path) as c:
        for r in results:
            c.execute(
                "INSERT INTO probes VALUES (?,?,?,?,?,?,?)",
                (
                    r["ts"],
                    r["key"],
                    r["status"],
                    r.get("ttft_p50_ms"),
                    int(bool(r.get("supports_tools"))),
                    int(bool(r.get("supports_json"))),
                    json.dumps(r, ensure_ascii=False),
                ),
            )


def latest(
    db_path: str | Path, max_age_s: float = 7 * 86_400
) -> dict[str, dict[str, Any]]:
    """The latest measurement per offering. Older than max_age_s is not
    used: free-tier limits change more often than the docs do."""
    path = Path(db_path)
    if not path.exists():
        return {}
    since = time.time() - max_age_s
    try:
        with sqlite3.connect(path) as c:
            rows = c.execute(
                "SELECT key, payload, MAX(ts) FROM probes WHERE ts > ? " "GROUP BY key",
                (since,),
            ).fetchall()
    except sqlite3.Error:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for key, payload, _ in rows:
        try:
            out[key] = json.loads(payload)
        except (json.JSONDecodeError, TypeError):
            continue
    return out


# Two consecutive dead probes before an offering leaves the pool. One is
# not evidence: gateways return 403 during a rotation, proxies return 404
# for a model that is merely cold, and a single bad minute used to remove
# a working model permanently.
DEAD_STREAK_LIMIT = 2


def apply_to_registry(registry: Registry, probes: dict[str, dict[str, Any]]) -> int:
    """A measurement overrides the manifest — that is the whole point of
    probing.

    Death is a streak, not an event, and it leads to quarantine rather
    than to ``enabled = False``. The old flag was a one-way door: the
    probe job targets usable() offerings, a disabled one is not usable,
    so nothing ever re-checked it and a transient 403 removed the model
    for the life of the process.
    """
    applied = 0
    for key, r in probes.items():
        o = registry.get(key)
        if o is None:
            continue
        measured: dict[str, Any] = {}
        if r.get("ttft_p50_ms"):
            measured["ttft_p50_ms"] = float(r["ttft_p50_ms"])
        for field in ("limit_rpm", "limit_rpd", "limit_tpm"):
            if r.get(field):
                measured[field] = int(r[field])
        # Declared capabilities are not working capabilities. The canary
        # is the only thing that knows the difference.
        if r.get("supports_tools") is False and Capability.TOOLS in o.caps:
            o.caps = o.caps - {Capability.TOOLS}
            applied += 1
        if r.get("supports_json") is False and Capability.JSON in o.caps:
            o.caps = o.caps - {Capability.JSON}
            applied += 1

        status = r.get("status")
        if status == "dead":
            o.dead_streak += 1
            if o.dead_streak >= DEAD_STREAK_LIMIT:
                registry.transition(
                    key, "quarantine", f"dead probe x{o.dead_streak}"
                )
        elif status == "healthy":
            o.dead_streak = 0
            if o.lifecycle in ("candidate", "shadow", "quarantine"):
                # Passed a live probe: candidates graduate to shadow and
                # earn full traffic later, quarantined models come back.
                registry.transition(
                    key,
                    "active" if o.lifecycle == "shadow" else "shadow",
                    "probe healthy",
                )
        if measured:
            registry.apply_probe(key, **measured)
            applied += 1
    return applied
