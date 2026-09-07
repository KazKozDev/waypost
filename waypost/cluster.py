"""Shared state for more than one replica.

Three objects in this router are counters that only mean anything if
every replica agrees on them:

  Ledger   — the free-tier quota. Two replicas each tracking "20 of 30
             rpm used" will happily send 40.
  Breaker  — provider health. A dead provider discovered by one replica
             is discovered again, at the cost of real requests, by every
             other one.
  Governor — the learned rate limit. Same argument.

Everything else (bandit posteriors, latency EMAs, the registry) tolerates
being per-replica: it converges, and disagreement costs a little routing
quality rather than a burst of 429s.

Until now ``InstanceLock`` made the question moot by refusing to start a
second process. That is the honest default for a single-node router, but
it is a wall, not an answer. This module is the answer: point
``ROUTER_REDIS_URL`` at a Redis and the three counters move there, with
the same interfaces the rest of the code already calls.

Two design points.

*Atomicity lives in Lua, not in Python.* A read-modify-write across the
network is exactly the race the shared counter was meant to remove.
Every mutation here is one server-side script, so "check the remaining
quota and take a slot" cannot interleave between replicas.

*Failing open, not closed.* When Redis is unreachable the router falls
back to its in-process state and logs it. A quota counter that cannot be
read is a reason to be slightly over budget, never a reason to stop
serving traffic.
"""
from __future__ import annotations

import logging
import time
from typing import Any

log = logging.getLogger("waypost.cluster")

# Take a slot from every bucket of one key, atomically, and only if all of
# them have room. Returns 1 on success, 0 when any bucket refuses — so a
# replica can never observe "rpm had room" and act on it after another
# replica consumed the last slot.
#
# KEYS[1] = hash of the key's buckets   ARGV = now, [name, limit, window, cost]...
RESERVE_LUA = """
local now = tonumber(ARGV[1])
local blocked = tonumber(redis.call('HGET', KEYS[1], 'blocked_until') or '0')
if now < blocked then return 0 end
local n = (#ARGV - 1) / 4
for i = 0, n - 1 do
  local name   = ARGV[2 + i*4]
  local limit  = tonumber(ARGV[3 + i*4])
  local window = tonumber(ARGV[4 + i*4])
  local cost   = tonumber(ARGV[5 + i*4])
  local used   = tonumber(redis.call('HGET', KEYS[1], name .. ':used') or '0')
  local start  = tonumber(redis.call('HGET', KEYS[1], name .. ':start') or '0')
  if now - start >= window then used = 0 end
  if used + cost > limit then return 0 end
end
for i = 0, n - 1 do
  local name   = ARGV[2 + i*4]
  local window = tonumber(ARGV[4 + i*4])
  local cost   = tonumber(ARGV[5 + i*4])
  local used   = tonumber(redis.call('HGET', KEYS[1], name .. ':used') or '0')
  local start  = tonumber(redis.call('HGET', KEYS[1], name .. ':start') or '0')
  if now - start >= window then
    used = 0
    redis.call('HSET', KEYS[1], name .. ':start', now)
  end
  redis.call('HSET', KEYS[1], name .. ':used', used + cost)
end
redis.call('EXPIRE', KEYS[1], 172800)
return 1
"""

# Correct a bucket after the answer: the estimate always lies.
ADJUST_LUA = """
local used = tonumber(redis.call('HGET', KEYS[1], ARGV[1] .. ':used') or '0')
local delta = tonumber(ARGV[2])
local fresh = used + delta
if fresh < 0 then fresh = 0 end
redis.call('HSET', KEYS[1], ARGV[1] .. ':used', fresh)
return fresh
"""

# One failure, evaluated server-side, so the threshold is counted once
# across replicas rather than once per replica.
FAILURE_LUA = """
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local threshold = tonumber(ARGV[3])
local base = tonumber(ARGV[4])
local maxcd = tonumber(ARGV[5])
local probe = tonumber(redis.call('HGET', KEYS[1], 'probe') or '0')
if probe > 0 then
  local cd = tonumber(redis.call('HGET', KEYS[1], 'cooldown') or base)
  cd = math.min(maxcd, cd * 2)
  redis.call('HSET', KEYS[1], 'probe', 0, 'opened_at', now, 'cooldown', cd)
  redis.call('DEL', KEYS[2])
  redis.call('EXPIRE', KEYS[1], 86400)
  return cd
end
redis.call('ZADD', KEYS[2], now, now .. ':' .. math.random())
redis.call('ZREMRANGEBYSCORE', KEYS[2], '-inf', now - window)
redis.call('EXPIRE', KEYS[2], math.ceil(window) + 60)
if redis.call('ZCARD', KEYS[2]) >= threshold then
  -- The cooldown must be written, not merely defaulted on read: a reader
  -- that falls back to its own default would disagree with the replica
  -- that opened the circuit about when it may be probed again.
  local cd = tonumber(redis.call('HGET', KEYS[1], 'cooldown') or base)
  redis.call('HSET', KEYS[1], 'opened_at', now, 'probe', 0, 'cooldown', cd)
  redis.call('DEL', KEYS[2])
  redis.call('EXPIRE', KEYS[1], 86400)
  return cd
end
return 0
"""

# Take the single half-open probe token — across the whole cluster, so
# one replica probes a recovering provider and the others wait.
#
# The token comes back as a STRING. Redis converts a Lua number reply to
# an integer, which would truncate the timestamp and leave release_probe
# comparing 1757280000 against the stored 1757280000.1234 — the token
# would never match, never be released, and the cluster would lose the
# provider exactly the way the single-process bug did.
PROBE_LUA = """
local now = tonumber(ARGV[1])
local ttl = tonumber(ARGV[2])
local opened = tonumber(redis.call('HGET', KEYS[1], 'opened_at') or '0')
if opened == 0 then return 'closed' end
local cd = tonumber(redis.call('HGET', KEYS[1], 'cooldown') or ARGV[3])
if now - opened < cd then return 'busy' end
local probe = tonumber(redis.call('HGET', KEYS[1], 'probe') or '0')
if probe > 0 and now - probe < ttl then return 'busy' end
local token = ARGV[1]
redis.call('HSET', KEYS[1], 'probe', token)
redis.call('EXPIRE', KEYS[1], 86400)
return token
"""

PROBE_CLOSED = "closed"
PROBE_BUSY = "busy"


def connect(url: str, *, timeout_s: float = 0.25) -> Any | None:
    """Open a Redis connection, or return None and say why.

    The timeouts are deliberately short: this sits on the request path,
    and a Redis that needs three seconds to answer has already cost more
    than the quota mistake it was preventing.
    """
    try:
        import redis  # noqa: PLC0415  — optional dependency
    except ImportError:
        log.error(
            "ROUTER_REDIS_URL is set but redis-py is missing "
            "(pip install 'waypost[cluster]'); staying single-node"
        )
        return None
    try:
        client = redis.Redis.from_url(
            url,
            socket_timeout=timeout_s,
            socket_connect_timeout=timeout_s,
            decode_responses=True,
        )
        client.ping()
    except Exception as exc:  # noqa: BLE001 — any failure means single-node
        log.error("redis at %s unreachable (%s); staying single-node", url, exc)
        return None
    log.info("cluster state: redis at %s", url)
    return client


class SharedState:
    """The Redis half of the shared counters.

    Every method returns None when Redis fails, which the callers read as
    "decide locally": the fallback is a slightly stale counter, not an
    outage.
    """

    def __init__(self, client: Any, namespace: str = "waypost"):
        self.client = client
        self.ns = namespace
        self._scripts: dict[str, Any] = {}
        self.failures = 0
        self.last_error: str | None = None

    def _script(self, name: str, body: str) -> Any:
        script = self._scripts.get(name)
        if script is None:
            script = self.client.register_script(body)
            self._scripts[name] = script
        return script

    def _run(self, name: str, body: str, keys: list[str], args: list[Any]) -> Any:
        try:
            return self._script(name, body)(keys=keys, args=args)
        except Exception as exc:  # noqa: BLE001
            self.failures += 1
            self.last_error = f"{type(exc).__name__}: {exc}"[:200]
            log.warning("redis %s failed (%s); deciding locally", name, exc)
            return None

    # ------------------------------------------------------------ quota
    def _quota_key(self, bucket_key: str) -> str:
        return f"{self.ns}:quota:{bucket_key}"

    def reserve(self, bucket_key: str, specs: list[tuple[str, int, int, int]]):
        """specs: (bucket name, limit, window seconds, cost)."""
        args: list[Any] = [time.time()]
        for name, limit, window, cost in specs:
            args += [name, limit, window, cost]
        got = self._run("reserve", RESERVE_LUA, [self._quota_key(bucket_key)], args)
        return None if got is None else bool(got)

    def adjust(self, bucket_key: str, bucket: str, delta: int) -> int | None:
        return self._run(
            "adjust", ADJUST_LUA, [self._quota_key(bucket_key)], [bucket, delta]
        )

    def block(self, bucket_key: str, until: float) -> None:
        try:
            key = self._quota_key(bucket_key)
            with self.client.pipeline() as pipe:
                pipe.hset(key, "blocked_until", until)
                pipe.expire(key, 172800)
                pipe.execute()
        except Exception as exc:  # noqa: BLE001
            self.failures += 1
            self.last_error = str(exc)[:200]

    def usage(self, bucket_key: str) -> dict[str, float] | None:
        try:
            raw = self.client.hgetall(self._quota_key(bucket_key))
        except Exception:  # noqa: BLE001
            self.failures += 1
            return None
        return {k: float(v) for k, v in raw.items()} if raw else {}

    # ---------------------------------------------------------- breaker
    def _breaker_keys(self, provider: str) -> tuple[str, str]:
        return (
            f"{self.ns}:breaker:{provider}",
            f"{self.ns}:breaker:{provider}:fails",
        )

    def on_failure(
        self,
        provider: str,
        *,
        window_s: float,
        threshold: int,
        base_cooldown_s: float,
        max_cooldown_s: float,
    ) -> float | None:
        state_key, fails_key = self._breaker_keys(provider)
        return self._run(
            "failure",
            FAILURE_LUA,
            [state_key, fails_key],
            [time.time(), window_s, threshold, base_cooldown_s, max_cooldown_s],
        )

    def on_success(self, provider: str) -> None:
        state_key, fails_key = self._breaker_keys(provider)
        try:
            with self.client.pipeline() as pipe:
                pipe.delete(state_key)
                pipe.delete(fails_key)
                pipe.execute()
        except Exception as exc:  # noqa: BLE001
            self.failures += 1
            self.last_error = str(exc)[:200]

    def acquire_probe(
        self, provider: str, ttl_s: float, base_cooldown_s: float = 30.0
    ) -> str | None:
        """"closed", "busy" (open, or a probe already out), or the token
        string this replica now holds. None means Redis failed."""
        state_key, _ = self._breaker_keys(provider)
        return self._run(
            "probe",
            PROBE_LUA,
            [state_key],
            [time.time(), ttl_s, base_cooldown_s],
        )

    def release_probe(self, provider: str, token: str) -> None:
        state_key, _ = self._breaker_keys(provider)
        try:
            if self.client.hget(state_key, "probe") == token:
                self.client.hset(state_key, "probe", 0)
        except Exception:  # noqa: BLE001
            self.failures += 1

    def breaker_state(self, provider: str) -> str | None:
        state_key, _ = self._breaker_keys(provider)
        try:
            raw = self.client.hgetall(state_key)
        except Exception:  # noqa: BLE001
            self.failures += 1
            return None
        opened = float(raw.get("opened_at") or 0)
        if not opened:
            return "closed"
        cooldown = float(raw.get("cooldown") or 30.0)
        return "half_open" if time.time() - opened >= cooldown else "open"

    # --------------------------------------------------------- governor
    def set_factor(self, slot: str, factor: float) -> None:
        try:
            key = f"{self.ns}:rate:{slot}"
            with self.client.pipeline() as pipe:
                pipe.set(key, factor)
                pipe.expire(key, 86400)
                pipe.execute()
        except Exception:  # noqa: BLE001
            self.failures += 1

    def get_factor(self, slot: str) -> float | None:
        try:
            raw = self.client.get(f"{self.ns}:rate:{slot}")
        except Exception:  # noqa: BLE001
            self.failures += 1
            return None
        return float(raw) if raw is not None else None

    def snapshot(self) -> dict[str, Any]:
        return {
            "backend": "redis",
            "namespace": self.ns,
            "failures": self.failures,
            "last_error": self.last_error,
        }
