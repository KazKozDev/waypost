"""Shared state across replicas.

These run against a real Redis when one is reachable, and skip otherwise.
A distributed counter tested only against a mock is a counter whose whole
reason for existing — what happens when two writers race — was never
tested at all.
"""
import os
import time
import uuid

import pytest

from waypost.breaker import CircuitBreaker
from waypost.cluster import SharedState, connect
from waypost.ledger import Ledger
from waypost.ratelimit import RateGovernor
from waypost.registry import Offering
from waypost.schemas import Capability

REDIS_URL = os.environ.get("WAYPOST_TEST_REDIS_URL", "redis://127.0.0.1:6379/15")
CAPS = {Capability.STREAM, Capability.JSON}


@pytest.fixture
def shared():
    client = connect(REDIS_URL)
    if client is None:
        pytest.skip(f"no redis at {REDIS_URL}")
    ns = f"wptest:{uuid.uuid4().hex[:8]}"
    yield SharedState(client, ns)
    for key in client.scan_iter(f"{ns}:*"):
        client.delete(key)


def off(name="a", rpm=None, rpd=None, tpm=None):
    return Offering(
        provider=name,
        model_id="m",
        base_url=f"http://{name}/v1",
        caps=set(CAPS),
        limit_rpm=rpm,
        limit_rpd=rpd,
        limit_tpm=tpm,
    )


def two_replicas(shared, o, tmp_path):
    """Two Ledgers over one shared counter — the actual scenario."""
    a = Ledger(tmp_path / "a.db", shared=shared)
    b = Ledger(tmp_path / "b.db", shared=shared)
    a.register(o)
    b.register(o)
    return a, b


# ------------------------------------------------------------ the quota


def test_two_replicas_share_one_quota(shared, tmp_path):
    """The reason this module exists: two replicas each tracking '9 of 10
    used' will happily send 20."""
    o = off(rpm=10)
    a, b = two_replicas(shared, o, tmp_path)

    granted = sum(1 for i in range(20) for led in (a, b) if led.reserve(o, 10))
    assert granted == 10


def test_a_replica_that_loses_the_race_is_told(shared, tmp_path):
    o = off(rpm=2)
    a, b = two_replicas(shared, o, tmp_path)
    assert a.reserve(o, 10) is True
    assert b.reserve(o, 10) is True
    assert b.reserve(o, 10) is False  # the shared counter refuses


def test_the_binding_bucket_decides(shared, tmp_path):
    """Room in rpm is not permission when rpd is spent."""
    o = off(rpm=100, rpd=3)
    a, _ = two_replicas(shared, o, tmp_path)
    assert [a.reserve(o, 10) for _ in range(5)] == [True, True, True, False, False]


def test_a_shared_429_block_stops_every_replica(shared, tmp_path):
    """A refusal one replica collected is a refusal all of them would
    have collected. Sharing the block is the whole point."""
    o = off(rpm=100)
    a, b = two_replicas(shared, o, tmp_path)
    a.penalize(o, retry_after_s=30.0)
    assert b.reserve(o, 10) is False


def test_the_actual_usage_correction_is_shared(shared, tmp_path):
    o = off(tpm=1000)
    a, b = two_replicas(shared, o, tmp_path)
    a.reserve(o, 500)
    a.commit(o, 500, 100)  # the estimate lied by 400
    b.sync_from_shared()
    assert b.snapshot()["a/m"]["tpm"] == 900


def test_local_state_catches_up_for_scoring(shared, tmp_path):
    """Scoring reads local buckets on the hot path — no network hop per
    candidate — so the local copy is refreshed by the control plane."""
    o = off(rpm=10)
    a, b = two_replicas(shared, o, tmp_path)
    for _ in range(6):
        a.reserve(o, 10)
    assert b.pressure(o) == pytest.approx(0.0)  # b has not seen it yet
    b.sync_from_shared()
    assert b.pressure(o) == pytest.approx(0.6)


def test_redis_going_away_does_not_stop_traffic(shared, tmp_path):
    """A quota counter that cannot be read is a reason to be slightly over
    budget, never a reason to stop serving."""
    o = off(rpm=10)
    led = Ledger(tmp_path / "a.db", shared=shared)
    led.register(o)
    shared.client.connection_pool.disconnect()
    shared.client = _Broken()
    assert led.reserve(o, 10) is True  # falls back to the local bucket
    assert shared.failures >= 1


class _Broken:
    def __getattr__(self, name):
        def boom(*a, **kw):
            raise ConnectionError("redis is gone")

        return boom


# ---------------------------------------------------------- the breaker


def test_the_threshold_is_counted_once_for_the_cluster(shared, tmp_path):
    """With four replicas and a local threshold of four, sixteen requests
    fail before anyone opens the circuit."""
    a = CircuitBreaker(threshold=3, base_cooldown_s=30.0, shared=shared)
    b = CircuitBreaker(threshold=3, base_cooldown_s=30.0, shared=shared)
    a.on_failure("p")
    b.on_failure("p")
    assert a.state("p") == "closed"
    b.on_failure("p")  # the third across the cluster, not the third here
    assert a.state("p") == "open"
    assert b.state("p") == "open"


def test_a_success_anywhere_closes_the_circuit(shared):
    a = CircuitBreaker(threshold=2, shared=shared)
    b = CircuitBreaker(threshold=2, shared=shared)
    a.on_failure("p")
    a.on_failure("p")
    assert b.state("p") == "open"
    b.on_success("p")
    assert a.state("p") == "closed"


def test_only_one_replica_gets_the_half_open_probe(shared):
    a = CircuitBreaker(threshold=1, base_cooldown_s=0.05, shared=shared)
    b = CircuitBreaker(threshold=1, base_cooldown_s=0.05, shared=shared)
    a.on_failure("p")
    time.sleep(0.06)
    assert a.state("p") == "half_open"

    ta = a.acquire_probe("p")
    tb = b.acquire_probe("p")
    assert not CircuitBreaker.probe_lost_race(ta)  # a probes
    assert CircuitBreaker.probe_lost_race(tb)  # b stands down

    a.release_probe(ta)
    assert not CircuitBreaker.probe_lost_race(b.acquire_probe("p"))


def test_a_closed_circuit_needs_no_token(shared):
    b = CircuitBreaker(shared=shared)
    assert b.acquire_probe("p") is None


# --------------------------------------------------------- the governor


def test_a_learned_limit_reaches_the_other_replicas(shared, tmp_path):
    o = off(rpm=20)
    led_a = Ledger(tmp_path / "a.db", shared=shared)
    led_b = Ledger(tmp_path / "b.db", shared=shared)
    led_a.register(o)
    led_b.register(o)
    ga = RateGovernor(led_a, burst_threshold=2, shared=shared)
    gb = RateGovernor(led_b, burst_threshold=2, shared=shared)

    ga.on_rate_limit(o, 0)
    ga.on_rate_limit(o, 0)
    assert ga.factor(o) == pytest.approx(0.75)
    assert gb.factor(o) == 1.0  # not yet

    assert gb.sync_from_shared([o]) == 1
    assert gb.factor(o) == pytest.approx(0.75)
    assert led_b._buckets["a/m#0"]["rpm"].limit == 15


def test_sync_is_idempotent(shared, tmp_path):
    o = off(rpm=20)
    led = Ledger(tmp_path / "a.db", shared=shared)
    led.register(o)
    g = RateGovernor(led, burst_threshold=2, shared=shared)
    shared.set_factor("a/m#0", 0.5)
    assert g.sync_from_shared([o]) == 1
    assert g.sync_from_shared([o]) == 0  # nothing changed the second time
