"""Hot reload of the offering pool.

Reloading configuration that came from the network is only safe if a bad
build can be refused before it lands, and a build that passed every check
but routes worse can still be taken back.
"""
import asyncio

import pytest

from waypost.registry import Offering, Registry
from waypost.schemas import Capability, Tier
from waypost.selfupdate import (
    ProbeTriggers,
    RegistryReloader,
    check_invariants,
    collect_triggers,
)

CAPS = {Capability.STREAM, Capability.JSON, Capability.TOOLS}


def off(model_id="m", provider="cloud", **kw):
    kw.setdefault("caps", set(CAPS))
    return Offering(
        provider=provider, model_id=model_id, base_url="http://h/v1", **kw
    )


def local(model_id="q"):
    return Offering(
        provider="mlx",
        model_id=model_id,
        base_url="http://127.0.0.1:8081/v1",
        is_local=True,
        trains_on_data=False,
        caps=set(CAPS),
    )


def pool(n=4, **kw):
    return [off(f"m{i}", **kw) for i in range(n)] + [local()]


# ----------------------------------------------------------- invariants


def test_an_empty_pool_is_refused():
    ok, why = check_invariants([], pool())
    assert not ok and "no usable" in why


def test_losing_the_local_fallback_is_refused():
    """The local rung is the one that always answers. A pool without it
    can only fail upward, into a 502."""
    ok, why = check_invariants([off("m0"), off("m1")], pool())
    assert not ok and "local fallback" in why


def test_a_halved_pool_is_treated_as_a_bad_read():
    """A provider returning a truncated /models page must not be able to
    wipe the pool in one cycle."""
    ok, why = check_invariants([off("m0"), local()], pool(8))
    assert not ok and "shrank" in why


def test_losing_a_tier_is_refused():
    current = [off("big", tier=Tier.L), off("mid", tier=Tier.M), local()]
    fresh = [off("mid", tier=Tier.M), off("mid2", tier=Tier.M), local()]
    ok, why = check_invariants(fresh, current)
    assert not ok and "tier" in why


def test_losing_every_tool_capable_model_is_refused():
    current = [off("a"), off("b"), local()]
    stripped = [off("a", caps={Capability.STREAM}), off("b", caps={Capability.STREAM})]
    stripped.append(
        Offering(
            provider="mlx",
            model_id="q",
            base_url="http://127.0.0.1:8081/v1",
            is_local=True,
            caps={Capability.STREAM},
        )
    )
    ok, why = check_invariants(stripped, current)
    assert not ok and "tools" in why


def test_a_paid_offering_cannot_enter_a_free_only_pool():
    ok, why = check_invariants(pool() + [off("paid", free=False)], pool())
    assert not ok and "free_only" in why


def test_a_sound_pool_passes():
    ok, why = check_invariants(pool(5), pool(4))
    assert ok and why == ""


# ------------------------------------------------------------ the swap


def make_reloader(registry, build, rate=lambda: (100, 0.95), **kw):
    return RegistryReloader(
        registry, build=build, success_rate=rate, observe_window_s=0.05, **kw
    )


def test_a_refused_build_leaves_the_pool_untouched():
    """Refusing to reload is always safe. Reloading halfway is not."""
    registry = Registry(pool(4))
    before = registry.version
    r = make_reloader(registry, build=lambda: [])
    result = r.reload(observe=False)
    assert not result.ok
    assert registry.version == before
    assert len(registry.all()) == 5


def test_a_good_build_swaps_and_bumps_the_version():
    registry = Registry(pool(4))
    fresh = pool(4) + [off("new")]
    r = make_reloader(registry, build=lambda: fresh)
    result = r.reload(observe=False)
    assert result.ok
    assert result.added == ["cloud/new"]
    assert registry.version == 2
    assert registry.get("cloud/new") is not None


def test_an_identical_build_does_not_bump_the_version():
    """Bumping on every cycle would make the rollback history meaningless."""
    registry = Registry(pool(4))
    r = make_reloader(registry, build=lambda: pool(4))
    assert r.reload(observe=False).reason == "no change"
    assert registry.version == 1


def test_learned_state_survives_a_reload():
    """A rebuild from the manifest must not throw away what the control
    plane measured, or every reload starts the pool cold."""
    registry = Registry(pool(4))
    live = registry.get("cloud/m0")
    live.ttft_p50_ms = 420.0
    live.success_rate = 0.77
    live.lifecycle = "shadow"
    live.dead_streak = 1

    r = make_reloader(registry, build=lambda: pool(5))  # a fresh manifest read
    assert r.reload(observe=False).ok
    revived = registry.get("cloud/m0")
    assert revived.ttft_p50_ms == 420.0
    assert revived.success_rate == 0.77
    assert revived.lifecycle == "shadow"
    assert revived.dead_streak == 1


def test_a_billing_proof_survives_a_reload():
    """A manifest claiming a model is free cannot undo an actual bill."""
    from waypost import pricing

    registry = Registry(pool(4))
    registry.get("cloud/m0").set_pricing(
        pricing.Verdict(pricing.Cost.PAID, pricing.Source.BILLED, "billed 0.4")
    )
    r = make_reloader(registry, build=lambda: pool(5), free_only=False)
    assert r.reload(observe=False).ok
    assert registry.get("cloud/m0").free is False


def test_buckets_are_registered_before_traffic():
    seen = []
    registry = Registry(pool(4))
    r = make_reloader(
        registry, build=lambda: pool(5), on_swap=lambda offs: seen.append(len(offs))
    )
    r.reload(observe=False)
    assert seen == [6]


# ------------------------------------------------------------- rollback


def test_rollback_restores_the_previous_pool():
    registry = Registry(pool(4))
    r = make_reloader(registry, build=lambda: pool(4) + [off("new")])
    r.reload(observe=False)
    assert registry.get("cloud/new") is not None
    assert r.rollback("test").ok
    assert registry.get("cloud/new") is None
    assert registry.version == 3  # forward, not backward: history is append-only


def test_rollback_without_history_is_a_no_op():
    registry = Registry(pool(4))
    r = make_reloader(registry, build=lambda: pool(4))
    assert not r.rollback().ok


@pytest.mark.asyncio
async def test_a_regression_rolls_itself_back():
    """Invariants only catch what we thought to check. This catches the
    rest: a pool that passes every check and still routes worse."""
    registry = Registry(pool(4))
    rates = iter([(100, 0.95), (200, 0.40)])  # before the swap, then after
    r = make_reloader(
        registry, build=lambda: pool(4) + [off("bad")], rate=lambda: next(rates)
    )
    result = r.reload(observe=True)
    assert result.ok
    assert registry.get("cloud/bad") is not None
    await asyncio.sleep(0.3)  # the observation window plus slack
    assert registry.get("cloud/bad") is None
    assert any(e["kind"] == "rollback" for e in r.events())


@pytest.mark.asyncio
async def test_a_healthy_pool_is_left_alone():
    registry = Registry(pool(4))
    rates = iter([(100, 0.95), (200, 0.94)])
    r = make_reloader(
        registry, build=lambda: pool(4) + [off("fine")], rate=lambda: next(rates)
    )
    r.reload(observe=True)
    await asyncio.sleep(0.3)
    assert registry.get("cloud/fine") is not None
    assert not any(e["kind"] == "rollback" for e in r.events())


@pytest.mark.asyncio
async def test_too_little_traffic_is_not_a_verdict():
    registry = Registry(pool(4))
    rates = iter([(100, 0.95), (105, 0.10)])  # 5 attempts says nothing
    r = make_reloader(
        registry, build=lambda: pool(4) + [off("quiet")], rate=lambda: next(rates)
    )
    r.reload(observe=True)
    await asyncio.sleep(0.3)
    assert registry.get("cloud/quiet") is not None


# ------------------------------------------------------ probe triggers


def test_triggers_deduplicate_and_drain():
    t = ProbeTriggers()
    t.request("a/m", "half_open")
    t.request("a/m", "drift")  # the first reason stands
    assert t.pending() == {"a/m": "half_open"}
    assert t.drain() == {"a/m": "half_open"}
    assert t.pending() == {}


def test_triggers_are_bounded():
    t = ProbeTriggers(max_pending=2)
    for i in range(10):
        t.request(f"a/m{i}", "drift")
    assert len(t.pending()) == 2


def test_a_half_open_provider_is_queued_for_a_probe():
    """A half-open provider is waiting for one call to decide its fate. If
    no plan happens to reach it, that call has to come from the probe."""
    from waypost.breaker import CircuitBreaker
    from waypost.latency import LatencyTracker

    registry = Registry([off("m0")])
    breaker = CircuitBreaker(threshold=1, base_cooldown_s=0.0)
    breaker.on_failure("cloud")
    t = ProbeTriggers()
    collect_triggers(t, registry, breaker, LatencyTracker())
    assert t.pending() == {"cloud/m0": "breaker half_open"}


def test_a_drifting_offering_is_queued_for_a_probe():
    from waypost.breaker import CircuitBreaker
    from waypost.latency import LatencyTracker

    registry = Registry([off("m0")])
    lat = LatencyTracker(drift_ratio=2.5, min_samples=3)
    lat.seed("cloud/m0", 500)
    for _ in range(8):
        lat.observe("cloud/m0", 9000)
    t = ProbeTriggers()
    collect_triggers(t, registry, CircuitBreaker(), lat)
    assert t.pending() == {"cloud/m0": "latency drift"}
