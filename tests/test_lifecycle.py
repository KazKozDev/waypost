"""Offering lifecycle: discovered → probed → serving → quarantined → back.

The pool used to be one-way. Discovery only ever added; a model that
vanished from /models stayed forever; a model that answered 403 once was
disabled permanently, and since the probe job only looks at usable
offerings, nothing ever re-checked it.
"""
import time

import pytest

from waypost.discovery import MISS_STREAK_LIMIT, _apply_gone
from waypost.probe import DEAD_STREAK_LIMIT, apply_to_registry
from waypost.registry import Offering, Registry
from waypost.schemas import Capability

CAPS = {Capability.STREAM, Capability.JSON, Capability.TOOLS}


def off(model_id="m", provider="cloud", **kw):
    kw.setdefault("caps", set(CAPS))
    return Offering(
        provider=provider,
        model_id=model_id,
        base_url="http://h/v1",
        **kw,
    )


# ------------------------------------------------------------- gone models


def test_missing_once_is_not_evidence():
    """Providers rotate /models pages. Acting on a single absence would let
    one partial response wipe the pool."""
    o = off()
    reg = Registry([o])
    res = _apply_gone(reg, "cloud", set())
    assert res["retired"] == []
    assert o.lifecycle == "active"
    assert o.miss_streak == 1


def test_missing_for_three_cycles_is_retired():
    o = off()
    reg = Registry([o])
    for _ in range(MISS_STREAK_LIMIT):
        _apply_gone(reg, "cloud", set())
    assert o.lifecycle == "quarantine"
    assert o.usable is False


def test_reappearing_resets_the_streak():
    o = off()
    reg = Registry([o])
    _apply_gone(reg, "cloud", set())
    _apply_gone(reg, "cloud", {"m"})
    assert o.miss_streak == 0
    assert o.lifecycle == "active"


# -------------------------------------------------------------- dead probes


def test_one_dead_probe_does_not_remove_a_model():
    """Gateways return 403 during a rotation. A single bad minute used to
    remove a working model for the life of the process."""
    o = off()
    reg = Registry([o])
    apply_to_registry(reg, {o.key: {"status": "dead", "http_status": 403}})
    assert o.lifecycle == "active"
    assert o.dead_streak == 1


def test_two_dead_probes_quarantine():
    o = off()
    reg = Registry([o])
    for _ in range(DEAD_STREAK_LIMIT):
        apply_to_registry(reg, {o.key: {"status": "dead", "http_status": 404}})
    assert o.lifecycle == "quarantine"


def test_a_healthy_probe_resurrects_a_quarantined_model():
    o = off()
    reg = Registry([o])
    for _ in range(DEAD_STREAK_LIMIT):
        apply_to_registry(reg, {o.key: {"status": "dead"}})
    assert o.usable is False
    apply_to_registry(reg, {o.key: {"status": "healthy", "ttft_p50_ms": 900}})
    assert o.lifecycle == "shadow"
    assert o.usable is True
    assert o.ttft_p50_ms == 900


def test_quarantine_is_only_retried_after_the_cooldown():
    o = off()
    reg = Registry([o])
    reg.transition(o.key, "quarantine")
    assert reg.quarantined(older_than_s=3600) == []
    o.quarantined_at = time.time() - 7200
    assert reg.quarantined(older_than_s=3600) == [o]


def test_a_model_proven_to_bill_is_never_resurrected():
    """A billing record outranks any later claim of being free."""
    from waypost import pricing

    o = off()
    o.set_pricing(pricing.Verdict(pricing.Cost.PAID, pricing.Source.BILLED, "billed 0.4"))
    reg = Registry([o])
    reg.transition(o.key, "quarantine")
    o.quarantined_at = time.time() - 7200
    assert reg.quarantined(older_than_s=3600) == []


# ------------------------------------------------------- candidate intake


def test_a_candidate_may_not_serve_traffic():
    """A model discovered thirty seconds ago has a made-up quality score and
    an unmeasured TTFT. Letting it into the pool means it can win a plan on
    invented numbers."""
    o = off(lifecycle="candidate")
    assert o.usable is False


def test_a_probed_candidate_becomes_a_shadow():
    o = off(lifecycle="candidate")
    reg = Registry([o])
    apply_to_registry(reg, {o.key: {"status": "healthy", "ttft_p50_ms": 700}})
    assert o.lifecycle == "shadow"
    assert o.usable is True
    assert o.weight <= 0.2  # a small share, not a coin flip against a proven model


# ------------------------------------------------- declared vs working caps


def test_a_failed_canary_strips_the_declared_capability():
    """Provider docs say the model supports tool calling. The canary is the
    only thing that knows whether it actually does."""
    o = off()
    reg = Registry([o])
    apply_to_registry(
        reg,
        {o.key: {"status": "healthy", "supports_tools": False, "supports_json": True}},
    )
    assert Capability.TOOLS not in o.caps
    assert Capability.JSON in o.caps
