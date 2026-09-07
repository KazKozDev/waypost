"""Closing the learning loop: feedback, the predictor, the neighbourhood.

Everything here exists because the router was collecting the data and
using none of it — the same disease as the latency loop, one level up.
"""
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

from waypost.bandit import Bandit
from waypost.feedback import (
    FeedbackCollector,
    Signal,
    looks_like_correction,
    messages_hash,
)
from waypost.neighbors import NeighborIndex
from waypost.predictor import ModelQualityPredictor
from waypost.registry import Offering
from waypost.schemas import ChatMessage, ChatRequest, RequestProfile, Tier
from waypost.telemetry import AttemptLogEntry, Telemetry


@pytest.fixture
def db():
    with tempfile.TemporaryDirectory() as d:
        yield str(Path(d) / "t.db")


def req(text, session="s1", history=None):
    msgs = [ChatMessage(role=r, content=c) for r, c in (history or [])]
    msgs.append(ChatMessage(role="user", content=text))
    return ChatRequest(messages=msgs, session_id=session)


def collector(db, bandit=None):
    return FeedbackCollector(Telemetry(db), bandit)


# ------------------------------------------------------- reading intent


def test_a_rejection_is_recognised():
    for text in ("нет, не то", "No, that's not what I meant", "ты ошибся", "try again"):
        assert looks_like_correction(text), text


def test_an_ordinary_message_is_not_a_rejection():
    for text in ("расскажи про Python", "no problem, thanks!", "напиши тест"):
        assert not looks_like_correction(text), text


def test_a_long_message_starting_with_no_is_a_new_task():
    """"No" opening a paragraph is a new request, not a reaction."""
    long_one = "нет " + "и вот что мне на самом деле нужно сделать " * 6
    assert not looks_like_correction(long_one)


# ------------------------------------------------------------- signals


def test_repeating_the_same_request_reads_as_a_regeneration(db):
    c = collector(db)
    r = req("объясни рекурсию")
    c.remember("s1", "req1", "a/m", "chat", r.messages)
    signal = c.infer(req("объясни рекурсию"))
    assert signal is not None
    assert signal.kind == "regenerated"
    assert signal.reward < 0.5


def test_a_correction_scores_lower_than_a_regeneration(db):
    c = collector(db)
    c.remember("s1", "req1", "a/m", "chat", req("объясни рекурсию").messages)
    corrected = c.infer(req("нет, не так", history=[("user", "объясни рекурсию")]))
    c._judged.clear()
    c.remember("s1", "req1", "a/m", "chat", req("объясни рекурсию").messages)
    regen = c.infer(req("объясни рекурсию"))
    assert corrected.kind == "corrected"
    assert corrected.reward < regen.reward


def test_carrying_on_is_a_weak_positive(db):
    c = collector(db)
    c.remember("s1", "req1", "a/m", "chat", req("объясни рекурсию").messages)
    signal = c.infer(req("а теперь пример", history=[("user", "объясни рекурсию")]))
    assert signal.kind == "continued"
    assert signal.reward > 0.5


def test_no_session_means_no_inference(db):
    """Without a session there is no previous answer to judge."""
    c = collector(db)
    c.remember(None, "req1", "a/m", "chat", req("x").messages)
    assert c.infer(ChatRequest(messages=[ChatMessage(role="user", content="x")])) is None


def test_an_answer_is_judged_only_once(db):
    """A long conversation must not keep re-punishing one answer."""
    c = collector(db)
    c.remember("s1", "req1", "a/m", "chat", req("вопрос").messages)
    assert c.record(c.infer(req("нет", history=[("user", "вопрос")]))) is True
    again = c.infer(req("опять не то", history=[("user", "вопрос")]))
    assert again is None or c.record(again) is False


def test_a_stale_session_is_not_a_verdict(db):
    """An hour later the next message is a new conversation, not a
    reaction to what was said before it."""
    c = FeedbackCollector(Telemetry(db), None, session_ttl_s=0.01)
    c.remember("s1", "req1", "a/m", "chat", req("вопрос").messages)
    time.sleep(0.02)
    assert c.infer(req("нет", history=[("user", "вопрос")])) is None


# ------------------------------------------------- reaching the bandit


def test_a_correction_teaches_the_bandit(db):
    """The point of the whole module: for free-form text this is the only
    quality signal there is."""
    bandit = Bandit(db)
    c = collector(db, bandit)
    before = bandit.quality("chat", "a/m", prior=0.5)
    c.remember("s1", "req1", "a/m", "chat", req("вопрос").messages)
    c.record(c.infer(req("нет, не то", history=[("user", "вопрос")])))
    assert bandit.quality("chat", "a/m", prior=0.5) < before


def test_an_explicit_rating_outweighs_what_was_inferred(db):
    bandit = Bandit(db)
    c = collector(db, bandit)
    c.remember("s1", "req1", "a/m", "chat", req("вопрос").messages)
    c.record(c.infer(req("а ещё", history=[("user", "вопрос")])))  # weak positive
    mid = bandit.quality("chat", "a/m", prior=0.5)
    c.rate("req1", "bad", session_id="s1")
    assert bandit.quality("chat", "a/m", prior=0.5) < mid


def test_feedback_is_persisted_for_training(db):
    t = Telemetry(db)
    c = FeedbackCollector(t, None)
    c.remember("s1", "req1", "a/m", "chat", req("вопрос").messages)
    c.record(c.infer(req("нет", history=[("user", "вопрос")])))
    stats = t.feedback_stats()
    assert stats["by_kind"]["corrected"] == 1
    assert stats["by_offering"]["a/m"]["n"] == 1


def test_a_hash_distinguishes_requests():
    assert messages_hash(req("a").messages) != messages_hash(req("b").messages)
    assert messages_hash(req("a").messages) == messages_hash(req("a").messages)


# ----------------------------------------------------------- predictor


def offering(key="p/m", quality=0.5):
    provider, model = key.split("/", 1)
    return Offering(
        provider=provider, model_id=model, base_url="http://h/v1",
        quality_score=quality,
    )


def profile():
    return RequestProfile(task_probs={"chat": 1.0}, tier=Tier.M)


def test_an_untrained_predictor_returns_the_prior():
    p = ModelQualityPredictor()
    assert p.predict_p_pass(offering(quality=0.42), profile(), [0.1] * 8) == 0.42


def test_a_thin_fit_barely_moves_the_prior():
    """A regression on forty rows is mostly noise. Trusting it outright
    would route confidently on nothing."""
    p = ModelQualityPredictor()
    p.weights = {"p/m": {"bias": 8.0, "coefs": [0.0] * 4, "n": 10}}
    p.is_trained = True
    got = p.predict_p_pass(offering(quality=0.3), profile(), [0.0] * 4)
    assert 0.3 < got < 0.45  # nudged, not replaced


def test_a_well_supported_fit_is_trusted():
    p = ModelQualityPredictor()
    p.weights = {"p/m": {"bias": 8.0, "coefs": [0.0] * 4, "n": 5000}}
    p.is_trained = True
    assert p.predict_p_pass(offering(quality=0.3), profile(), [0.0] * 4) > 0.9


def test_weights_from_a_different_embedder_are_ignored():
    """Stale coefficients applied to a different vector space are not a
    degraded prediction, they are a random one."""
    p = ModelQualityPredictor()
    p.weights = {"p/m": {"bias": 8.0, "coefs": [1.0] * 256, "n": 5000}}
    p.is_trained = True
    assert p.predict_p_pass(offering(quality=0.31), profile(), [0.0] * 4) == 0.31


def test_training_recovers_a_signal_it_can_learn(db):
    """Model A is good on queries whose first coordinate is positive, B on
    the rest. The fit has to find that."""
    from scripts.train_predictor import train

    t = Telemetry(db)
    rng = np.random.default_rng(0)
    for i in range(300):
        emb = rng.normal(size=16)
        for model, good in (("A", emb[0] > 0), ("B", emb[0] <= 0)):
            ok = good if rng.random() > 0.1 else not good
            t.log_attempt_row(
                AttemptLogEntry(
                    request_id=f"r{i}{model}", attempt_no=1, ts=time.time(),
                    embedding=list(emb), embedder_version="t", l0_labels={},
                    l1_prediction=None, input_tokens=10, modality="text",
                    image_count=0, audio_duration_s=0.0, vision_token_budget=None,
                    provider="p", backend="cloud", model=model, model_version="",
                    tier="M", thinking_mode=False, routing_source="l0",
                    is_exploration=False, quota_remaining_pct=100.0,
                    quota_window_reset_in_s=0, quota_binding_limit="requests",
                    status="ok" if ok else "error", error_class="", latency_ms=100,
                    ttft_ms=10, output_tokens=5, reasoning_tokens=None,
                    peak_memory_mb=None, is_final=True,
                    outcome="pass" if ok else "fail",
                    outcome_source="hard_check", outcome_detail={},
                )
            )
    out = Path(db).parent / "pred.json"
    payload = train(db, out, verbose=False)
    assert set(payload["models"]) == {"p/A", "p/B"}
    assert payload["models"]["p/A"]["accuracy"] > 0.75


def test_a_model_with_no_variance_is_skipped(db):
    """Everything passed. A separable fit sends coefficients to infinity
    and manufactures certainty out of an accident."""
    from scripts.train_predictor import train

    t = Telemetry(db)
    rng = np.random.default_rng(1)
    for i in range(120):
        t.log_attempt_row(
            AttemptLogEntry(
                request_id=f"r{i}", attempt_no=1, ts=time.time(),
                embedding=list(rng.normal(size=8)), embedder_version="t",
                l0_labels={}, l1_prediction=None, input_tokens=10, modality="text",
                image_count=0, audio_duration_s=0.0, vision_token_budget=None,
                provider="p", backend="cloud", model="always_ok", model_version="",
                tier="M", thinking_mode=False, routing_source="l0",
                is_exploration=False, quota_remaining_pct=100.0,
                quota_window_reset_in_s=0, quota_binding_limit="requests",
                status="ok", error_class="", latency_ms=100, ttft_ms=10,
                output_tokens=5, reasoning_tokens=None, peak_memory_mb=None,
                is_final=True, outcome="pass", outcome_source="hard_check",
                outcome_detail={},
            )
        )
    payload = train(db, Path(db).parent / "p.json", verbose=False)
    assert payload["models"] == {}
    assert "no variance" in payload["skipped"]["p/always_ok"]


# -------------------------------------------------------- neighbourhood


def test_the_neighbourhood_separates_what_the_task_class_averages():
    """Two requests inside one class can be unrelated. The embedding is
    what tells them apart."""
    idx = NeighborIndex(k=10, min_neighbors=2, full_trust_n=4)
    for _ in range(10):
        idx.add([1.0, 0.0], "good_at_left", 1.0)
        idx.add([1.0, 0.0], "bad_at_left", 0.0)
        idx.add([0.0, 1.0], "good_at_left", 0.0)
        idx.add([0.0, 1.0], "bad_at_left", 1.0)

    left = idx.estimate([1.0, 0.05])
    assert left["good_at_left"][0] > 0.8
    assert left["bad_at_left"][0] < 0.2
    right = idx.estimate([0.05, 1.0])
    assert right["good_at_left"][0] < 0.2


def test_too_few_neighbours_is_no_estimate():
    idx = NeighborIndex(min_neighbors=5)
    idx.add([1.0, 0.0], "a", 1.0)
    assert idx.estimate([1.0, 0.0]) == {}


def test_trust_grows_with_the_number_of_neighbours():
    idx = NeighborIndex(k=50, min_neighbors=2, full_trust_n=10)
    for _ in range(3):
        idx.add([1.0, 0.0], "thin", 1.0)
    for _ in range(20):
        idx.add([1.0, 0.0], "thick", 1.0)
    est = idx.estimate([1.0, 0.0])
    assert est["thin"][1] < est["thick"][1]
    assert est["thick"][1] == 1.0


def test_a_different_embedder_is_refused():
    idx = NeighborIndex()
    assert idx.add([1.0, 0.0], "a", 1.0) is True
    assert idx.add([1.0, 0.0, 0.0], "a", 1.0) is False


def test_the_index_is_bounded():
    idx = NeighborIndex(capacity=50)
    for i in range(200):
        idx.add([float(i), 1.0], "a", 1.0)
    assert idx.snapshot()["rows"] == 50


def test_old_evidence_weighs_less():
    """A model swapped behind the same id last month is weak evidence
    about today."""
    idx = NeighborIndex(k=50, min_neighbors=2, full_trust_n=2, half_life_days=1.0)
    now = time.time()
    for _ in range(5):
        idx.add([1.0, 0.0], "stale", 1.0, ts=now - 30 * 86_400)
        idx.add([1.0, 0.0], "fresh", 1.0, ts=now)
    fresh_recent = idx.estimate([1.0, 0.0])
    assert "fresh" in fresh_recent
    # Both mean 1.0; what differs is the weight behind them, visible in
    # how a contrary fresh observation moves each.
    idx.add([1.0, 0.0], "stale", 0.0, ts=now)
    idx.add([1.0, 0.0], "fresh", 0.0, ts=now)
    after = idx.estimate([1.0, 0.0])
    assert after["stale"][0] < after["fresh"][0]
