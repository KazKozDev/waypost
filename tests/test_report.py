"""Проверка отчёта по телеметрии."""
import os
import tempfile
import time

import pytest

from waypost.report import generate


@pytest.fixture
def db():
    with tempfile.TemporaryDirectory() as d:
        yield os.path.join(d, "t.db")


def _seed(db):
    import sqlite3

    c = sqlite3.connect(db)
    c.execute(
        """CREATE TABLE IF NOT EXISTS attempts (
        ts REAL, offering TEXT, task_class TEXT, tier TEXT,
        complexity REAL, confidence REAL, classifier_source TEXT,
        est_tokens INTEGER, tokens INTEGER, ok INTEGER,
        verdict TEXT, latency_ms REAL)"""
    )
    now = time.time()
    rows = [
        (now, "p/m1", "chat", "M", 0.4, 0.6, "rules", 100, 80, 1, "ok", 200),
        (now, "p/m1", "code", "L", 0.8, 0.5, "rules", 500, 400, 1, "ok", 300),
        (now, "p/m2", "code", "L", 0.8, 0.5, "rules", 500, 300, 0, "switch", 100),
    ]
    c.executemany("INSERT INTO attempts VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    c.commit()
    c.close()


def test_report_overview(db):
    _seed(db)
    r = generate(db)
    assert r["overview"]["attempts"] == 3
    assert r["overview"]["success_rate"] == pytest.approx(2 / 3, abs=0.001)
    assert r["overview"]["tokens"] == 780
    assert r["overview"]["verdicts"] == {"ok": 2, "switch": 1}


def test_report_by_offering(db):
    _seed(db)
    r = generate(db)
    by = {row["offering"]: row for row in r["by_offering"]}
    assert by["p/m1"]["attempts"] == 2
    assert by["p/m1"]["success_rate"] == 1.0
    assert by["p/m2"]["success_rate"] == 0.0


def test_report_by_task(db):
    _seed(db)
    r = generate(db)
    by = {row["task"]: row for row in r["by_task"]}
    assert by["code"]["attempts"] == 2
    assert by["chat"]["attempts"] == 1


def _seed_routing(db):
    from waypost.telemetry import AttemptLogEntry, Telemetry

    t = Telemetry(db)
    now = time.time()

    def entry(request_id, attempt_no, outcome, latency_ms, in_tok, out_tok,
              arm="", is_final=True, exploration=False):
        t.log_attempt_row(
            AttemptLogEntry(
                request_id=request_id, attempt_no=attempt_no, ts=now,
                embedding=None, embedder_version="t", l0_labels={},
                l1_prediction=None, input_tokens=in_tok, modality="text",
                image_count=0, audio_duration_s=0.0, vision_token_budget=None,
                provider="p", backend="cloud", model="m", model_version="",
                tier="M", thinking_mode=False, routing_source="l0",
                is_exploration=exploration, quota_remaining_pct=100.0,
                quota_window_reset_in_s=0, quota_binding_limit="requests",
                status="ok" if outcome == "pass" else "error", error_class="",
                latency_ms=latency_ms, ttft_ms=10, output_tokens=out_tok,
                reasoning_tokens=None, peak_memory_mb=None, is_final=is_final,
                outcome=outcome, outcome_source="hard_check", outcome_detail={},
                arm=arm,
            )
        )

    entry("A", 1, "pass", 100, 10, 5)
    entry("B", 1, "fail", 50, 10, 0, is_final=False)
    entry("B", 2, "pass", 200, 10, 20, arm="b")
    entry("C", 1, "fail", 150, 10, 0, arm="b")
    entry("shadow-1", 98, "pass", 300, 10, 10, exploration=True)
    t.log_feedback(
        request_id="B", offering="p/m", task_class="chat",
        kind="corrected", reward=0.15, source="implicit",
    )


def test_routing_quality_counts_requests_not_deliveries(db):
    from waypost.report import routing_quality

    _seed_routing(db)
    q = routing_quality(db)

    assert q["requests"] == 3  # the shadow measurement is not a request
    assert q["first_attempt_pass_rate"] == round(1 / 3, 3)
    assert q["escalation_rate"] == round(1 / 3, 3)
    assert q["correction_rate"] == round(1 / 3, 3)
    assert q["mean_attempts"] == round(4 / 3, 2)
    assert q["p50_latency_ms"] == 150
    assert q["p95_latency_ms"] == 250
    assert q["total_tokens"] == 65


def test_routing_quality_by_arm(db):
    from waypost.report import routing_quality

    _seed_routing(db)
    arms = routing_quality(db)["by_arm"]

    assert arms["control"]["requests"] == 1
    assert arms["control"]["first_attempt_pass_rate"] == 1.0
    assert arms["b"]["requests"] == 2
    assert arms["b"]["first_attempt_pass_rate"] == 0.0
    assert arms["b"]["escalation_rate"] == 0.5
    assert arms["b"]["correction_rate"] == 0.5


def test_routing_quality_empty_without_attempt_log(db):
    from waypost.report import routing_quality

    _seed(db)
    q = routing_quality(db)
    assert q["requests"] == 0


def test_generate_includes_routing(db):
    _seed(db)
    assert generate(db)["routing"]["requests"] == 0
