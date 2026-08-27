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
