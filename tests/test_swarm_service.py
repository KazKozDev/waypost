import uuid

import pytest

from waypost.swarm.service import SwarmService
from waypost.swarm.store import RunStore


def test_ui_run_control_and_event_cursor(tmp_path, monkeypatch):
    service = SwarmService(tmp_path, "http://127.0.0.1:8081/v1")
    launched = []
    monkeypatch.setattr(service, "_launch", lambda directory, **kwargs: launched.append(kwargs))
    run_id = service.create("Make a report")
    assert len(run_id) == 32 and service.status(run_id)["status"] == "interrupted"
    assert service.events(run_id)["events"][0]["event"] == "user_task"

    service.pause(run_id)
    service.message(run_id, "Add a summary")
    assert len(launched) == 1
    assert service.status(run_id)["pending_messages"][0]["text"] == "Add a summary"
    service.resume(run_id)
    assert launched[-1]["resume"] is True
    assert not service.status(run_id)["paused"]

    first = service.events(run_id, 0)
    assert first["next_offset"] == len(first["events"])
    assert service.events(run_id, first["next_offset"])["events"] == []
    with pytest.raises(ValueError):
        service.directory("../../etc")


def test_ui_status_reads_checkpoint(tmp_path):
    service = SwarmService(tmp_path, "http://127.0.0.1:8081/v1")
    run_id = uuid.uuid4().hex
    directory = tmp_path / run_id
    store = RunStore(directory)
    (directory / "task.txt").write_text("Analyze")
    store.save({"status": "completed", "phase": "review", "round": 2,
                "calls": 7, "task": "Analyze", "draft": "Done", "monitor": []})
    artifact = directory / "workspace/artifacts/r1/writer/report.md"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("report")
    status = service.status(run_id)
    assert status["status"] == "completed" and status["calls"] == 7
    assert status["artifacts"] == ["artifacts/r1/writer/report.md"]


def test_correction_does_not_restart_interrupted_run_until_resume(tmp_path, monkeypatch):
    service = SwarmService(tmp_path, "http://127.0.0.1:8081/v1")
    launched = []
    monkeypatch.setattr(service, "_launch", lambda directory, **kwargs: launched.append(kwargs))
    run_id = service.create("Draft a report")
    directory = service.directory(run_id)
    RunStore(directory).save({"status": "interrupted", "task": "Draft a report"})
    service.message(run_id, "Include an example")
    assert len(launched) == 1
    assert service.status(run_id)["pending_messages"][0]["text"] == "Include an example"
    service.resume(run_id)
    assert launched[-1]["resume"] is True
