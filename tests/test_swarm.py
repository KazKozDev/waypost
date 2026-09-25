import json
import threading
import time

import httpx
import pytest

from waypost.swarm import SwarmConfig, SwarmEngine
from waypost.swarm.llm import Budget, BudgetExceeded, WaypostLLM, SwarmsBackend
from waypost.swarm.models import Plan
from waypost.swarm.store import RunStore
from waypost.swarm.tools import WorkspaceTools


def plan(tasks=None):
    return {"acceptance": ["Deliver an evidenced answer"], "tasks": tasks or [
        {"id": "a", "role": "analyst", "instruction": "Compute", "depends_on": []}]}


def final(text="done"):
    return {"kind": "final", "answer": text}


class Scripted:
    def __init__(self, script):
        self.script = {key: list(value) for key, value in script.items()}
        self.prompts = []
        self.lock = threading.Lock()

    def factory(self, config, budget, store):
        self.budget = budget
        return self

    def ask(self, role, system, prompt, schema=None):
        self.budget.reserve()
        with self.lock:
            self.prompts.append((role, prompt))
            value = (self.script[role].pop(0) if role in self.script else
                     {"action": "continue", "reason": "Work is progressing", "guidance": ""}
                     if role == "progress-monitor" else None)
            if value is None:
                raise KeyError(role)
        if isinstance(value, Exception):
            raise value
        return json.dumps(value) if not isinstance(value, str) else value


def script(**overrides):
    values = {"supervisor": [plan()], "r1:a": [final()],
              "r1:synthesis": [final("A concrete answer")], "r1:audit": [final("Evidence checked")],
              "review-verdict": [{"passed": True, "findings": [], "repair": None}]}
    values.update(overrides)
    return Scripted(values)


def test_graph_rejects_cycles_and_missing_dependencies():
    for dep in ["a", "missing"]:
        with pytest.raises(ValueError):
            Plan.model_validate(plan([{"id": "a", "role": "x", "instruction": "x", "depends_on": [dep]}]))
    with pytest.raises(ValueError):
        Plan.model_validate(plan([{"id": "synthesis", "role": "x", "instruction": "x"}]))


def test_completion_and_tool_artifact(tmp_path):
    backend = script(**{"r1:a": [{"kind": "tool", "tool": "write_file", "arguments": {"path": "answer.txt", "content": "42"}}, final("42")]})
    state = SwarmEngine(tmp_path, backend_factory=backend.factory).run("Compute 6 * 7")
    assert state["status"] == "completed"
    assert (tmp_path / "workspace/artifacts/r1/a/answer.txt").read_text() == "42"
    assert (tmp_path / "result.md").read_text() == "A concrete answer"
    assert state["calls"] == 8
    assert SwarmEngine(tmp_path, backend_factory=backend.factory).run(resume=True) == state


def test_repair_round_preserves_original_acceptance(tmp_path):
    repair = plan()
    repair["acceptance"] = ["Weaker replacement"]
    backend = script(**{"review-verdict": [{"passed": False, "findings": ["Fix calculation"], "repair": repair},
                                           {"passed": True, "findings": [], "repair": None}],
                        "r2:a": [final("fixed")], "r2:synthesis": [final("corrected")], "r2:audit": [final("verified")]})
    state = SwarmEngine(tmp_path, backend_factory=backend.factory).run("Task")
    assert state["status"] == "completed" and state["round"] == 2
    assert state["acceptance"] == ["Deliver an evidenced answer"]
    assert state["draft"] == "corrected"


def test_budget_failure_is_not_success(tmp_path):
    backend = script()
    state = SwarmEngine(tmp_path, SwarmConfig(max_calls=1), backend.factory).run("Task")
    assert state["status"] == "budget_exhausted"
    assert state["calls"] == 1


def test_resume_does_not_repeat_finished_task(tmp_path):
    backend = script(**{"r1:synthesis": [RuntimeError("outage"), final("recovered")]})
    engine = SwarmEngine(tmp_path, backend_factory=backend.factory)
    assert engine.run("Task")["status"] == "failed"
    state = engine.run(resume=True)
    assert state["status"] == "completed"
    assert sum(role == "r1:a" for role, _ in backend.prompts) == 1


def test_parallel_dependencies(tmp_path):
    tasks = [{"id": x, "role": x, "instruction": x, "depends_on": [] if x != "c" else ["a", "b"]} for x in "abc"]
    barrier = threading.Barrier(2)
    backend = script(**{"supervisor": [plan(tasks)], "r1:b": [final("B")], "r1:c": [final("C")]})
    original = backend.ask
    def ask(role, system, prompt, schema=None):
        if role in {"r1:a", "r1:b"}:
            barrier.wait(timeout=5)
        if role == "r1:c":
            assert '"a": "done"' in prompt and '"b": "B"' in prompt
        return original(role, system, prompt, schema)
    backend.ask = ask
    assert SwarmEngine(tmp_path, backend_factory=backend.factory).run("Task")["status"] == "completed"


def test_downstream_agent_receives_dependency_file_path(tmp_path):
    tasks = [
        {"id": "write", "role": "writer", "instruction": "Write", "depends_on": []},
        {"id": "check", "role": "checker", "instruction": "Read writer file", "depends_on": ["write"]},
    ]
    backend = script(**{"supervisor": [plan(tasks)], "r1:write": [
        {"kind": "tool", "tool": "write_file", "arguments": {"path": "CHECKLIST.md", "content": "five checks"}},
        final("written")], "r1:check": [final("checked")]})
    original = backend.ask
    def ask(role, system, prompt, schema=None):
        if role == "r1:check":
            assert "artifacts/r1/write/CHECKLIST.md" in prompt
            assert "Do not recreate a file just to inspect it" in system
        return original(role, system, prompt, schema)
    backend.ask = ask
    assert SwarmEngine(tmp_path, backend_factory=backend.factory).run("Task")["status"] == "completed"


def test_malformed_plan_repaired(tmp_path):
    backend = script(**{"supervisor": ["not json", plan()]})
    assert SwarmEngine(tmp_path, backend_factory=backend.factory).run("Task")["status"] == "completed"


def test_repeated_invalid_review_replans_then_finishes_on_its_own(tmp_path):
    invalid = {"passed": False, "findings": ["repair needed"],
               "repair": plan([{"id": "fix", "role": "fixer", "instruction": "Fix",
                                "depends_on": ["unknown"]}])}
    many = {f"r{r}:{k}": [final("retried")] * 3 for r in range(1, 12) for k in ("a", "synthesis", "audit")}
    backend = script(**many, **{"supervisor": [plan()] * 12, "review-verdict": [invalid] * 60})
    state = SwarmEngine(tmp_path, backend_factory=backend.factory).run("Task")
    assert state["status"] == "completed"
    assert "Завершено автономно" in state["draft"]
    assert any(e == "replan" for e in (json.loads(l)["event"] for l in
               (tmp_path / "events.jsonl").read_text().splitlines()))


def test_review_limit(tmp_path):
    backend = script(**{"review-verdict": [{"passed": False, "findings": ["missing"], "repair": plan()}]})
    state = SwarmEngine(tmp_path, SwarmConfig(max_rounds=1), backend.factory).run("Task")
    assert state["status"] == "needs_attention"


def test_workspace_escape_and_python_opt_in(tmp_path):
    tools = WorkspaceTools(tmp_path / "workspace", "artifacts/a")
    for name, args in [("read_file", {"path": "../../secret"}),
                       ("write_file", {"path": "../escape", "content": "x"}),
                       ("run_python", {"code": "print(1)"})]:
        with pytest.raises(ValueError):
            tools.execute(name, args)
    (tools.root / "link").symlink_to(tmp_path)
    with pytest.raises(ValueError):
        tools.execute("read_file", {"path": "link/secret"})


def test_write_accepts_advertised_workspace_relative_output_path(tmp_path):
    tools = WorkspaceTools(tmp_path, "artifacts/r1/writer")
    result = json.loads(tools.execute("write_file", {
        "path": "artifacts/r1/writer/CHECKLIST.md", "content": "done"}))
    assert result["written"] == "artifacts/r1/writer/CHECKLIST.md"
    assert "done" in tools.execute("read_file", {"path": result["written"]})


def test_same_round_agents_cannot_create_duplicate_artifact_path(tmp_path):
    first = WorkspaceTools(tmp_path, "artifacts/r1/writer")
    second = WorkspaceTools(tmp_path, "artifacts/r1/reviewer")
    first.execute("write_file", {"path": "CHECKLIST.md", "content": "original"})
    with pytest.raises(ValueError, match="read that path"):
        second.execute("write_file", {"path": "CHECKLIST.md", "content": "duplicate"})
    assert (tmp_path / "artifacts/r1/writer/CHECKLIST.md").read_text() == "original"


def test_python_result_and_timeout(tmp_path):
    tools = WorkspaceTools(tmp_path, "out", allow_python=True)
    assert json.loads(tools.execute("run_python", {"code": "print(6 * 7)"}))["output"].strip() == "42"
    assert json.loads(tools.execute("run_python", {"code": "import time; time.sleep(10)"}, timeout=0.1))["timed_out"]


def test_waypost_adapter_contract(tmp_path):
    store = RunStore(tmp_path)
    state = {"calls": 0}
    config = SwarmConfig(privacy="strict")
    budget = Budget(config, state, store)
    def handler(request):
        data = json.loads(request.content)
        assert str(request.url) == "http://127.0.0.1:8080/v1/chat/completions"
        assert data["model"] == "auto" and data["privacy"] == "strict"
        assert data["latency_class"] == "batch" and data["no_cache"]
        assert data["session_id"] == "run:worker"
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}], "router": {"provider": "local"}})
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert WaypostLLM(config, budget, store, "run:worker", "system", client).run("task") == "ok"
    assert state["calls"] == 1


def test_unlimited_run_can_pause_and_resume(tmp_path):
    assert SwarmConfig().max_calls is None and SwarmConfig().max_rounds is None
    backend = script()
    engine = SwarmEngine(tmp_path, backend_factory=backend.factory)
    engine.store.update_control(paused=True)
    result = []
    thread = threading.Thread(target=lambda: result.append(engine.run("Task")))
    thread.start()
    for _ in range(50):
        if (tmp_path / "state.json").exists() and engine.store.load().get("status") == "paused":
            break
        time.sleep(0.02)
    assert engine.store.load()["status"] == "paused"
    engine.store.update_control(paused=False)
    thread.join(timeout=5)
    assert not thread.is_alive() and result[0]["status"] == "completed"


def test_user_correction_replans_completed_run(tmp_path):
    backend = script(**{"supervisor": [plan(), plan()],
                        "r2:a": [final("changed")], "r2:synthesis": [final("updated")],
                        "r2:audit": [final("checked")],
                        "review-verdict": [{"passed": True, "findings": [], "repair": None},
                                           {"passed": True, "findings": [], "repair": None}]})
    engine = SwarmEngine(tmp_path, backend_factory=backend.factory)
    assert engine.run("Original task")["status"] == "completed"
    engine.store.update_control(message="Change the result")
    state = engine.run(resume=True)
    assert state["status"] == "completed" and state["round"] == 2
    assert state["messages"][-1]["text"] == "Change the result"
    assert any(role == "supervisor" and "Change the result" in prompt for role, prompt in backend.prompts)


def test_progress_agent_resolves_ambiguity_itself(tmp_path):
    """Autonomous: needs_input becomes a replan by assumption, not a stop."""
    backend = script(**{"supervisor": [plan(), plan()],
                        "r2:a": [final()], "r2:synthesis": [final("answer")],
                        "r2:audit": [final("checked")],
                        "progress-monitor": [{"action": "needs_input",
                            "reason": "Source data is ambiguous", "guidance": "Ask which source to use"}]
                            + [{"action": "continue", "reason": "Progressing"}] * 5})
    state = SwarmEngine(tmp_path, backend_factory=backend.factory).run("Task")
    assert state["status"] == "completed" and state["round"] == 2
    assert state["monitor"][0]["trigger"] == "specialists_finished"
    assert "assumption" in state["monitor_guidance"]


def test_endless_failed_reviews_finish_best_effort(tmp_path):
    """The loop fuse: a review that never passes ends the run with the best
    draft and the open findings, instead of repairing forever."""
    failing = {"passed": False, "findings": ["still missing X"], "repair": plan()}
    many = {f"r{r}:{k}": [final(f"r{r}")] * 3 for r in range(1, 12) for k in ("a", "synthesis", "audit")}
    backend = script(**many, **{"supervisor": [plan()] * 12, "review-verdict": [failing] * 40})
    state = SwarmEngine(tmp_path, backend_factory=backend.factory).run("Task")
    assert state["status"] == "completed"
    assert "Завершено автономно" in state["draft"] and "still missing X" in state["draft"]


def test_progress_agent_can_reject_reviewers_premature_success(tmp_path):
    backend = script(**{"supervisor": [plan(), plan()],
                        "r2:a": [final("rechecked")], "r2:synthesis": [final("corrected")],
                        "r2:audit": [final("verified")],
                        "review-verdict": [{"passed": True, "findings": [], "repair": None}] * 2,
                        "progress-monitor": [
                            {"action": "continue", "reason": "Work exists"},
                            {"action": "redirect", "reason": "Evidence is insufficient",
                             "guidance": "Verify the source before accepting"},
                            {"action": "continue", "reason": "Work rechecked"},
                            {"action": "continue", "reason": "Evidence verified"}]})
    state = SwarmEngine(tmp_path, backend_factory=backend.factory).run("Task")
    assert state["status"] == "completed" and state["round"] == 2
    assert any(item["trigger"] == "review_passed" and item["action"] == "redirect"
               for item in state["monitor"])


def test_waypost_adapter_accepts_swarms_messages_and_rejects_empty_choices(tmp_path):
    store = RunStore(tmp_path)
    state = {"calls": 0}
    budget = Budget(SwarmConfig(), state, store)
    requests = []
    def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        return httpx.Response(200, json={"choices": []})
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        llm = WaypostLLM(SwarmConfig(), budget, store, "run:worker", "system", client)
        with pytest.raises(ValueError, match="no choices"):
            llm.run(task=None, messages=[{"role": "user", "content": "the task"}])
    assert requests[0]["messages"] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "the task"},
    ]


def test_real_swarms_uses_custom_adapter(tmp_path, monkeypatch):
    monkeypatch.setenv("SWARMS_TELEMETRY_ON", "false")
    pytest.importorskip("swarms")
    store = RunStore(tmp_path)
    config = SwarmConfig()
    budget = Budget(config, {"calls": 0}, store)
    calls = []
    def post(self, url, **kwargs):
        calls.append(kwargs["json"])
        return httpx.Response(200, request=httpx.Request("POST", url),
            json={"choices": [{"message": {"content": '{"ok":true}'}, "finish_reason": "stop"}]})
    monkeypatch.setattr(httpx.Client, "post", post)
    backend = SwarmsBackend(config, budget, store)
    assert backend.ask("test", "Only JSON", "test task") == '{"ok":true}'
    assert len(calls) == 1 and calls[0]["model"] == "auto"
    assert any(message["role"] == "user" and "test task" in message["content"]
               for message in calls[0]["messages"])


def test_reviewer_cannot_write(tmp_path):
    backend = script(**{"r1:audit": [{"kind": "tool", "tool": "write_file", "arguments": {"path": "oops.txt", "content": "x"}}, final("checked")]})
    state = SwarmEngine(tmp_path, backend_factory=backend.factory).run("Task")
    assert state["status"] == "completed"
    assert not (tmp_path / "workspace/artifacts/r1/audit/oops.txt").exists()
    assert "Reviewer only has read access" in state["records"]["r1:audit"]["history"][1]["observation"]


def test_interrupted_tool_requires_acknowledgment(tmp_path):
    backend = script(**{"r1:synthesis": [RuntimeError("outage"), final("recovered")]})
    engine = SwarmEngine(tmp_path, backend_factory=backend.factory)
    state = engine.run("Task")
    state["records"]["r1:a"]["pending_tool"] = {"tool": "write_file"}
    engine.store.save(state)
    with pytest.raises(RuntimeError, match="outcome is unknown"):
        engine.run(resume=True)
    assert engine.run(resume=True, acknowledge_interrupted_tools=True)["status"] == "completed"


def test_concurrent_reservations_do_not_overspend(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    budget = Budget(SwarmConfig(max_calls=3), {"calls": 0}, RunStore(tmp_path))
    def reserve(_):
        try:
            budget.reserve()
            return True
        except BudgetExceeded:
            return False
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert sum(pool.map(reserve, range(20))) == 3
    assert budget.state["calls"] == 3


def test_run_lock_prevents_concurrent_resume(tmp_path):
    one, two = RunStore(tmp_path), RunStore(tmp_path)
    with one.exclusive():
        with pytest.raises(RuntimeError, match="already active"):
            with two.exclusive():
                pass


def test_step_exhaustion(tmp_path):
    backend = script(**{"r1:a": [{"kind": "tool", "tool": "list_files", "arguments": {}}]})
    state = SwarmEngine(tmp_path, SwarmConfig(max_steps=1), backend.factory).run("Task")
    assert state["status"] == "budget_exhausted"


def test_http_error_does_not_become_a_swarms_answer(tmp_path, monkeypatch):
    monkeypatch.setenv("SWARMS_TELEMETRY_ON", "false")
    pytest.importorskip("swarms")
    store = RunStore(tmp_path)
    config = SwarmConfig()
    budget = Budget(config, {"calls": 0}, store)
    def post(self, url, **kwargs):
        return httpx.Response(503, request=httpx.Request("POST", url))
    monkeypatch.setattr(httpx.Client, "post", post)
    with pytest.raises(httpx.HTTPStatusError):
        SwarmsBackend(config, budget, store).ask("test", "Only JSON", "test")
    assert budget.state["calls"] == 1


def test_real_swarms_preserves_budget_exception(tmp_path, monkeypatch):
    monkeypatch.setenv("SWARMS_TELEMETRY_ON", "false")
    pytest.importorskip("swarms")
    store = RunStore(tmp_path)
    config = SwarmConfig(max_calls=1)
    budget = Budget(config, {"calls": 1}, store)
    with pytest.raises(BudgetExceeded):
        SwarmsBackend(config, budget, store).ask("test", "Only JSON", "test")


def test_resume_uses_current_router_instead_of_saved_one(tmp_path):
    backend = script(**{"supervisor": [plan(), plan()],
                        "r2:a": [final("changed")], "r2:synthesis": [final("updated")],
                        "r2:audit": [final("checked")],
                        "review-verdict": [{"passed": True, "findings": [], "repair": None},
                                           {"passed": True, "findings": [], "repair": None}]})
    engine = SwarmEngine(tmp_path, SwarmConfig(base_url="http://127.0.0.1:8081/v1"),
                         backend_factory=backend.factory)
    assert engine.run("Original task")["status"] == "completed"
    engine.store.update_control(message="Change the result")
    resumed = SwarmEngine(tmp_path, backend_factory=backend.factory)
    state = resumed.run(resume=True, base_url="http://127.0.0.1:8080/v1")
    assert state["status"] == "completed"
    assert resumed.config.base_url == "http://127.0.0.1:8080/v1"
    assert RunStore(tmp_path).load()["config"]["base_url"] == "http://127.0.0.1:8080/v1"
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert any(e["event"] == "base_url_changed" and e["previous"].endswith(":8081/v1") for e in events)
