"""Swarm v2: a council decides every fork; board with volunteers; debate."""
import json

from waypost.swarm import SwarmConfig, SwarmEngine
from waypost.swarm.llm import Answer


def ledger(*roles, plan=("write it",), council=2):
    roles = roles or ("developer", "critic")
    return {"acceptance": ["It works"], "facts": [], "guesses": [], "plan": list(plan),
            "team": [{"role": r, "strengths": r} for r in roles], "council": council}


def step(action, *, progress=True, looping=False, done=False, width=1, requests=(), question=None,
         reasoning="because"):
    reqs = []
    for r in requests:
        rid, _, dep = r.partition("<")
        reqs.append({"id": rid, "need": f"do {rid}", "why": "", "done_when": "done",
                     "depends_on": [dep] if dep else []})
    return {"is_done": done, "progress": progress, "looping": looping, "reasoning": reasoning,
            "action": action, "width": width, "question": question, "requests": reqs}


def final(text="done"):
    return {"kind": "final", "answer": text}


def vote(n, persuaded=""):
    return {"stance": "-", "argument": "-", "confidence": 0.8, "supports": n, "persuaded_by": persuaded}


def claim(request, approach):
    return {"claims": [{"request": request, "confidence": 0.5, "approach": approach}]}


PASS = {"passed": True, "findings": [], "repair": None}
NONE = {"claims": []}


def both(role, values, second=None):
    """The same answers from the council's two members (`role`, `role#2`)."""
    return {role: list(values), f"{role}#2": list(second if second is not None else values)}


class Script:
    FAMILIES = {"": "llama", "#2": "qwen", "#3": "gemma", "#4": "mistral", "#5": "openai"}

    def __init__(self, script):
        self.script = {k: list(v) for k, v in script.items()}
        self.calls = []

    def factory(self, config, budget, store):
        self.budget = budget
        return self

    def ask(self, role, system, prompt, schema=None, avoid_families=None):
        self.calls.append(role)
        if role == "progress-monitor" and role not in self.script:
            return Answer(json.dumps({"action": "continue", "reason": "ok"}), None)
        if not self.script.get(role):
            raise KeyError(role)
        value = self.script[role].pop(0)
        if isinstance(value, Exception):
            raise value
        self.budget.reserve()
        suffix = role[role.index("#"):] if "#" in role else ""
        return Answer(json.dumps(value), self.FAMILIES.get(suffix))


def run(tmp_path, script, **config):
    backend = Script(script)
    state = SwarmEngine(tmp_path, SwarmConfig(engine="coordinator", **config), backend.factory).run("Task")
    events = [json.loads(l) for l in (tmp_path / "events.jsonl").read_text().splitlines()]
    return state, backend, events


def simple(**extra):
    return {**both("coordinator-plan", [ledger()]),
            **both("coordinator-plan-vote", [vote(0)]),
            **both("coordinator", [step("post", requests=["build"]), step("finish", done=True)]),
            "volunteer:developer": [claim("build", "write main.py")],
            "volunteer:critic": [NONE],
            "r1:build": [final("built it")],
            "r1:synthesis": [final("FILES: main.py\n<the code>")],
            "r1:audit-1": [final("checked")],
            **both("review-verdict", [PASS]), **extra}


def test_plan_and_every_step_are_council_decisions(tmp_path):
    state, backend, events = run(tmp_path, simple())
    assert state["status"] == "completed" and state["draft"].startswith("FILES:")
    councils = [e for e in events if e["event"] == "council"]
    assert [c["subject"] for c in councils] == ["coordinator-plan", "coordinator", "coordinator"]
    assert all(len(c["families"]) == 2 for c in councils)
    # the steps agreed independently: no vote rounds for them
    assert "coordinator-vote" not in backend.calls
    assert backend.calls.count("coordinator-plan-vote") == 1


def test_council_disagreement_is_settled_by_argument_not_headcount(tmp_path):
    state, backend, events = run(tmp_path, simple(**{
        "coordinator-plan": [ledger(plan=("big bang",))],
        "coordinator-plan#2": [ledger(plan=("small steps",))],
        "coordinator-plan-vote": [vote(0), vote(0)], "coordinator-plan-vote#2": [vote(1), vote(1)],
        "coordinator-plan-judge": [{"chosen": 1, "why": "smaller steps are verifiable"}]}))
    plan = next(e for e in events if e["event"] == "council" and e["subject"] == "coordinator-plan")
    assert plan["chosen"] == 1 and "verifiable" in plan["why"]
    assert state["ledger"]["plan"] == ["small steps"]


def test_a_member_persuaded_by_an_argument_ends_the_vote(tmp_path):
    state, backend, events = run(tmp_path, simple(**{
        "coordinator-plan": [ledger(plan=("A",))], "coordinator-plan#2": [ledger(plan=("B",))],
        "coordinator-plan-vote": [vote(1, "B tests first")], "coordinator-plan-vote#2": [vote(1)]}))
    plan = next(e for e in events if e["event"] == "council" and e["subject"] == "coordinator-plan")
    assert plan["why"] == "consensus" and state["ledger"]["plan"] == ["B"]
    assert "coordinator-plan-judge" not in backend.calls


def test_contested_request_goes_to_the_better_argued_approach(tmp_path):
    state, backend, events = run(tmp_path, simple(**{
        "volunteer:developer": [{"claims": [{"request": "build", "confidence": 1.0, "approach": "just do it"}]}],
        "volunteer:critic": [{"claims": [{"request": "build", "confidence": 0.3,
                                          "approach": "tests first, then pygame loop"}]}],
        "volunteer-judge": [{"assignments": [{"request": "build", "role": "critic",
                                              "why": "concrete and testable"}]}]}))
    request = state["requests"][0]
    assert request["by"] == "critic" and request["why_chosen"] == "concrete and testable"


def test_dependent_request_waits_for_its_dependency(tmp_path):
    state, backend, events = run(tmp_path, simple(**{
        **both("coordinator", [step("post", requests=["build", "review<build"]), step("finish", done=True)]),
        "volunteer:developer": [claim("build", "write it"), NONE],
        "volunteer:critic": [NONE, claim("review", "read it")],
        "r1:review": [final("reviewed the built file")]}))
    order = [e["request"] for e in events if e["event"] == "request_done"]
    assert order == ["build", "review"]
    assert state["requests"][1]["by"] == "critic"


def test_a_request_nobody_takes_is_settled_by_the_council(tmp_path):
    """"Decide the storage" is a fork, not work: nobody volunteers, so the
    council debates it and its ruling is the request's result."""
    json_file = {"stance": "a JSON file", "argument": "stdlib, human-readable", "confidence": 0.8}
    state, backend, events = run(tmp_path, simple(**{
        **both("coordinator", [step("post", requests=["decide_storage"]),
                               step("post", requests=["build<decide_storage"]), step("finish", done=True)]),
        "volunteer:developer": [NONE, claim("build", "write it")],
        "volunteer:critic": [NONE, NONE],
        **both("deliberate", [json_file]), **both("deliberate-vote", [vote(0)])}))
    decided = state["requests"][0]
    assert decided["status"] == "done" and decided["by"] == "council"
    assert decided["result"] == "Council decision: a JSON file"
    assert state["status"] == "completed" and state["requests"][1]["status"] == "done"


def test_reposting_a_request_keeps_its_id_and_skips_done_work(tmp_path):
    state, backend, events = run(tmp_path, simple(**{
        **both("coordinator", [step("post", requests=["build"]),
                               step("post", requests=["build", "docs<build"]), step("finish", done=True)]),
        "volunteer:developer": [claim("build", "write it"), NONE],
        "volunteer:critic": [NONE, claim("docs", "write README")],
        "r1:docs": [final("readme")]}))
    assert [r["id"] for r in state["requests"]] == ["build", "docs"]
    assert backend.calls.count("r1:build") == 1
    assert any(e["event"] == "request_already_done" and e["request"] == "build" for e in events)


def test_deliberate_without_question_uses_the_reasoning():
    from waypost.swarm.models import CoordinatorStep
    s = CoordinatorStep.model_validate(step("deliberate", reasoning="JSON or SQLite?"))
    assert s.question == "JSON or SQLite?"


def test_fork_in_the_work_is_debated(tmp_path):
    stance_a = {"stance": "sqlite", "argument": "simple", "confidence": 0.6}
    stance_b = {"stance": "a JSON file", "argument": "no dependency, 50 rows", "confidence": 0.7}
    state, backend, events = run(tmp_path, simple(**{
        **both("coordinator", [step("deliberate", question="storage?"), step("post", requests=["build"]),
                               step("finish", done=True)]),
        "deliberate": [stance_a], "deliberate#2": [stance_b],
        "deliberate-vote": [vote(1, "no dependency")], "deliberate-vote#2": [vote(1)]}))
    ruling = next(e for e in events if e["event"] == "ruling")
    assert ruling["stance"] == "a JSON file"
    assert any(e["kind"] == "decision" and "JSON file" in e["text"] for e in state["board"])


def test_two_stalled_steps_make_the_council_replan(tmp_path):
    stalled = step("review", progress=False, looping=True, reasoning="rereading the same file")
    state, backend, events = run(tmp_path, simple(**{
        **both("coordinator-plan", [ledger(), ledger(plan=("try another way",))]),
        **both("coordinator-plan-vote", [vote(0), vote(0)]),
        **both("coordinator", [step("post", requests=["build"]), stalled, stalled,
                               step("finish", done=True)])}))
    assert state["status"] == "completed" and state["round"] == 2
    assert state["ledger"]["plan"] == ["try another way"]
    assert any(e["kind"] == "dead_end" and "rereading" in e["text"] for e in state["board"])


def test_stall_fuse_delivers_what_there_is(tmp_path):
    stalled = step("review", progress=False, looping=True)
    state, backend, events = run(tmp_path, simple(**{
        **both("coordinator-plan", [ledger()] * 10), **both("coordinator-plan-vote", [vote(0)] * 10),
        **both("coordinator", [step("post", requests=["build"])] + [stalled] * 20)}))
    assert state["status"] == "completed"
    assert "Завершено автономно" in state["draft"] and "FILES:" in state["draft"]


def test_one_family_alive_decides_alone_and_says_so(tmp_path):
    script = simple()
    for role in ("coordinator-plan#2", "coordinator-plan-vote", "coordinator-plan-vote#2",
                 "coordinator#2", "review-verdict#2"):
        script.pop(role)
    state, backend, events = run(tmp_path, script)
    assert state["status"] == "completed"
    assert all(c["why"] == "only one family answered" for c in events if c["event"] == "council")


def test_a_worker_that_learns_nothing_new_hands_back(tmp_path):
    looping = [{"kind": "tool", "tool": "list_files", "arguments": {"path": "."}}] * 12
    state, backend, events = run(tmp_path, simple(**{"r1:build": looping}))
    assert state["requests"][0]["status"] == "failed"
    assert state["requests"][0]["result"].startswith("STUCK:")
    assert backend.calls.count("r1:build") < 10


def test_runs_saved_before_v2_resume_on_the_fixed_phases(tmp_path):
    from waypost.swarm.store import RunStore
    store = RunStore(tmp_path)
    config = SwarmConfig(engine="pipeline").model_dump()
    config.pop("engine")
    store.save({"version": 1, "task": "t", "config": config, "status": "interrupted", "phase": "plan",
                "round": 1, "calls": 0, "elapsed_seconds": 0, "records": {}, "reviews": [], "draft": "",
                "messages": [], "monitor": [], "monitor_guidance": ""})
    backend = Script({"supervisor": [{"acceptance": ["a"], "tasks": [
        {"id": "a", "role": "r", "instruction": "i", "depends_on": []}]}],
        "r1:a": [final()], "r1:synthesis": [final("x")], "r1:audit": [final("y")],
        "review-verdict": [PASS]})
    state = SwarmEngine(tmp_path, backend_factory=backend.factory).run(resume=True)
    assert state["config"]["engine"] == "pipeline" and "supervisor" in backend.calls


def test_python_is_on_by_default_and_can_be_switched_off(tmp_path, monkeypatch):
    from waypost.swarm import cli
    seen = []
    monkeypatch.setattr(cli.SwarmEngine, "run", lambda self, task=None, **kw: seen.append(self.config)
                        or {"status": "completed"})
    cli.main(["run", "--task", "t", "--run-dir", str(tmp_path / "a")])
    cli.main(["run", "--task", "t", "--run-dir", str(tmp_path / "b"), "--no-python"])
    assert seen[0].allow_python is True and seen[1].allow_python is False


# ------------------------------------------------------- sandboxed python

import os as _os
import sys as _sys
from pathlib import Path as _Path

import pytest as _pytest

_has_sandbox = _sys.platform == "darwin" and _os.access("/usr/bin/sandbox-exec", _os.X_OK)


def _run_py(tmp_path, code):
    from waypost.swarm.tools import WorkspaceTools
    tools = WorkspaceTools(tmp_path / "ws", "artifacts/r1/a", allow_python=True)
    return json.loads(tools.execute("run_python", {"code": code}, timeout=20))


@_pytest.mark.skipif(not _has_sandbox, reason="needs macOS sandbox-exec")
def test_sandboxed_python_works_in_its_own_directory(tmp_path):
    out = _run_py(tmp_path, "open('x.txt','w').write('1'); print(6*7)")
    assert out["exit_code"] == 0 and out["output"].strip() == "42"
    assert (tmp_path / "ws/artifacts/r1/a/x.txt").read_text() == "1"


@_pytest.mark.skipif(not _has_sandbox, reason="needs macOS sandbox-exec")
def test_sandboxed_python_cannot_write_outside_read_home_or_use_network(tmp_path):
    outside = tmp_path / "outside.txt"
    out = _run_py(tmp_path, f"open({str(outside)!r},'w').write('x')")
    assert out["exit_code"] != 0 and not outside.exists()
    out = _run_py(tmp_path, f"import os; print(os.listdir({str(_Path.home())!r}))")
    assert out["exit_code"] != 0 and "Operation not permitted" in out["output"]
    out = _run_py(tmp_path, "import socket; socket.create_connection(('1.1.1.1', 443), timeout=3)")
    assert out["exit_code"] != 0
    # what the code starts is confined too
    out = _run_py(tmp_path, f"import subprocess; subprocess.run(['/bin/ls', {str(_Path.home())!r}], check=True)")
    assert out["exit_code"] != 0


def test_no_sandbox_means_no_execution(tmp_path, monkeypatch):
    from waypost.swarm.tools import WorkspaceTools
    monkeypatch.setattr(WorkspaceTools, "SANDBOX_EXEC", "/nonexistent/sandbox-exec")
    tools = WorkspaceTools(tmp_path, "artifacts/r1/a", allow_python=True)
    with _pytest.raises(ValueError, match="no OS sandbox"):
        tools.execute("run_python", {"code": "print(1)"})
