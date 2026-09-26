"""Swarm v2: the coordinator decides; board with volunteers; debate."""
import json

from waypost.swarm import SwarmConfig, SwarmEngine
from waypost.swarm.llm import Answer


def ledger(*roles, plan=("write it",)):
    return {"acceptance": ["It works"], "facts": [], "guesses": [], "plan": list(plan),
            "team": [{"role": r, "strengths": r} for r in (roles or ("developer",))]}


def step(action, *, progress=True, looping=False, done=False, width=1, requests=(), question=None,
         reasoning="because"):
    return {"is_done": done, "progress": progress, "looping": looping, "reasoning": reasoning,
            "action": action, "width": width, "question": question,
            "requests": [{"id": r, "need": f"do {r}", "why": "", "done_when": "done"} for r in requests]}


def final(text="done"):
    return {"kind": "final", "answer": text}


PASS = {"passed": True, "findings": [], "repair": None}


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
    return {"coordinator-plan": [ledger()],
            "coordinator": [step("post", requests=["build"]), step("finish", done=True)],
            "r1:build": [final("built it")],
            "r1:synthesis": [final("the deliverable")],
            "r1:audit-1": [final("checked")],
            "review-verdict": [PASS], **extra}


def test_simple_task_takes_few_calls(tmp_path):
    state, backend, events = run(tmp_path, simple())
    assert state["status"] == "completed" and state["draft"] == "the deliverable"
    # plan, 2 coordinator steps, worker, synthesis, audit, 1 reviewer
    assert state["calls"] == 7, backend.calls
    assert backend.calls.count("r1:synthesis") == 1
    assert not any(c.startswith("volunteer") for c in backend.calls)  # a team of one just works


def test_request_goes_to_the_most_confident_volunteer(tmp_path):
    state, backend, events = run(tmp_path, simple(**{
        "coordinator-plan": [ledger("backend", "frontend")],
        "volunteer:backend": [{"claims": [{"request": "build", "confidence": 0.4, "approach": "flask"}]}],
        "volunteer:frontend": [{"claims": [{"request": "build", "confidence": 0.9, "approach": "react"}]}]}))
    assert state["status"] == "completed"
    assert state["requests"][0]["by"] == "frontend" and state["requests"][0]["approach"] == "react"


def test_unclaimed_request_goes_back_to_the_coordinator(tmp_path):
    state, backend, events = run(tmp_path, {
        "coordinator-plan": [ledger("a", "b")],
        "coordinator": [step("post", requests=["x"]), step("post", requests=["y"]),
                        step("finish", done=True)],
        "volunteer:a": [{"claims": []}, {"claims": [{"request": "y", "confidence": 0.8, "approach": "ok"}]}],
        "volunteer:b": [{"claims": []}, {"claims": []}],
        "r1:y": [final("y done")], "r1:synthesis": [final("d")], "r1:audit-1": [final("ok")],
        "review-verdict": [PASS]})
    assert state["status"] == "completed"
    assert state["requests"][0]["status"] == "unclaimed"
    assert any("Nobody on the team took request x" in e["text"] for e in state["board"])
    assert "Nobody on the team took request x" in "".join(
        p for p in [json.dumps(state["board"])])


def _debate(votes_round2, judge=None):
    positions = [{"stance": "use sqlite", "argument": "simple", "confidence": 0.6},
                 {"stance": "use sqlite too", "argument": "fine", "confidence": 0.5},
                 {"stance": "use a JSON file", "argument": "no dependency, 50 rows", "confidence": 0.7}]
    script = simple(**{
        "coordinator": [step("deliberate", question="storage?", width=3),
                        step("post", requests=["build"]), step("finish", done=True)],
        "deliberate": [positions[0]], "deliberate#2": [positions[1]], "deliberate#3": [positions[2]],
        "deliberate-r2": [votes_round2[0]], "deliberate-r2#2": [votes_round2[1]],
        "deliberate-r2#3": [votes_round2[2]],
        "deliberate-r3": [votes_round2[0]], "deliberate-r3#2": [votes_round2[1]],
        "deliberate-r3#3": [votes_round2[2]]})
    if judge is not None:
        script["debate-judge"] = [judge]
    return script


def _vote(n, persuaded=""):
    return {"stance": "-", "argument": "-", "confidence": 0.8, "supports": n, "persuaded_by": persuaded}


def test_minority_with_the_stronger_argument_can_win(tmp_path):
    state, backend, events = run(tmp_path, _debate([_vote(2, "no dependency"), _vote(2, "50 rows"), _vote(2)]))
    ruling = next(e for e in events if e["event"] == "ruling")
    assert ruling["stance"] == "use a JSON file" and ruling["why"] == "consensus"
    assert any(e["kind"] == "decision" and "JSON file" in e["text"] for e in state["board"])
    assert "deliberate-r3" not in backend.calls  # consensus after round 2: no more rounds


def test_no_consensus_the_judge_rules_by_argument(tmp_path):
    state, backend, events = run(tmp_path, _debate([_vote(0), _vote(0), _vote(2)],
                                                   judge={"chosen": 2, "why": "the only concrete argument"}))
    ruling = next(e for e in events if e["event"] == "ruling")
    assert ruling["stance"] == "use a JSON file" and "concrete" in ruling["why"]
    assert "deliberate-r3" in backend.calls and "deliberate-r3#3" in backend.calls


def test_two_steps_without_progress_force_a_replan(tmp_path):
    state, backend, events = run(tmp_path, simple(**{
        "coordinator-plan": [ledger(), ledger(plan=("try another way",))],
        "coordinator": [step("post", requests=["build"]),
                        step("post", progress=False, looping=True, requests=["build"]),
                        step("review", progress=False, looping=True, reasoning="rereading the same file"),
                        step("post", requests=["build2"]),
                        step("finish", done=True)],
        "r2:build2": [final("built again")], "r2:synthesis": [final("second")],
        "r2:audit-1": [final("ok")]}))
    assert state["status"] == "completed" and state["round"] == 2
    assert backend.calls.count("coordinator-plan") == 2
    assert any(e["kind"] == "dead_end" and "rereading" in e["text"] for e in state["board"])


def test_stall_fuse_delivers_what_there_is(tmp_path):
    stalled = [step("post", requests=["build"])] + [step("review", progress=False, looping=True)] * 20
    state, backend, events = run(tmp_path, simple(**{
        "coordinator-plan": [ledger()] * 10, "coordinator": stalled}))
    assert state["status"] == "completed"
    assert "Завершено автономно" in state["draft"] and "the deliverable" in state["draft"]


def test_a_worker_that_learns_nothing_new_hands_back(tmp_path):
    looping = [{"kind": "tool", "tool": "list_files", "arguments": {"path": "."}}] * 12
    state, backend, events = run(tmp_path, simple(**{"r1:build": looping,
                                                     "r1:synthesis": [final("partial")]}))
    assert state["requests"][0]["status"] == "failed"
    assert state["requests"][0]["result"].startswith("STUCK:")
    assert any(e["event"] == "worker_stuck" for e in events)
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
