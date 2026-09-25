"""Collective swarm, stage 1: members come from different model families."""
import json

import httpx
import pytest

from waypost.classify import classify_l0
from waypost.ledger import Ledger
from waypost.breaker import CircuitBreaker
from waypost.registry import Offering, Registry
from waypost.router import Router
from waypost.schemas import Capability, ChatMessage, ChatRequest
from waypost.swarm import SwarmConfig, SwarmEngine
from waypost.swarm.llm import Answer, Budget, SwarmsBackend
from waypost.swarm.models import ProgressDecision
from waypost.swarm.store import RunStore

CAPS = {Capability.JSON, Capability.STREAM}


# ------------------------------------------------------------------ router


def _offering(model_id, score):
    return Offering(provider=model_id.split("-")[0], model_id=model_id,
                    base_url=f"http://{model_id}/v1", caps=CAPS, quality_score=score,
                    limit_rpd=100)


def test_router_puts_avoided_families_last(tmp_path):
    offerings = [_offering("llama-70b", 0.95), _offering("qwen-32b", 0.8),
                 _offering("gemma-27b", 0.7)]
    ledger = Ledger(str(tmp_path / "r.db"))
    for o in offerings:
        ledger.register(o)
    router = Router(Registry(offerings), ledger, CircuitBreaker(), stochastic=False)

    def first(avoid):
        req = ChatRequest(messages=[ChatMessage(role="user", content="hi")],
                          avoid_families=avoid)
        return [c.offering.model_id for c in router.plan(req, classify_l0(req))]

    assert first(None)[0] == "llama-70b"
    ladder = first(["llama"])
    assert ladder[0] == "qwen-32b"
    assert ladder[-1] == "llama-70b"  # last, not gone
    assert first(["llama", "qwen", "gemma"])  # everything avoided: still a plan


def test_avoid_families_is_not_sent_upstream():
    req = ChatRequest(messages=[ChatMessage(role="user", content="hi")],
                      avoid_families=["llama"], output_schema={"type": "object"})
    body = req.provider_payload("m")
    assert "avoid_families" not in body and "output_schema" not in body


# ------------------------------------------------------------------- swarm


class Families:
    """Answers each call from the next scripted family, recording what the
    caller asked it to avoid."""

    def __init__(self, families, fail_at=None):
        self.families = list(families)
        self.fail_at = fail_at
        self.calls = []

    def factory(self, config, budget, store):
        self.budget = budget
        return self

    def ask(self, role, system, prompt, schema=None, avoid_families=None):
        self.budget.reserve()
        self.calls.append((role, list(avoid_families or [])))
        if self.fail_at is not None and len(self.calls) - 1 == self.fail_at:
            raise RuntimeError("router refused")
        body = json.dumps({"action": "continue", "reason": "ok"})
        return Answer(body, self.families[len(self.calls) - 1])


def _engine(tmp_path, backend, **config):
    engine = SwarmEngine(tmp_path, SwarmConfig(**config), backend.factory)
    engine.state = {"task": "t", "calls": 0, "round": 1, "phase": "plan", "records": {},
                    "reviews": [], "status": "running"}
    engine.budget = Budget(engine.config, engine.state, engine.store)
    engine.backend = backend.factory(engine.config, engine.budget, engine.store)
    return engine


def _ask(engine):
    return engine._collective("judge", "Decide", "ctx", ProgressDecision)


def test_each_member_is_told_the_families_already_heard(tmp_path):
    backend = Families(["llama", "qwen", "gemma"])
    members = _ask(_engine(tmp_path, backend))
    assert [f for _, f in members] == ["llama", "qwen", "gemma"]
    assert backend.calls == [("judge", []), ("judge#2", ["llama"]),
                             ("judge#3", ["llama", "qwen"])]


def test_collective_stops_when_only_one_family_is_alive(tmp_path):
    backend = Families(["qwen", "qwen", "qwen"])  # e.g. only the local tail answers
    members = _ask(_engine(tmp_path, backend))
    assert len(members) == 1
    assert len(backend.calls) == 2  # tried once more, then stopped


def test_a_later_failure_narrows_instead_of_failing(tmp_path):
    backend = Families(["llama", "qwen", "gemma"], fail_at=1)
    members = _ask(_engine(tmp_path, backend))
    assert [f for _, f in members] == ["llama"]


def test_first_member_failure_is_the_callers(tmp_path):
    backend = Families(["llama"], fail_at=0)
    with pytest.raises(RuntimeError):
        _ask(_engine(tmp_path, backend))


def test_width_one_is_the_old_single_call(tmp_path):
    backend = Families(["llama", "qwen"])
    members = _ask(_engine(tmp_path, backend, collective_width=1))
    assert len(members) == 1 and backend.calls == [("judge", [])]


def test_client_sends_avoid_families_and_reports_the_family(tmp_path, monkeypatch):
    monkeypatch.setenv("SWARMS_TELEMETRY_ON", "false")
    pytest.importorskip("swarms")
    store = RunStore(tmp_path)
    config = SwarmConfig()
    budget = Budget(config, {"calls": 0}, store)
    sent = []

    def post(self, url, **kwargs):
        sent.append(kwargs["json"])
        return httpx.Response(200, request=httpx.Request("POST", url), json={
            "choices": [{"message": {"content": '{"ok":true}'}, "finish_reason": "stop"}],
            "router": {"provider": "ollama", "model": "cloud/gpt-oss:120b"}})

    monkeypatch.setattr(httpx.Client, "post", post)
    answer = SwarmsBackend(config, budget, store).ask("t", "Only JSON", "task",
                                                      avoid_families=["llama"])
    assert sent[0]["avoid_families"] == ["llama"]
    assert answer == '{"ok":true}' and answer.family == "openai"


# -------------------------------------------------------- stage 2: panel


def _plan(tasks=None):
    return {"acceptance": ["Deliver an evidenced answer"], "tasks": tasks or [
        {"id": "a", "role": "analyst", "instruction": "Compute", "depends_on": []}]}


def _final(text="done"):
    return {"kind": "final", "answer": text}


PASS = {"passed": True, "findings": [], "repair": None}


def _fail(finding):
    return {"passed": False, "findings": [finding], "repair": _plan()}


class FamilyScript:
    """Scripted backend whose answers carry a model family per call index:
    role -> list of values; `role#2` etc. answer from other families."""

    FAMILIES = {"": "llama", "#2": "qwen", "#3": "gemma"}

    def __init__(self, script):
        self.script = {k: list(v) for k, v in script.items()}
        self.calls = []

    def factory(self, config, budget, store):
        self.budget = budget
        return self

    def ask(self, role, system, prompt, schema=None, avoid_families=None):
        self.calls.append(role)
        self.avoided = getattr(self, "avoided", {})
        self.avoided[role] = list(avoid_families or [])
        self.prompts = getattr(self, "prompts", {})
        self.prompts.setdefault(role, []).append(prompt)
        if role == "progress-monitor" and role not in self.script:
            return Answer(json.dumps({"action": "continue", "reason": "ok"}), None)
        if role not in self.script or not self.script[role]:
            raise KeyError(role)
        self.budget.reserve()
        suffix = role[role.index("#"):] if "#" in role else ""
        return Answer(json.dumps(self.script[role].pop(0)), self.FAMILIES.get(suffix))


def _run(tmp_path, script, **config):
    base = {"supervisor": [_plan()], "r1:a": [_final()], "r1:synthesis": [_final("answer")],
            "r1:audit": [_final("evidence")]}
    backend = FamilyScript({**base, **script})
    state = SwarmEngine(tmp_path, SwarmConfig(**config), backend.factory).run("Task")
    events = [json.loads(l) for l in (tmp_path / "events.jsonl").read_text().splitlines()]
    return state, backend, events


def test_panel_majority_passes(tmp_path):
    state, backend, events = _run(tmp_path, {
        "review-verdict": [PASS], "review-verdict#2": [_fail("style")],
        "review-verdict#3": [PASS]})
    assert state["status"] == "completed" and state["round"] == 1
    panel = next(e for e in events if e["event"] == "review_panel")
    assert panel["passed"] and len(panel["votes"]) == 3
    assert "review-consensus" not in backend.calls


def test_panel_confirmed_finding_triggers_repair(tmp_path):
    state, backend, events = _run(tmp_path, {
        "review-verdict": [_fail("no tests"), PASS], "review-verdict#2": [_fail("tests missing"), PASS],
        "review-verdict#3": [PASS, PASS],
        "review-consensus": [{"confirmed": ["no tests"], "repair": _plan()}],
        "r2:a": [_final()], "r2:synthesis": [_final("fixed")], "r2:audit": [_final("ok")]})
    assert state["status"] == "completed" and state["round"] == 2
    assert state["reviews"][0] == {"passed": False, "findings": ["no tests"],
                                   "repair": _plan()}


def test_panel_disagreeing_objections_do_not_block(tmp_path):
    state, backend, events = _run(tmp_path, {
        "review-verdict": [_fail("naming")], "review-verdict#2": [_fail("colours")],
        "review-verdict#3": [PASS],
        "review-consensus": [{"confirmed": [], "repair": None}]})
    assert state["status"] == "completed" and state["round"] == 1


def test_lone_objection_is_recorded_not_enforced(tmp_path):
    state, backend, events = _run(tmp_path, {
        "review-verdict": [_fail("I dislike it")], "review-verdict#2": [PASS],
        "review-verdict#3": [PASS]})
    assert state["status"] == "completed"
    assert state["reviews"][0]["findings"] == ["(не подтверждено панелью) I dislike it"]


def test_panel_off_is_a_single_reviewer(tmp_path):
    state, backend, events = _run(tmp_path, {"review-verdict": [PASS]}, review_panel=False)
    assert state["status"] == "completed"
    assert backend.calls.count("review-verdict") == 1
    assert not any(c.startswith("review-verdict#") for c in backend.calls)


# ------------------------------------------------ stage 3: proposals, judge


def _tasks(tag):
    return _plan([{"id": tag, "role": tag, "instruction": "Do " + tag, "depends_on": []}])


def _choice(chosen, merged=None):
    return {"critiques": [{"proposal": 0, "strengths": ["s"], "flaws": ["f"]}],
            "chosen": chosen, "merged": merged}


def test_judge_picks_among_independent_plans(tmp_path):
    state, backend, events = _run(tmp_path, {
        "supervisor": [_tasks("x")], "supervisor#2": [_tasks("y")], "supervisor#3": [_tasks("z")],
        "judge-plan": [_choice(1)],
        "r1:y": [_final()], "review-verdict": [PASS]}, review_panel=False)
    assert state["status"] == "completed"
    assert [t["id"] for t in state["plan"]["tasks"]] == ["y"]
    assert backend.avoided["judge-plan"] == ["llama", "qwen", "gemma"]  # not an author
    judged = next(e for e in events if e["event"] == "proposals_judged")
    assert judged["subject"] == "plan" and judged["chosen"] == 1


def test_judge_can_merge_plans(tmp_path):
    state, backend, events = _run(tmp_path, {
        "supervisor": [_tasks("x")], "supervisor#2": [_tasks("y")],
        "judge-plan": [_choice(0, merged=_tasks("m"))],
        "r1:m": [_final()], "review-verdict": [PASS]}, review_panel=False)
    assert [t["id"] for t in state["plan"]["tasks"]] == ["m"]


def test_failed_judge_falls_back_to_the_first_plan(tmp_path):
    state, backend, events = _run(tmp_path, {
        "supervisor": [_tasks("x")], "supervisor#2": [_tasks("y")],
        "r1:x": [_final()], "review-verdict": [PASS]}, review_panel=False)
    assert state["status"] == "completed"
    assert [t["id"] for t in state["plan"]["tasks"]] == ["x"]
    assert any(e["event"] == "judge_failed" for e in events)


def test_judge_picks_among_independent_drafts(tmp_path):
    state, backend, events = _run(tmp_path, {
        "r1:synthesis": [_final("draft one")], "r1:synthesis#2": [_final("draft two")],
        "judge-draft": [_choice(1)], "review-verdict": [PASS]}, review_panel=False)
    assert state["draft"] == "draft two"
    assert (tmp_path / "result.md").read_text() == "draft two"


def test_width_one_has_no_judge(tmp_path):
    state, backend, events = _run(tmp_path, {"review-verdict": [PASS]}, collective_width=1)
    assert state["status"] == "completed"
    assert not any(c.startswith("judge") or "#" in c for c in backend.calls)


@pytest.mark.asyncio
async def test_fanout_returns_only_a_verified_answer():
    from unittest.mock import AsyncMock, MagicMock
    from waypost.ensemble import fanout_ensemble
    from waypost.router import Candidate
    from waypost.schemas import ChatResponse, RequestProfile, Tier

    def reply(text):
        return ChatResponse(id="x", model="m", choices=[
            {"index": 0, "message": {"role": "assistant", "content": text}}])

    executor = MagicMock()
    executor.execute = AsyncMock(side_effect=[reply("bad"), reply("good")])
    cands = [Candidate(offering=_offering(m, 0.9), score=1.0, reasons={})
             for m in ("qwen-7b", "llama-8b")]

    class Verifier:
        def verify(self, req, profile, body):
            return body["choices"][0]["message"]["content"] == "good", ""

    req = ChatRequest(messages=[ChatMessage(role="user", content="q")])
    profile = RequestProfile(task_probs={"chat": 1.0}, tier=Tier.M)
    res = await fanout_ensemble(executor, req, profile, cands, verifier=Verifier())
    assert res is not None and res.choices[0]["message"]["content"] == "good"

    executor.execute = AsyncMock(side_effect=[reply("bad"), reply("bad")])
    assert await fanout_ensemble(executor, req, profile, cands, verifier=Verifier()) is None


# ------------------------------------------------------------ stage 4: board


def _board(state):
    return [(e["kind"], e["text"]) for e in state["board"]]


def test_specialist_post_reaches_the_next_agent(tmp_path):
    state, backend, events = _run(tmp_path, {
        "r1:a": [{"kind": "tool", "tool": "board_post",
                  "arguments": {"kind": "fact", "text": "the API caps at 50 rows"}}, _final()],
        "review-verdict": [PASS]}, collective_width=1)
    assert ("fact", "the API caps at 50 rows") in _board(state)
    assert "the API caps at 50 rows" in backend.prompts["r1:synthesis"][0]


def test_abandoned_approach_is_a_dead_end_the_next_plan_sees(tmp_path):
    state, backend, events = _run(tmp_path, {
        "supervisor": [_plan(), _plan()],
        "progress-monitor": [{"action": "replan", "reason": "brute force is too slow"}]
                            + [{"action": "continue", "reason": "ok"}] * 5,
        "r2:a": [_final()], "r2:synthesis": [_final("answer")], "r2:audit": [_final("ok")],
        "review-verdict": [PASS]}, collective_width=1)
    assert state["status"] == "completed" and state["round"] == 2
    dead = [t for k, t in _board(state) if k == "dead_end"]
    assert dead and "brute force is too slow" in dead[0]
    assert "brute force is too slow" in backend.prompts["supervisor"][1]


def test_swarm_assumptions_are_in_the_result(tmp_path):
    state, backend, events = _run(tmp_path, {
        "supervisor": [_plan(), _plan()],
        "progress-monitor": [{"action": "needs_input", "reason": "units unspecified"}]
                            + [{"action": "continue", "reason": "ok"}] * 5,
        "r2:a": [_final()], "r2:synthesis": [_final("answer")], "r2:audit": [_final("ok")],
        "review-verdict": [PASS]}, collective_width=1)
    result = (tmp_path / "result.md").read_text()
    assert result.startswith("answer") and "Допущения роя" in result and "units unspecified" in result


def test_board_survives_resume(tmp_path):
    state, backend, events = _run(tmp_path, {
        "r1:a": [{"kind": "tool", "tool": "board_post",
                  "arguments": {"kind": "decision", "text": "use SQLite"}}, _final()],
        "review-verdict": [PASS]}, collective_width=1)
    saved = json.loads((tmp_path / "state.json").read_text())
    assert ("decision", "use SQLite") in _board(saved)


def test_board_view_is_bounded(tmp_path):
    engine = _engine(tmp_path, Families(["llama"]))
    engine.state["board"] = []
    for i in range(60):
        engine._post("fact", f"fact {i}", "t")
    engine._post("dead_end", "never again", "t")
    view = engine._board_view()
    assert sum(e["kind"] == "fact" for e in view) == 20
    assert view[-1]["text"] == "never again" and view[0]["text"] == "fact 40"


def test_board_off_adds_nothing(tmp_path):
    state, backend, events = _run(tmp_path, {"review-verdict": [PASS]},
                                  collective_width=1, board=False)
    assert state["board"] == []  # the plan decision was not posted
    assert '"board": []' in backend.prompts["r1:a"][0]
