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
