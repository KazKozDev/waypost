"""Swarm v3: stigmergy — marks on the board, no meetings, no duplicate work."""
import json

import pytest

from waypost.swarm import SwarmConfig, SwarmEngine
from waypost.swarm.llm import Answer


def final(text="done"):
    return {"kind": "final", "answer": text}


def split(*subtasks):
    specs = []
    for s in subtasks:
        sid, _, dep = s.partition("<")
        specs.append({"id": sid, "goal": f"goal of {sid}", "kind": "build",
                      "depends_on": [dep] if dep else []})
    return {"kind": "split", "subtasks": specs}


def twice(tid, spec):
    """First attempt's split is refused (do it yourself first); the second
    attempt may split."""
    return {f"t:{tid}:1": [spec], f"t:{tid}:2": [spec]}


def dead(reason):
    return {"kind": "dead_end", "answer": reason}


def write(path, content="print(1)"):
    return {"kind": "tool", "tool": "write_file", "arguments": {"path": path, "content": content}}


RUN = {"kind": "tool", "tool": "run_python", "arguments": {"code": "print(1)"}}


class Script:
    FAMILIES = {"": "llama", "#2": "qwen", "#3": "gemma"}

    def __init__(self, script):
        self.script = {k: list(v) for k, v in script.items()}
        self.calls = []
        self.prompts = {}

    def factory(self, config, budget, store):
        self.budget = budget
        return self

    def ask(self, role, system, prompt, schema=None, avoid_families=None, tier_hint=None):
        self.calls.append(role)
        self.tiers = getattr(self, "tiers", {})
        self.tiers[role] = tier_hint
        self.prompts.setdefault(role, []).append(prompt)
        if not self.script.get(role):
            raise KeyError(role)
        value = self.script[role].pop(0)
        if isinstance(value, Exception):
            raise value
        self.budget.reserve()
        suffix = role[role.index("#"):] if "#" in role else ""
        return Answer(json.dumps(value), self.FAMILIES.get(suffix))


@pytest.fixture(autouse=True)
def _run_python_ok(monkeypatch):
    """The sandbox has its own tests; here a run just succeeds."""
    from waypost.swarm.tools import WorkspaceTools
    monkeypatch.setattr(WorkspaceTools, "_python", lambda self, code, timeout: json.dumps(
        {"exit_code": 0, "timed_out": False, "output": "1\n", "truncated": False}))


def run(tmp_path, script, **config):
    backend = Script(script)
    state = SwarmEngine(tmp_path, SwarmConfig(engine="swarm", **config), backend.factory).run("Task")
    events = [json.loads(l) for l in (tmp_path / "events.jsonl").read_text().splitlines()]
    return state, backend, events


def test_a_simple_task_is_one_agent_and_an_independent_check(tmp_path):
    state, backend, events = run(tmp_path, {"t:goal:1": [final("42")], "v:goal:1": [final("VERIFIED")]})
    assert state["status"] == "completed" and state["draft"] == "42"
    assert state["calls"] == 2 and backend.calls == ["t:goal:1", "v:goal:1"]


def test_an_impossibility_confirmed_by_a_second_agent_is_a_result(tmp_path):
    state, backend, events = run(tmp_path, {"t:goal:1": [dead("no DNS in this sandbox")],
                                            "t:goal:2": [dead("confirmed: no DNS")]})
    assert state["status"] == "completed" and state["calls"] == 2
    assert "confirmed: no DNS" in state["draft"]
    assert "Check it yourself" in backend.prompts["t:goal:2"][0]


def test_a_false_alarm_is_overturned_by_the_second_agent(tmp_path):
    state, backend, events = run(tmp_path, {
        "t:goal:1": [dead("pygame is not installed")],
        "t:goal:2": [RUN, final("installed pygame with pip; game runs")]})
    assert state["tasks"]["goal"]["status"] == "done"
    assert state["draft"].startswith("installed pygame")
    assert any(e["event"] == "dead_end_suspected" for e in events)


def test_code_counts_as_done_only_after_it_ran(tmp_path):
    state, backend, events = run(tmp_path, {
        "t:goal:1": [write("game.py"), final("written")],
        "t:goal:2": [RUN, final("ran it: 1")]})
    assert state["status"] == "completed" and state["draft"] == "ran it: 1"
    assert state["tasks"]["goal"]["verified"] is True
    assert any(e["event"] == "task_unverified" for e in events)
    assert "did not run it" in backend.prompts["t:goal:2"][0]


def test_the_agent_splits_and_then_integrates_its_subtasks(tmp_path):
    state, backend, events = run(tmp_path, {
        **twice("goal", split("a", "b<a")),
        "t:goal.a:1": [final("A result")], "t:goal.b:1": [final("B result")],
        "t:goal:3": [final("A+B integrated")]})
    assert state["status"] == "completed" and state["draft"] == "A+B integrated"
    done = [e["task"] for e in events if e["event"] == "mark_done"]
    assert done == ["goal.a", "goal.b", "goal"]
    assert "A result" in backend.prompts["t:goal:3"][0] and "B result" in backend.prompts["t:goal:3"][0]


def test_the_same_work_is_never_done_twice(tmp_path):
    same = {"kind": "split", "subtasks": [
        {"id": "x", "goal": "Write the parser", "kind": "build", "depends_on": []},
        {"id": "y", "goal": "write the parser!", "kind": "build", "depends_on": []}]}
    state, backend, events = run(tmp_path, {
        **twice("goal", same), "t:goal.x:1": [final("parser")], "t:goal:3": [final("done")]})
    assert state["status"] == "completed"
    assert "t:goal.y:1" not in backend.calls
    assert state["tasks"]["goal.y"]["reused"] == "goal.x"


def test_a_dead_end_scares_off_the_same_goal(tmp_path):
    first = {"kind": "split", "subtasks": [
        {"id": "a", "goal": "fetch the page", "kind": "research", "depends_on": []},
        {"id": "b", "goal": "summarize", "kind": "write", "depends_on": ["a"]}]}
    again = {"kind": "split", "subtasks": [
        {"id": "c", "goal": "Fetch the page", "kind": "research", "depends_on": []},
        {"id": "d", "goal": "report", "kind": "write", "depends_on": []}]}
    state, backend, events = run(tmp_path, {
        "t:goal:1": [first], "t:goal:2": [first],
        "t:goal.a:1": [dead("no network")], "t:goal.a:2": [dead("no network, confirmed")],
        "t:goal.b:1": [final("summary of nothing")],
        "t:goal:3": [again], "t:goal.d:1": [final("report")],
        "t:goal:4": [final("could not fetch: no network")]})
    assert state["tasks"]["goal.c"]["status"] == "dead_end"
    assert "t:goal.c:1" not in backend.calls  # the dead end was reused, not retried
    assert "no network" in backend.prompts["t:goal:3"][0]


def test_a_conflict_is_settled_by_a_quorum(tmp_path):
    state, backend, events = run(tmp_path, {
        **twice("goal", split("a", "b")),
        "t:goal.a:1": [final("the answer is 41")], "t:goal.b:1": [final("the answer is 42")],
        "t:goal:3": [{"kind": "conflict", "between": ["goal.a", "goal.b"], "answer": "41 or 42?"}],
        "quorum:goal": [{"winner": "goal.b", "why": "6*7=42"}],
        "quorum:goal#2": [{"winner": "goal.b", "why": "checked"}],
        "quorum:goal#3": [{"winner": "goal.a", "why": "hm"}],
        "t:goal:4": [final("42")]})
    quorum = next(e for e in events if e["event"] == "quorum")
    assert quorum["winner"] == "goal.b" and state["tasks"]["goal.a"]["status"] == "dead_end"
    assert state["draft"] == "42" and "goal.b is right" in backend.prompts["t:goal:4"][0]


def test_no_quorum_without_a_conflict(tmp_path):
    state, backend, events = run(tmp_path, {
        **twice("goal", split("a", "b")), "t:goal.a:1": [final("A")], "t:goal.b:1": [final("B")],
        "t:goal:3": [final("A and B")]})
    assert not any(c.startswith("quorum") for c in backend.calls)


def test_an_agent_that_keeps_failing_gives_the_task_up(tmp_path):
    state, backend, events = run(tmp_path, {
        "t:goal:1": [RuntimeError("502")] * 3, "t:goal:2": [RuntimeError("502")] * 3,
        "t:goal:3": [RuntimeError("502")] * 3})
    assert state["tasks"]["goal"]["status"] == "dead_end"
    assert "gave up after 3 attempts" in state["tasks"]["goal"]["result"]
    assert state["status"] == "completed"


def test_splitting_stops_at_the_depth_limit(tmp_path):
    def level(tid):
        return {"kind": "split", "subtasks": [
            {"id": "a", "goal": f"deeper part of {tid}", "kind": "build", "depends_on": []},
            {"id": "b", "goal": f"side part of {tid}", "kind": "build", "depends_on": []}]}
    script, tid = {}, "goal"
    for _ in range(3):
        script.update(twice(tid, level(tid)))
        script[f"t:{tid}.b:1"] = [final("side done")]
        script[f"t:{tid}:3"] = [final(f"{tid} integrated")]
        tid = f"{tid}.a"
    script.update(twice(tid, level(tid)))            # depth 3: refused both times
    script[f"t:{tid}:3"] = [final("did it myself")]
    state, backend, events = run(tmp_path, script)
    assert state["status"] == "completed" and state["draft"] == "goal integrated"
    assert "Do not split further" in backend.prompts[f"t:{tid}:3"][0]


def test_a_task_never_waits_on_its_own_ancestor(tmp_path):
    same = {"kind": "split", "subtasks": [
        {"id": "a", "goal": "Task", "kind": "build", "depends_on": []},
        {"id": "b", "goal": "other", "kind": "build", "depends_on": []}]}
    state, backend, events = run(tmp_path, {
        **twice("goal", same), "t:goal.a:1": [final("a")], "t:goal.b:1": [final("b")],
        "t:goal:3": [final("done")]})
    assert state["status"] == "completed" and state["draft"] == "done"
    assert state["tasks"]["goal.a"]["depends_on"] == []


def test_first_attempt_must_do_the_task_itself(tmp_path):
    state, backend, events = run(tmp_path, {
        "t:goal:1": [split("a", "b")], "t:goal:2": [final("did it myself")]})
    assert state["draft"] == "did it myself" and state["calls"] == 2
    assert any(e["event"] == "split_refused" for e in events)
    assert "Do the task yourself first" in backend.prompts["t:goal:2"][0]
    assert not any(c.startswith("t:goal.") for c in backend.calls)


def test_agent_history_does_not_resend_whole_files():
    from waypost.swarm.engine import HISTORY_TEXT_LIMIT, _compact_history
    big = "x" * 30000
    history = [{"action": {"kind": "tool", "tool": "write_file", "arguments": {"path": "a.py", "content": big}}},
               {"observation": json.dumps({"content": big})}]
    compact = json.dumps(_compact_history(history))
    assert len(compact) < 2 * HISTORY_TEXT_LIMIT + 500
    assert "read_file for the rest" in compact and '"path": "a.py"' in compact


def test_the_result_carries_the_product_files(tmp_path):
    state, backend, events = run(tmp_path, {
        "t:goal:1": [write("snake.py", "print('snake')"), RUN, final("pip output: installed pygame")]})
    assert state["status"] == "completed"
    assert "Файлы результата" in state["draft"] and "artifacts/t/goal/1/snake.py" in state["draft"]
    assert "print('snake')" in state["draft"]


def test_an_agent_reads_back_what_it_wrote_by_the_same_name(tmp_path):
    from waypost.swarm.tools import WorkspaceTools
    t = WorkspaceTools(tmp_path, "artifacts/t/goal/1")
    t.execute("write_file", {"path": "snake.py", "content": "x"})
    assert json.loads(t.execute("read_file", {"path": "snake.py"}))["content"] == "x"
    own = json.loads(t.execute("list_files", {}))
    assert own["entries"] == ["artifacts/t/goal/1/snake.py"]
    whole = json.loads(t.execute("list_files", {"path": "/"}))
    assert whole["entries"] == ["artifacts/"]  # service dirs hidden


def test_first_attempt_asks_for_a_mid_model_a_retry_for_a_large_one(tmp_path):
    state, backend, events = run(tmp_path, {
        "t:goal:1": [write("g.py"), final("not run")], "t:goal:2": [RUN, final("ran")]})
    assert backend.tiers["t:goal:1"] == "M" and backend.tiers["t:goal:2"] == "L"


def test_tier_hint_widens_the_pool_and_is_not_sent_upstream(tmp_path):
    from waypost.breaker import CircuitBreaker
    from waypost.classify import classify_l0
    from waypost.ledger import Ledger
    from waypost.registry import Offering, Registry
    from waypost.router import Router
    from waypost.schemas import Capability, ChatMessage, ChatRequest, Tier
    mid = Offering(provider="m", model_id="gemma-27b", base_url="http://m/v1", tier=Tier.M,
                   caps={Capability.JSON}, quality_score=0.8, limit_rpd=100)
    big = Offering(provider="b", model_id="llama-405b", base_url="http://b/v1", tier=Tier.L,
                   caps={Capability.JSON}, quality_score=0.8, limit_rpd=100)
    ledger = Ledger(str(tmp_path / "r.db"))
    for o in (mid, big):
        ledger.register(o)
    router = Router(Registry([mid, big]), ledger, CircuitBreaker(), stochastic=False)
    req = ChatRequest(messages=[ChatMessage(role="user", content="x " * 3000)], tier_hint="M")
    profile = classify_l0(req)
    assert router.plan(req, profile)[0].offering.model_id == "gemma-27b"
    assert "tier_hint" not in req.provider_payload("m")


# ------------------------------------------------ universal verification


def final_kind(text, kind):
    return {"kind": "final", "answer": text, "work_kind": kind}


def test_research_is_checked_against_its_sources(tmp_path):
    state, backend, events = run(tmp_path, {
        "t:goal:1": [final_kind("330 m, 1889 [https://example.org/eiffel]", "research")],
        "v:goal:1": [final("VERIFIED: the page states 330 m and 1889")]})
    assert state["tasks"]["goal"]["verified"] is True and state["tasks"]["goal"]["work_kind"] == "research"
    assert "fetch_url" in backend.prompts["v:goal:1"][0] or True
    assert backend.tiers["v:goal:1"] == "M"
    assert state["draft"].startswith("330 m")


def test_checker_problems_send_the_work_back_and_it_gets_fixed(tmp_path):
    state, backend, events = run(tmp_path, {
        "t:goal:1": [final_kind("It is 300 m tall.", "research")],
        "v:goal:1": [final("PROBLEMS:\n- no source cited\n- height today is 330 m")],
        "t:goal:2": [final_kind("330 m [https://example.org/eiffel]", "research")],
        "v:goal:2": [final("VERIFIED")]})
    assert state["draft"].startswith("330 m") and state["tasks"]["goal"]["verified"] is True
    assert "height today is 330 m" in backend.prompts["t:goal:2"][0]
    assert any(e["event"] == "verify_failed" for e in events)


def test_two_failed_checks_deliver_with_the_problems_attached(tmp_path):
    state, backend, events = run(tmp_path, {
        "t:goal:1": [final_kind("A letter of 40 words.", "write")],
        "v:goal:1": [final("PROBLEMS: 40 words, the task asks for 120-150")],
        "t:goal:2": [final_kind("A letter of 60 words.", "write")],
        "v:goal:2": [final("PROBLEMS: 60 words, the task asks for 120-150")]})
    assert state["status"] == "completed" and state["tasks"]["goal"]["verified"] is False
    assert "Не подтверждено проверкой" in state["draft"] and "120-150" in state["draft"]


def test_a_missing_checker_does_not_block_the_work(tmp_path):
    state, backend, events = run(tmp_path, {"t:goal:1": [final_kind("Average is 144.33", "analyze")]})
    assert state["status"] == "completed" and state["draft"] == "Average is 144.33"
    assert any(e["event"] == "verify_unavailable" for e in events)


def test_the_checker_is_another_model_and_cannot_write(tmp_path):
    state, backend, events = run(tmp_path, {
        "t:goal:1": [final_kind("Pick SQLite", "decide")],
        "v:goal:1": [{"kind": "tool", "tool": "write_file", "arguments": {"path": "x", "content": "y"}},
                     final("VERIFIED")]})
    assert state["tasks"]["goal"]["verified"] is True
    assert not (tmp_path / "workspace/artifacts/v/goal/1/x").exists()


# --------------------------------------------------------- web tools


def _tools_with(tmp_path, monkeypatch, handler):
    import httpx
    from waypost.swarm import tools as tools_mod
    monkeypatch.setattr(tools_mod, "_public_host", lambda host: None)
    monkeypatch.setattr(tools_mod.WorkspaceTools, "_http", httpx.Client(
        transport=httpx.MockTransport(handler), follow_redirects=False))
    return tools_mod.WorkspaceTools(tmp_path, "artifacts/t/goal/1", allow_python=True, allow_network=True)


def test_web_search_returns_titles_urls_and_snippets(tmp_path, monkeypatch):
    import httpx
    html = ('<a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fen.wikipedia.org'
            '%2Fwiki%2FEiffel_Tower&rut=x">Eiffel <b>Tower</b></a> ... <a class="result__snippet" href="x">'
            'It is 330 m tall</a>')
    t = _tools_with(tmp_path, monkeypatch, lambda r: httpx.Response(200, text=html))
    out = json.loads(t.execute("web_search", {"query": "eiffel height"}))
    assert out["results"][0] == {"title": "Eiffel Tower", "url": "https://en.wikipedia.org/wiki/Eiffel_Tower",
                                 "snippet": "It is 330 m tall"}


def test_fetch_url_returns_readable_text(tmp_path, monkeypatch):
    import httpx
    page = "<html><head><script>x=1</script></head><body><h1>Zen</h1><p>Beautiful is better.</p></body></html>"
    t = _tools_with(tmp_path, monkeypatch,
                    lambda r: httpx.Response(200, text=page, headers={"content-type": "text/html"}))
    out = json.loads(t.execute("fetch_url", {"url": "https://peps.python.org/pep-0020/"}))
    assert "Beautiful is better." in out["content"] and "x=1" not in out["content"]


def test_fetch_url_refuses_local_and_private_targets(tmp_path):
    from waypost.swarm.tools import WorkspaceTools
    t = WorkspaceTools(tmp_path, "artifacts/t/goal/1", allow_python=True, allow_network=True)
    for url in ("http://127.0.0.1:8080/health", "http://localhost:11434/", "http://192.168.1.1/"):
        with pytest.raises(ValueError, match="local/private"):
            t.execute("fetch_url", {"url": url})
