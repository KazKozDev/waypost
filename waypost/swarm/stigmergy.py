"""Swarm v3: stigmergy (docs/swarm-stigmergy.md).

No meetings and no boss. Agents coordinate only through marks on the
board: a task is needed, claimed, done, a dead end, or waiting for its
subtasks. One agent claims a task and either does it, splits it, or marks
it a dead end. The same work is never done twice: a task whose goal was
already solved takes that result; one already claimed is waited for.
The collective only acts where there is disagreement (a quorum of
independent checks). The environment verifies code by running it. The
swarm stops by itself when the goal is done.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import re

from .models import QuorumVote

# Safety net only.
MAX_DEPTH = 3
MAX_ATTEMPTS = 3
STALE_STEPS = 3
QUORUM = 3
KINDS = ("build", "check", "research", "write", "decide")
_PASSTHROUGH = ("RunInterrupted", "BudgetExceeded")

AGENT_RULES = (
    "You are one agent of a swarm. There are no meetings: the board shows what is done, claimed and "
    "dead — never redo done work, reuse its results and files. For YOUR task choose exactly one:\n"
    "- kind=final: you did it; answer = the result itself (for code: run it and its tests with "
    "run_python first and include the real output).\n"
    "- kind=split: ONLY after a previous attempt showed the task is too big for one agent (see your "
    "notes): 2-6 subtasks (id, goal, kind of build/check/research/write/decide, depends_on). Your first "
    "attempt must do the task itself — one agent writes a whole small program in one go.\n"
    "- kind=dead_end: impossible in this environment (no network, no access, missing tool); answer = "
    "the exact reason. Do not retry impossible things — the reason is a valid result.\n"
    "- kind=conflict: you are integrating subtask results and two of them contradict; between = "
    "their two ids, answer = what they disagree on.\n"
    "If you have subtask results, integrate them into your result instead of redoing them."
)


def _fingerprint(goal: str) -> str:
    words = re.findall(r"\w+", goal.lower())
    return hashlib.sha256(" ".join(words).encode()).hexdigest()[:16]


class StigmergyMixin:
    """Mixed into SwarmEngine; uses its workers, board, memory, sandbox and
    resilience."""

    # --------------------------------------------------------------- board
    def _tasks(self) -> dict:
        return self.state.setdefault("tasks", {})

    def _new_task(self, tid: str, goal: str, kind: str = "build", parent: str | None = None,
                  depends_on=(), depth: int = 0) -> dict:
        tasks = self._tasks()
        fp = _fingerprint(goal)
        twin = next((t for t in tasks.values() if t["fp"] == fp and t["status"] in ("done", "dead_end")), None)
        task = {"id": tid, "goal": goal, "kind": kind if kind in KINDS else "build", "parent": parent,
                "depends_on": list(depends_on), "depth": depth, "fp": fp, "status": "needed",
                "attempts": 0, "notes": [], "children": [], "result": "", "files": [], "verified": False}
        if twin:
            # Solved (or proven impossible) before: take that mark, not the work.
            task.update(status=twin["status"], result=twin["result"], files=twin["files"],
                        verified=twin["verified"], reused=twin["id"])
            self.store.event("task_reused", task=tid, twin=twin["id"], status=twin["status"])
        else:
            ancestors, up = set(), parent
            while up:
                ancestors.add(up)
                up = tasks[up].get("parent")
            busy = next((t for t in tasks.values() if t["fp"] == fp and t["id"] != tid
                         and t["id"] not in ancestors
                         and t["status"] in ("needed", "claimed", "waiting")), None)
            if busy:
                # Someone already has this work: wait for it and take its result.
                task["depends_on"].append(busy["id"])
                task["twin"] = busy["id"]
        tasks[tid] = task
        self.store.event("task_posted", task=tid, goal=goal[:300], work_kind=task["kind"], parent=parent,
                         depends_on=list(depends_on), status=task["status"])
        return task

    def _ready_tasks(self) -> list[dict]:
        tasks = self._tasks()
        finished = {tid for tid, t in tasks.items() if t["status"] in ("done", "dead_end")}
        ready = []
        for t in tasks.values():
            if t["status"] != "needed":
                continue
            # A dependency on a twin still being worked on waits for it.
            if all(dep in finished or dep not in tasks for dep in t["depends_on"]):
                ready.append(t)
        return sorted(ready, key=lambda t: -t["depth"])  # leaves first

    def _wake_parents(self):
        """A parent waiting on its subtasks is needed again when all of them
        are finished: its agent integrates their results."""
        tasks = self._tasks()
        for t in tasks.values():
            if t["status"] == "waiting" and all(tasks[c]["status"] in ("done", "dead_end") for c in t["children"]):
                t["status"] = "needed"
                self.store.event("task_ready_to_integrate", task=t["id"])

    # ---------------------------------------------------------------- step
    def _swarm_step(self):
        tasks = self._tasks()
        if not tasks:
            self._new_task("goal", self.state["task"], kind="build")
            self.state.setdefault("acceptance", [self.state["task"]])
        if not self.state.get("swarm_recovered"):
            # A claim held by a run that died is no one's claim now.
            for t in tasks.values():
                if t["status"] == "claimed":
                    t["status"] = "needed"
            self.state["swarm_recovered"] = True
        self._wake_parents()
        goal = tasks["goal"]
        if goal["status"] in ("done", "dead_end"):
            self._swarm_finish(goal)
            return
        ready = self._ready_tasks()
        marks_before = json.dumps({tid: t["status"] for tid, t in tasks.items()}, sort_keys=True)
        if ready:
            batch = ready[: max(1, self.config.concurrency)]
            with ThreadPoolExecutor(max_workers=len(batch)) as pool:
                futures = [pool.submit(self._swarm_work, t) for t in batch]
                errors = []
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as exc:  # noqa: BLE001
                        errors.append(exc)
            if errors:
                raise errors[0]
        marks_after = json.dumps({tid: t["status"] for tid, t in tasks.items()}, sort_keys=True)
        if marks_after == marks_before:
            stale = self.state["swarm_stale"] = self.state.get("swarm_stale", 0) + 1
            if stale >= STALE_STEPS:
                self._swarm_finish(goal, reason=f"{stale} steps without a single mark changing")
        else:
            self.state["swarm_stale"] = 0

    def _swarm_finish(self, goal: dict, reason: str = ""):
        if goal["status"] == "dead_end":
            self.state["draft"] = "Невозможно в этой среде: " + goal["result"]
        else:
            self.state["draft"] = goal["result"] or self._partial_results()
        wrote_code = any(f.endswith(".py") for t in self._tasks().values() for f in t["files"])
        ran = any(t["verified"] for t in self._tasks().values())
        if goal["status"] == "done" and wrote_code and not ran and self.config.allow_python:
            self.state["draft"] += "\n\n---\nНе проверено запуском."
        self.state["draft"] += self._package_files()
        if reason:
            self._finish_best_effort(reason)
            return
        self.state["status"] = "completed"
        self.store.event("swarm_done", goal_status=goal["status"], verified=goal["verified"])

    def _package_files(self) -> str:
        """The product is the files, whatever the agent chose to say: list
        them and include the code. Build byproducts are not the product."""
        noise = ("__pycache__", ".dist-info", "/.", "Library/Caches", "/include/", "/bin/", "/lib/")
        files = sorted({f for t in self._tasks().values() if t["status"] == "done" for f in t["files"]
                        if not any(n in f for n in noise)})
        if not files:
            return ""
        parts = ["\n\n---\nФайлы результата:\n" + "\n".join(f"- {f}" for f in files)]
        code = [f for f in files if f.endswith((".py", ".md", ".txt", ".json", ".toml"))][:4]
        for f in code:
            try:
                text = (self.store.workspace / f).read_text()
            except (OSError, UnicodeDecodeError):
                continue
            if len(text) <= 20000:
                lang = "python" if f.endswith(".py") else ""
                parts.append(f"\n### {f}\n```{lang}\n{text}\n```")
        return "\n".join(parts)

    def _partial_results(self) -> str:
        done = [t for t in self._tasks().values() if t["status"] == "done" and t["id"] != "goal"]
        return "\n\n".join(f"## {t['id']}\n{t['result'][:4000]}" for t in done)

    # ---------------------------------------------------------------- work
    def _board_brief(self, task: dict) -> dict:
        tasks = self._tasks()
        chain, parent = [], task.get("parent")
        while parent:
            chain.append(tasks[parent]["goal"][:500])
            parent = tasks[parent].get("parent")
        return {
            "your_task": {"id": task["id"], "goal": task["goal"], "kind": task["kind"]},
            "goals_above": chain,
            "subtask_results": [{"id": c, "status": tasks[c]["status"], "result": tasks[c]["result"][:4000],
                                 "files": tasks[c]["files"], "verified": tasks[c]["verified"]}
                                for c in task["children"]],
            "done_on_board": [{"id": t["id"], "goal": t["goal"][:200], "result": t["result"][:800],
                               "files": t["files"]}
                              for t in tasks.values() if t["status"] == "done" and t["id"] not in task["children"]][-12:],
            "dead_ends": [{"goal": t["goal"][:200], "reason": t["result"][:300]}
                          for t in tasks.values() if t["status"] == "dead_end"][-8:],
            "notes_from_previous_attempts": task["notes"][-3:],
        }

    def _swarm_work(self, task: dict):
        twin = self._tasks().get(task.get("twin") or "")
        if twin and twin["status"] in ("done", "dead_end"):
            task.update(status=twin["status"], result=twin["result"], files=twin["files"],
                        verified=twin["verified"], reused=twin["id"])
            self.store.event("task_reused", task=task["id"], twin=twin["id"], status=twin["status"])
            return
        task["status"] = "claimed"
        task["attempts"] += 1
        key = f"t:{task['id']}:{task['attempts']}"
        self.store.event("task_claimed", task=task["id"], attempt=task["attempts"], work_kind=task["kind"])
        avoid = self._poor_families(f"swarm-{task['kind']}") + task.get("tried_families", [])
        try:
            # Whoever is near tries first; the heavy load goes to the strong:
            # a first attempt may run on a mid-size model, a retry after a
            # failure asks for a large one.
            answer = self._worker(f"swarm agent ({task['kind']})", AGENT_RULES,
                                  json.dumps(self._board_brief(task), ensure_ascii=False), key,
                                  avoid_families=avoid or None,
                                  tier_hint="M" if task["attempts"] == 1 else "L")
        except Exception as exc:  # noqa: BLE001
            if type(exc).__name__ in _PASSTHROUGH:
                task["status"] = "needed"
                raise
            return self._attempt_failed(task, f"{type(exc).__name__}: {str(exc)[:300]}")
        record = self.state["records"].get(key, {})
        if record.get("family"):
            task.setdefault("tried_families", []).append(record["family"])
        outcome = record.get("outcome") or {"kind": "final", "answer": answer}
        if str(answer).startswith("STUCK:"):
            return self._attempt_failed(task, answer)
        getattr(self, f"_on_{outcome['kind']}")(task, outcome, key, record)

    def _attempt_failed(self, task: dict, why: str):
        task["notes"].append(f"attempt {task['attempts']} failed: {why}")
        if task["attempts"] >= MAX_ATTEMPTS:
            self._mark_dead(task, f"gave up after {task['attempts']} attempts: {why}")
        else:
            task["status"] = "needed"
            self.store.event("task_retry", task=task["id"], why=why[:300])

    def _mark_dead(self, task: dict, reason: str):
        task.update(status="dead_end", result=reason)
        self._post("dead_end", f"{task['goal'][:200]}: {reason[:300]}", task["id"])
        self.store.event("task_dead_end", task=task["id"], reason=reason[:400])
        self._note_kind(task, success=False)

    def _ran_ok(self, record: dict) -> bool:
        """The environment's word: did this attempt run code successfully?"""
        history = record.get("history", [])
        for i, item in enumerate(history):
            action = item.get("action") or {}
            if action.get("tool") == "run_python" and i + 1 < len(history):
                try:
                    if json.loads(history[i + 1].get("observation", "{}")).get("exit_code") == 0:
                        return True
                except (TypeError, ValueError):
                    continue
        return False

    # ------------------------------------------------------------ outcomes
    def _on_final(self, task, outcome, key, record):
        files = self._artifacts_for(key)
        verified = self._ran_ok(record)
        needs_run = task["kind"] == "build" and self.config.allow_python and files and any(
            f.endswith(".py") for f in files)
        if needs_run and not verified and task["attempts"] < 2:
            # Words are not evidence: back to the board with that note.
            task["notes"].append("You wrote code but did not run it. Run it and its tests with run_python, "
                                 "then finish with the real output.")
            task["status"] = "needed"
            self.store.event("task_unverified", task=task["id"])
            return
        task.update(status="done", result=str(outcome.get("answer", "")), files=files, verified=verified)
        self.store.event("mark_done", task=task["id"], verified=verified, files=files)
        self._post("fact", f"{task['id']} done{' (run ok)' if verified else ''}: {task['result'][:200]}", task["id"])
        self._note_kind(task, success=True)

    def _on_dead_end(self, task, outcome, key, record):
        self._mark_dead(task, str(outcome.get("answer", "")))

    def _on_split(self, task, outcome, key, record):
        subtasks = outcome.get("subtasks") or []
        if task["attempts"] < 2:
            # An ant recruits only after it failed to carry the load alone.
            # First attempt: do it yourself; what you made stays on disk.
            task["notes"].append("You split on your first attempt. Do the task yourself first; you may split "
                                 "only if an attempt shows it is really too big for one agent. Files you "
                                 "already wrote: " + ", ".join(self._artifacts_for(key)[:10]))
            task["status"] = "needed"
            self.store.event("split_refused", task=task["id"], reason="first attempt")
            return
        if task["depth"] >= MAX_DEPTH:
            task["notes"].append(f"Do not split further (depth {MAX_DEPTH} reached): do the task yourself.")
            task["status"] = "needed"
            return
        tasks = self._tasks()
        ids = {}
        for spec in subtasks[:6]:
            tid = f"{task['id']}.{spec['id']}"[:120]
            ids[spec["id"]] = tid
        for spec in subtasks[:6]:
            deps = [ids.get(d, d) for d in spec.get("depends_on", [])]
            if ids[spec["id"]] not in tasks:
                self._new_task(ids[spec["id"]], spec["goal"], spec.get("kind", "build"), parent=task["id"],
                               depends_on=deps, depth=task["depth"] + 1)
            task["children"].append(ids[spec["id"]])
        task["status"] = "waiting"
        self.store.event("task_split", task=task["id"], children=task["children"])

    def _on_conflict(self, task, outcome, key, record):
        """Bees choose by quorum only when there are rival options."""
        tasks = self._tasks()
        a, b = (outcome.get("between") or [None, None])[:2]
        if a not in tasks or b not in tasks:
            task["notes"].append("conflict named unknown subtasks; integrate or report instead")
            task["status"] = "needed"
            return
        brief = {"question": outcome.get("answer", ""), a: tasks[a]["result"][:4000], b: tasks[b]["result"][:4000],
                 "files": {a: tasks[a]["files"], b: tasks[b]["files"]}}
        votes = self._collective(f"quorum:{task['id']}", "Two results contradict. Check independently (read "
                                 f"files, run code if useful) which is right. winner = {a} or {b}.",
                                 json.dumps(brief, ensure_ascii=False), QuorumVote, width=QUORUM)
        tally = [v.winner for v, _ in votes if v.winner in (a, b)]
        winner = max((a, b), key=tally.count) if tally else a
        loser = b if winner == a else a
        self.store.event("quorum", task=task["id"], between=[a, b], votes=tally, winner=winner)
        self._mark_dead(tasks[loser], f"lost the quorum to {winner}: "
                        + "; ".join(v.why for v, _ in votes if v.winner == winner)[:300])
        task["notes"].append(f"Quorum: {winner} is right, {loser} is wrong. Integrate with that.")
        task["status"] = "needed"

    # --------------------------------------------------------------- memory
    def _note_kind(self, task: dict, success: bool):
        """Response thresholds: a family's record per kind of work is what
        makes it asked first (or last) for that kind next time."""
        family = (task.get("tried_families") or [None])[-1]
        if family:
            stats = self.state.setdefault("kind_stats", {}).setdefault(f"swarm-{task['kind']}", {})
            entry = stats.setdefault(family, {"ok": 0, "fail": 0})
            entry["ok" if success else "fail"] += 1
            used = self.state.setdefault("families_used", {}).setdefault(f"swarm-{task['kind']}", [])
            if family not in used:
                used.append(family)
