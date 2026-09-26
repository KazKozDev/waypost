"""Swarm v2: a coordinator that keeps the ledgers, and a council that decides.

Design (docs/swarm-coordinator-plan.md):
- the coordinator keeps a task ledger (facts, guesses, plan, team) and a
  progress ledger (done? moving? looping?) and notices dead ends — Magentic-One;
- every fork — the plan, what to do next (and in what order), whether the
  work is done, what to do in a dead end — is decided by a council of at
  least two model families: independent proposals, then rounds where each
  backs the proposal with the strongest reasoning; the argument wins, not
  the headcount — ReConcile / multi-agent debate;
- work goes through the board: requests are posted, a team of at least two
  agents on different models volunteer, and a contested request goes to the
  better-argued approach — the blackboard pattern;
- rules are only the safety net.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json

from .models import (Assignments, Claim, CoordinatorStep, DebateRuling, Position, TaskLedger,
                     Volunteer)

# Safety net only. Every decision is the council's.
COORDINATOR_STALLS = 2       # decided steps without progress before a forced replan
COORDINATOR_REPLANS = 3      # forced replans before delivering what there is
DEBATE_ROUNDS = 3
MAX_TEAM = 6
_PASSTHROUGH = ("RunInterrupted", "BudgetExceeded", "ReplanRequested")


def _step_key(step: CoordinatorStep):
    """Two proposed steps agree when they would do the same thing."""
    return (step.action, tuple(sorted(r.need.strip().lower()[:80] for r in step.requests)),
            (step.question or "").strip().lower()[:120])


class CoordinatorMixin:
    """Mixed into SwarmEngine; uses its model calls, workers, board, review
    panel, memory and resilience."""

    # ------------------------------------------------------------ council
    def _council_width(self) -> int:
        ledger = self.state.get("ledger") or {}
        return max(2, min(5, int(ledger.get("council", 2))))

    def _decide(self, kind: str, instruction: str, context: str, schema, key=None):
        """A fork decided by a council of different model families.

        Round 1: independent proposals. If they agree (`key`) — done. Else
        rounds 2-3: every member sees all proposals and backs the one whose
        reasoning is strongest, saying what persuaded it; stop on consensus
        or when nobody moves. No consensus: a judge from a family none of
        them used rules on the reasoning. A council that could only reach
        one family decides alone and says so."""
        width = self._council_width()
        members = self._collective(kind, instruction + " Think independently and give your reasoning.",
                                   context, schema, width=width)
        proposals = [value for value, _ in members]
        families = [family for _, family in members]
        if len(proposals) == 1:
            self.store.event("council", subject=kind, families=families, chosen=0, why="only one family answered")
            return proposals[0]
        if key is not None and len({key(p) for p in proposals}) == 1:
            self.store.event("council", subject=kind, families=families, chosen=0, why="agreed independently")
            return proposals[0]
        numbered = [{"n": i, "family": f, "proposal": p.model_dump()} for i, (p, f) in enumerate(members)]
        support = list(range(len(proposals)))
        for vote_round in range(2, DEBATE_ROUNDS + 1):
            votes = self._collective(
                f"{kind}-vote",
                "You are on the council deciding this. Read every proposal and its reasoning. Back the one "
                "whose REASONING is strongest (supports = its n); stance/argument = why. Change your mind "
                "only if an argument persuades you, and name it in persuaded_by. Never side with a majority "
                "because it is a majority.",
                context + "\nPROPOSALS:\n" + json.dumps(numbered, ensure_ascii=False), Position,
                width=len(proposals))
            new_support = [min(v.supports, len(proposals) - 1) if v.supports is not None else i
                           for i, (v, _) in enumerate(votes)]
            self.store.event("council_vote", subject=kind, round=vote_round, support=new_support,
                             persuaded=[v.persuaded_by for v, _ in votes if v.persuaded_by][:5])
            moved = new_support != support[:len(new_support)]
            support = new_support
            if len(set(support)) == 1 or not moved:
                break
        if len(set(support)) == 1:
            chosen, why = support[0], "consensus"
        else:
            try:
                ruling = self._structured_answer(
                    f"{kind}-judge", "The council did not agree. Rule by the STRENGTH OF THE REASONING, not by "
                    "how many back each proposal. chosen = the 0-based n.",
                    context + "\nPROPOSALS:\n" + json.dumps(numbered, ensure_ascii=False)
                    + "\nSUPPORT:\n" + json.dumps(support), DebateRuling,
                    avoid_families=[f for f in families if f] or None, max_failures=1)[0]
                chosen, why = min(ruling.chosen, len(proposals) - 1), ruling.why
            except Exception as exc:  # noqa: BLE001
                if type(exc).__name__ in _PASSTHROUGH:
                    raise
                chosen = max(set(support), key=support.count)
                why = "judge unavailable; most-backed reasoning"
        self.store.event("council", subject=kind, families=families, support=support, chosen=chosen,
                         why=why[:400])
        return proposals[chosen]

    # ------------------------------------------------------------ the loop
    def _coordinate(self):
        if not self.state.get("ledger"):
            self._coord_plan()
            return
        if reason := self.state.pop("pending_replan", None):
            self._coord_plan(reason=reason)
            return
        step = self._decide(
            "coordinator",
            "You are on the council steering an autonomous agent swarm. Read the task ledger, the progress, the "
            "requests on the board and their results. Judge honestly: is the task done, did the last step make "
            "real progress, are we looping (repeating work, re-reading, re-asking)? Then choose ONE action:\n"
            "- post: put work requests on the board (need, why, done_when, depends_on). Get the ORDER right: a "
            "request that needs another's output lists it in depends_on (review after write, tests after code). "
            "Post only what is not done yet; independent requests run in parallel.\n"
            "- deliberate: a fork in the work itself (approach, design, conflicting results); width = how many "
            "model families debate it.\n"
            "- review: have the current draft checked; width = reviewers.\n"
            "- finish: the work is done and the deliverable is the product itself.\n"
            "- replan: the plan or team is wrong.\n"
            "Scale effort to the task. Never repeat a request that is done — use its result. Autonomous: "
            "never wait for the user; resolve ambiguity by a stated assumption.",
            self._coord_context(), CoordinatorStep, key=_step_key)
        self._record_step(step)
        if self._coord_stalled(step):
            return
        if step.is_done and step.action not in ("review", "finish"):
            step = step.model_copy(update={"action": "finish"})
        getattr(self, f"_coord_{step.action}")(step)

    def _record_step(self, step: CoordinatorStep):
        entry = {"round": self.state["round"], "done": step.is_done, "progress": step.progress,
                 "looping": step.looping, "action": step.action, "reasoning": step.reasoning[:1000]}
        self.state.setdefault("progress_ledger", []).append(entry)
        self.store.event("coordinator_step", **entry, width=step.width,
                         requests=[r.id for r in step.requests], question=step.question)

    def _coord_stalled(self, step: CoordinatorStep) -> bool:
        """The council says there is no progress; the rule only acts when it
        has said so twice in a row, and then the council replans."""
        if step.progress and not step.looping:
            self.state["coord_stalls"] = 0
            return False
        stalls = self.state["coord_stalls"] = self.state.get("coord_stalls", 0) + 1
        if stalls < COORDINATOR_STALLS:
            return False
        replans = self.state["coord_replans"] = self.state.get("coord_replans", 0) + 1
        self.state["coord_stalls"] = 0
        if replans > COORDINATOR_REPLANS:
            self._current_draft()
            self._finish_best_effort(f"no progress after {replans - 1} replans")
            return True
        self._post("dead_end", f"Round {self.state['round']}: {step.reasoning[:400]}", "coordinator")
        self._coord_plan(reason="Two steps without progress: " + step.reasoning)
        return True

    # ------------------------------------------------------------- ledger
    def _coord_plan(self, reason: str = ""):
        revising = bool(self.state.get("ledger"))
        context = self._task_context()
        if revising:
            context += "\nCURRENT LEDGER:\n" + json.dumps(self.state["ledger"], ensure_ascii=False)
            context += "\nPROGRESS:\n" + json.dumps(self.state.get("progress_ledger", [])[-8:], ensure_ascii=False)
            context += "\nREASON FOR REPLAN:\n" + reason
        if lessons := self._lessons():
            context += "\nLESSONS FROM PREVIOUS RUNS:\n" + json.dumps(lessons, ensure_ascii=False)
        ledger = self._decide(
            "coordinator-plan",
            ("Revise" if revising else "Write") + " the task ledger: acceptance criteria, known facts, educated "
            "guesses, an ordered plan, and the team — at least TWO roles with different perspectives (for a "
            "simple task e.g. a builder and a critic; more only when the work really splits). council = how "
            "many model families should decide each fork (2 for simple work, up to 5 for hard, contested "
            "work). The swarm is autonomous. Roles can read/write files"
            + (" and run Python." if self.config.allow_python else "; Python and web access are unavailable."),
            context, TaskLedger)
        if revising:
            self.state["round"] += 1
            self.store.event("replan", reason=reason[:500], round=self.state["round"])
        self.state["ledger"] = ledger.model_dump()
        if not self.state.get("acceptance"):
            self.state["acceptance"] = ledger.acceptance
        self._post("decision", "Plan: " + "; ".join(ledger.plan)[:450], "council")
        self.store.event("coordinator_plan", round=self.state["round"], plan=ledger.plan,
                         team=[m.role for m in ledger.team], facts=ledger.facts[:8], council=ledger.council)

    def _coord_replan(self, step: CoordinatorStep):
        self._coord_plan(reason=step.reasoning)

    def _coord_context(self) -> str:
        requests = [{k: r.get(k) for k in ("id", "need", "status", "by", "done_when", "depends_on")}
                    | {"result": (r.get("result") or "")[:1500], "files": r.get("files", [])}
                    for r in self.state.get("requests", [])][-20:]
        data = {"task": self.state["task"], "ledger": self.state.get("ledger"),
                "progress": self.state.get("progress_ledger", [])[-6:],
                "requests": requests, "board": self._board_view(),
                "draft": (self.state.get("draft") or "")[:4000],
                "last_review": self.state["reviews"][-1] if self.state.get("reviews") else None,
                "user_messages": self.state.get("messages", [])}
        return json.dumps(data, ensure_ascii=False)

    # --------------------------------------------------- board: volunteers
    def _coord_post(self, step: CoordinatorStep):
        requests = self.state.setdefault("requests", [])
        known = {r["id"] for r in requests}
        for spec in step.requests:
            rid = spec.id if spec.id not in known else f"{spec.id}_{len(requests) + 1}"
            requests.append({"id": rid, "need": spec.need, "why": spec.why, "done_when": spec.done_when,
                             "depends_on": list(spec.depends_on), "status": "open",
                             "round": self.state["round"]})
            known.add(rid)
            self.store.event("request_posted", request=rid, need=spec.need[:400], depends_on=spec.depends_on)
        # Work in dependency order: whatever became ready runs next, until
        # nothing more can start.
        while self._dispatch():
            pass

    def _ready(self, request: dict) -> bool:
        done = {r["id"] for r in self.state.get("requests", []) if r["status"] == "done"}
        known = {r["id"] for r in self.state.get("requests", [])}
        # A dependency on something never posted cannot block forever.
        return all(dep in done or dep not in known for dep in request.get("depends_on", []))

    def _dispatch(self) -> bool:
        """Team members volunteer for the ready requests; a contested request
        goes to the better-argued approach. Returns whether anything ran."""
        ready = [r for r in self.state.get("requests", []) if r["status"] == "open" and self._ready(r)]
        if not ready:
            return False
        team = self.state["ledger"]["team"][:MAX_TEAM]
        board = [{k: r.get(k) for k in ("id", "need", "why", "done_when")} for r in ready]
        claims: dict[str, list[tuple[dict, Claim]]] = {}
        heard: list[str] = []
        for member in team:
            try:
                offer, family = self._structured_answer(
                    f"volunteer:{member['role'][:40]}",
                    f"You are {member['role']} ({member.get('strengths', '')}) in an agent swarm. Requests are "
                    "on the board. Take ONLY those you can do well; for each, describe concretely HOW you would "
                    "do it — your approach is compared with other volunteers'. Take nothing if none fit.",
                    json.dumps({"task": self.state["task"], "open_requests": board,
                                "board": self._board_view()}, ensure_ascii=False),
                    Volunteer, avoid_families=heard or None, max_failures=1)
            except Exception as exc:  # noqa: BLE001 - a silent member is not a failed run
                if type(exc).__name__ in _PASSTHROUGH:
                    raise
                continue
            member["family"] = family or member.get("family")
            if family:
                heard.append(family)
            for claim in offer.claims:
                claims.setdefault(claim.request, []).append((member, claim))
        picks = self._pick_volunteers(ready, claims)
        assigned = []
        for r in ready:
            offers = claims.get(r["id"], [])
            self.store.event("volunteers", request=r["id"],
                             offers=[{"role": m["role"], "family": m.get("family"), "approach": c.approach[:300]}
                                     for m, c in offers[:5]],
                             why=picks.get(r["id"], (None, None, ""))[2])
            if not offers:
                r["status"] = "unclaimed"
                self._post("fact", f"Nobody on the team took request {r['id']}: {r['need'][:300]}", "board")
                continue
            member, claim, why = picks[r["id"]]
            r.update(status="claimed", by=member["role"], approach=claim.approach, why_chosen=why)
            assigned.append(r)
        if assigned:
            self._execute_requests(assigned)
        return bool(assigned)

    def _pick_volunteers(self, ready, claims):
        """One volunteer: it takes the request. Several: a judge compares the
        approaches — confidence is not an argument."""
        picks = {}
        contested = {}
        for r in ready:
            offers = claims.get(r["id"], [])
            if len(offers) == 1:
                picks[r["id"]] = (offers[0][0], offers[0][1], "only volunteer")
            elif offers:
                contested[r["id"]] = offers
        if not contested:
            return picks
        brief = {rid: {"need": next(r["need"] for r in ready if r["id"] == rid),
                       "volunteers": [{"role": m["role"], "approach": c.approach} for m, c in offers]}
                 for rid, offers in contested.items()}
        families = [m.get("family") for offers in contested.values() for m, _ in offers if m.get("family")]
        decided = {}
        try:
            ruling = self._structured_answer(
                "volunteer-judge", "Several team members volunteered for the same request. For each request "
                "pick the volunteer whose APPROACH is most concrete and most likely to meet done_when; ignore "
                "self-confidence. role = the chosen role; why = the deciding argument.",
                json.dumps({"task": self.state["task"], "contested": brief}, ensure_ascii=False), Assignments,
                avoid_families=families or None, max_failures=1)[0]
            decided = {a.request: a for a in ruling.assignments}
        except Exception as exc:  # noqa: BLE001
            if type(exc).__name__ in _PASSTHROUGH:
                raise
        for rid, offers in contested.items():
            chosen = decided.get(rid)
            match = next(((m, c) for m, c in offers if chosen and m["role"] == chosen.role), None)
            if match:
                picks[rid] = (match[0], match[1], chosen.why)
            else:  # judge silent or named nobody on the list: most detailed approach
                m, c = max(offers, key=lambda o: len(o[1].approach))
                picks[rid] = (m, c, "judge unavailable; most detailed approach")
        return picks

    def _execute_requests(self, assigned: list[dict]):
        team = self.state["ledger"]["team"]

        def run(request):
            key = f"r{self.state['round']}:{request['id']}"
            others = [m.get("family") for m in team if m["role"] != request["by"] and m.get("family")]
            context = json.dumps({"task": self.state["task"], "acceptance": self.state["acceptance"],
                                  "request": {k: request.get(k) for k in ("need", "why", "done_when", "approach")},
                                  "done_requests": [{"id": d["id"], "result": (d.get("result") or "")[:2000],
                                                     "files": d.get("files", [])}
                                                    for d in self.state.get("requests", []) if d["status"] == "done"],
                                  "board": self._board_view()}, ensure_ascii=False)
            answer = self._worker(request["by"], "Do this request from the swarm board. " + request["need"]
                                  + (f"\nDone when: {request['done_when']}" if request.get("done_when") else "")
                                  + "\nBuild on done_requests and their files; do not redo them.",
                                  context, key, avoid_families=others or None)
            return request, key, answer

        with ThreadPoolExecutor(max_workers=self.config.concurrency) as pool:
            futures = [pool.submit(run, r) for r in assigned]
            errors = []
            for future in as_completed(futures):
                try:
                    request, key, answer = future.result()
                except Exception as exc:  # noqa: BLE001
                    errors.append(exc)
                    continue
                files = self._artifacts_for(key)
                stuck = str(answer).startswith("STUCK:")
                request.update(status="failed" if stuck else "done", result=str(answer)[:6000], files=files)
                self.store.event("request_done", request=request["id"], by=request["by"],
                                 status=request["status"], files=files)
                self._post("fact", f"{request['id']} {'stuck' if stuck else 'done'} ({request['by']}): "
                           + str(answer)[:300], request["by"])
        for request in assigned:
            if request["status"] == "claimed":
                request["status"] = "open"  # its worker failed; the council will see it open again
        if errors:
            raise errors[0]

    # ------------------------------------------------------------- debate
    def _coord_deliberate(self, step: CoordinatorStep):
        self._deliberate(step.question or "", step.width)

    def _deliberate(self, question: str, width: int | None = None) -> str:
        """A fork in the work itself: the council takes positions and backs
        the strongest argument (see _decide)."""
        context = json.dumps({"task": self.state["task"], "question": question, "board": self._board_view(),
                              "requests": [{k: r.get(k) for k in ("id", "status", "result")}
                                           for r in self.state.get("requests", [])][-10:]},
                             ensure_ascii=False)
        if width:
            self.state.setdefault("ledger", {})["council"] = max(self._council_width(), min(5, width))
        position = self._decide("deliberate", "A fork the swarm must decide. Give your stance, the strongest "
                                "argument for it, and your confidence.", context, Position)
        self._post("decision", f"{question[:200]} → {position.stance[:250]}", "council")
        self.store.event("ruling", question=question[:400], stance=position.stance[:600],
                         argument=position.argument[:600])
        return position.stance

    # ------------------------------------------------------ review, finish
    def _assemble(self) -> str:
        done = [r for r in self.state.get("requests", []) if r["status"] == "done"]
        if not done:
            return ""
        key = f"r{self.state['round']}:synthesis"
        self.state["records"].pop(key, None)  # a fresh assembly of the current results
        context = json.dumps({"task": self.state["task"], "acceptance": self.state["acceptance"],
                              "results": [{"id": r["id"], "by": r.get("by"), "result": r.get("result", "")[:6000],
                                           "files": r.get("files", [])} for r in done],
                              "board": self._board_view()}, ensure_ascii=False)
        return self._worker("synthesizer", "Deliver the PRODUCT, not a report about it. Start with the list of "
                            "final files (exact paths) — when several versions of a file exist, name the one "
                            "that is final and why. Then include the main deliverable itself (read the file and "
                            "put its content in the answer), then how to use it. Do not redo finished work. "
                            "Resolve contradictions noted on the board.", context, key)

    def _current_draft(self) -> str:
        """Assemble only when there is something new to assemble: the same
        done results give the same draft, and re-assembling it every finish
        would be the step repetition MAST ranks first among failures."""
        basis = sorted(f"{r['id']}:{len(r.get('result') or '')}"
                       for r in self.state.get("requests", []) if r["status"] == "done")
        if not (self.state.get("draft") and self.state.get("draft_basis") == basis):
            self.state["draft"] = self._assemble() or self.state.get("draft", "")
            self.state["draft_basis"] = basis
        return self.state["draft"]

    def _coord_review(self, step: CoordinatorStep):
        if not self._current_draft():
            self._post("fact", "Nothing to review yet: no request is done.", "coordinator")
            return
        evidence = self._worker("reviewer", "Independently audit the draft against the acceptance criteria. "
                                "Inspect actual files. Report concrete failures and missing evidence.",
                                self._context(include_draft=True),
                                key=f"r{self.state['round']}:audit-{len(self.state['reviews']) + 1}",
                                read_only=True)
        review = self._review_panel(self._context(include_draft=True) + "\nAUDIT:\n" + evidence,
                                    width=max(2, step.width, self._council_width()))
        self.state["reviews"].append(review.model_dump())
        self.state["reviewed_draft"] = self.state["draft"] if review.passed else None
        for finding in review.findings[:5]:
            self._post("fact", "Review: " + finding, "review-panel")

    def _coord_finish(self, step: CoordinatorStep):
        self._current_draft()
        if self.state.get("reviewed_draft") != self.state["draft"]:
            self._coord_review(step)
        if self.state.get("reviewed_draft") == self.state["draft"] and self.state["draft"]:
            self.state["status"] = "completed"
            self.store.event("coordinator_finish", round=self.state["round"])
