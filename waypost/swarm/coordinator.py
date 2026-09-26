"""Swarm v2: a coordinator that decides every step.

Design (docs/swarm-coordinator-plan.md):
- the coordinator keeps a task ledger (facts, guesses, plan, team) and a
  progress ledger (done? moving? looping? what next and why) — Magentic-One;
- work goes through the board: the coordinator posts requests, team members
  volunteer for the ones they can do — the blackboard pattern;
- forks go to a debate of different model families until positions stop
  moving; the stronger argument wins, not the headcount — ReConcile / MAD;
- rules are only the safety net: two steps without progress force a replan,
  too many replans deliver the best there is.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json

from .models import (Claim, CoordinatorStep, DebateRuling, Position, TaskLedger,
                     Volunteer)

# Safety net only. Everything else is the coordinator's call.
COORDINATOR_STALLS = 2       # steps without progress before a forced replan
COORDINATOR_REPLANS = 3      # forced replans before delivering what there is
DEBATE_ROUNDS = 3
MAX_TEAM_VOLUNTEERS = 6


class CoordinatorMixin:
    """Mixed into SwarmEngine; uses its model calls, workers, board, review
    panel, memory and resilience."""

    # ------------------------------------------------------------ the loop
    def _coordinate(self):
        if not self.state.get("ledger"):
            self._coord_plan()
            return
        if reason := self.state.pop("pending_replan", None):
            self._coord_plan(reason=reason)
            return
        step = self._structured(
            "coordinator",
            "You are the coordinator of an autonomous agent swarm. Read the task ledger, the progress so far, "
            "the requests on the board and their results. First judge honestly: is the task done, did the last "
            "step make real progress, are we looping (repeating work, re-reading the same files, re-asking the "
            "same thing)? Then choose ONE action:\n"
            "- post: put work requests on the board (need, why, done_when); team members volunteer for them. "
            "Post only what is not already done; independent requests run in parallel.\n"
            "- deliberate: a real fork where the choice matters (approach, design, conflicting results). Set "
            "width to how many independent model families should debate it: 2-3 for a genuine fork, more only "
            "for hard ones. Do not deliberate trivia.\n"
            "- review: have the current draft checked; width = number of reviewers (1 for simple work).\n"
            "- finish: the work is done; it will be assembled and checked, and comes back to you if it fails.\n"
            "- replan: the plan or team is wrong; rewrite it.\n"
            "Scale effort to the task: a simple task is one request and one check. Never repeat a request that "
            "is already done — use its result. The swarm is autonomous: never wait for the user; resolve "
            "ambiguity by a stated assumption.",
            self._coord_context(), CoordinatorStep)
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
        """The coordinator notices a dead end itself; the rule only acts when
        it has said so twice in a row (Magentic-One's stall counter)."""
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
        ledger = self._structured(
            "coordinator-plan",
            ("Revise" if revising else "Write") + " the task ledger: acceptance criteria, known facts, "
            "educated guesses, an ordered plan of open items, and the team — the roles this task needs "
            "(1 role for a simple task, more only when the work really splits). The swarm is autonomous. "
            "Each role can read/write files" + (" and run Python." if self.config.allow_python else
                                               "; Python and web access are unavailable."),
            context, TaskLedger)
        if revising:
            self.state["round"] += 1
            self.store.event("replan", reason=reason[:500], round=self.state["round"])
        self.state["ledger"] = ledger.model_dump()
        self.state["acceptance"] = ledger.acceptance if not self.state.get("acceptance") else \
            self.state["acceptance"]
        self._post("decision", "Plan: " + "; ".join(ledger.plan)[:450], "coordinator")
        self.store.event("coordinator_plan", round=self.state["round"], plan=ledger.plan,
                         team=[m.role for m in ledger.team], facts=ledger.facts[:8])

    def _coord_replan(self, step: CoordinatorStep):
        self._coord_plan(reason=step.reasoning)

    def _coord_context(self) -> str:
        requests = [{k: r.get(k) for k in ("id", "need", "status", "by", "done_when")}
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
                             "status": "open", "round": self.state["round"]})
            known.add(rid)
            self.store.event("request_posted", request=rid, need=spec.need[:400])
        self._dispatch()

    def _dispatch(self):
        """Team members volunteer for open requests; each request goes to the
        member most confident in a concrete approach. What nobody takes goes
        back to the coordinator as a fact on the board."""
        open_requests = [r for r in self.state.get("requests", []) if r["status"] == "open"]
        if not open_requests:
            return
        team = self.state["ledger"]["team"][:MAX_TEAM_VOLUNTEERS]
        claims: dict[str, list[tuple[float, str, Claim]]] = {}
        if len(team) == 1:
            # One member: it is the team; asking it to volunteer is ceremony.
            for r in open_requests:
                claims[r["id"]] = [(1.0, team[0]["role"], Claim(request=r["id"], confidence=1.0,
                                                                  approach="sole team member"))]
        else:
            board = [{k: r[k] for k in ("id", "need", "why", "done_when")} for r in open_requests]
            for member in team:
                try:
                    offer = self._structured_answer(
                        f"volunteer:{member['role'][:40]}",
                        f"You are {member['role']} ({member.get('strengths', '')}) on an agent swarm. Open "
                        "requests are on the board. Claim ONLY those you can do well, each with an honest "
                        "confidence (0-1) and the concrete approach you would take. Claim nothing if none fit.",
                        json.dumps({"task": self.state["task"], "open_requests": board,
                                    "board": self._board_view()}, ensure_ascii=False),
                        Volunteer, max_failures=1)[0]
                except Exception as exc:  # noqa: BLE001 - a silent member is not a failed run
                    if type(exc).__name__ in ("RunInterrupted", "BudgetExceeded", "ReplanRequested"):
                        raise
                    continue
                for claim in offer.claims:
                    claims.setdefault(claim.request, []).append((claim.confidence, member["role"], claim))
        assigned = []
        for r in open_requests:
            offers = sorted(claims.get(r["id"], []), key=lambda c: c[0], reverse=True)
            self.store.event("volunteers", request=r["id"],
                             offers=[{"role": role, "confidence": conf, "approach": c.approach[:300]}
                                     for conf, role, c in offers[:5]])
            if not offers:
                r["status"] = "unclaimed"
                self._post("fact", f"Nobody on the team took request {r['id']}: {r['need'][:300]}", "board")
                continue
            _, role, claim = offers[0]
            r.update(status="claimed", by=role, approach=claim.approach)
            assigned.append(r)
        self._execute_requests(assigned)

    def _execute_requests(self, assigned: list[dict]):
        def run(request):
            key = f"r{self.state['round']}:{request['id']}"
            context = json.dumps({"task": self.state["task"], "acceptance": self.state["acceptance"],
                                  "request": {k: request.get(k) for k in ("need", "why", "done_when", "approach")},
                                  "done_requests": [{"id": d["id"], "result": (d.get("result") or "")[:2000],
                                                     "files": d.get("files", [])}
                                                    for d in self.state.get("requests", []) if d["status"] == "done"],
                                  "board": self._board_view()}, ensure_ascii=False)
            answer = self._worker(request["by"], "Do this request from the swarm board. " + request["need"]
                                  + (f"\nDone when: {request['done_when']}" if request.get("done_when") else ""),
                                  context, key)
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
                request["status"] = "open"  # its worker failed; the coordinator will see it open again
        if errors:
            raise errors[0]

    # ------------------------------------------------------------- debate
    def _coord_deliberate(self, step: CoordinatorStep):
        self._deliberate(step.question or "", step.width)

    def _deliberate(self, question: str, width: int) -> str:
        """Independent positions, then rounds where every member sees all
        positions and arguments and backs one, saying what persuaded it.
        Stops when support stops moving; without consensus a judge from
        another family rules on the strength of the arguments."""
        base = {"task": self.state["task"], "question": question, "board": self._board_view(),
                "requests": [{k: r.get(k) for k in ("id", "status", "result")}
                             for r in self.state.get("requests", [])][-10:]}
        members = self._collective("deliberate", "A fork the swarm must decide. Give your stance, the "
                                   "strongest argument for it, and your confidence. Think independently.",
                                   json.dumps(base, ensure_ascii=False), Position, width=width)
        positions = [{"stance": p.stance, "argument": p.argument, "family": f} for p, f in members]
        self.store.event("deliberation", round=1, question=question[:400], positions=positions)
        support = list(range(len(positions)))
        for debate_round in range(2, DEBATE_ROUNDS + 1):
            if len(positions) < 2 or len(set(support)) == 1:
                break
            numbered = [{"n": i, **p} for i, p in enumerate(positions)]
            votes = self._collective(
                f"deliberate-r{debate_round}",
                "Here are all positions with their arguments. Back the one whose ARGUMENT is strongest "
                "(supports = its n). Change only if an argument actually persuades you, and say which "
                "(persuaded_by). Do not side with a majority just because it is a majority.",
                json.dumps({**base, "positions": numbered}, ensure_ascii=False), Position, width=len(positions))
            new_support = [min(v.supports, len(positions) - 1) if v.supports is not None else i
                           for i, (v, _) in enumerate(votes)]
            self.store.event("deliberation", round=debate_round, support=new_support,
                             persuaded=[v.persuaded_by for v, _ in votes if v.persuaded_by][:5])
            if new_support == support[:len(new_support)]:
                support = new_support
                break  # nobody moved: more rounds would only repeat
            support = new_support
        if len(set(support)) == 1:
            chosen, why = support[0], "consensus"
        else:
            try:
                ruling = self._structured_answer(
                    "debate-judge", "No consensus. Rule by the STRENGTH OF THE ARGUMENTS, not by how many back "
                    "each position. chosen = the 0-based index.",
                    json.dumps({**base, "positions": positions, "support": support}, ensure_ascii=False),
                    DebateRuling, avoid_families=[p["family"] for p in positions if p["family"]] or None,
                    max_failures=1)[0]
                chosen, why = min(ruling.chosen, len(positions) - 1), ruling.why
            except Exception as exc:  # noqa: BLE001
                if type(exc).__name__ in ("RunInterrupted", "BudgetExceeded", "ReplanRequested"):
                    raise
                chosen = max(set(support), key=support.count)
                why = "judge unavailable; most-backed argument"
        decision = positions[chosen]
        self._post("decision", f"{question[:200]} → {decision['stance'][:250]}", "debate")
        self.store.event("ruling", question=question[:400], stance=decision["stance"][:600],
                         argument=decision["argument"][:600], why=why[:400])
        return decision["stance"]

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
        return self._worker("synthesizer", "Assemble the final deliverable from the results and files. Do not "
                            "redo finished work: reference or quote it. Resolve contradictions on the board. "
                            "Return the actual answer, not a plan.", context, key)

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
                                    width=step.width)
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
