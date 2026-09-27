from __future__ import annotations

import os
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SwarmConfig(StrictModel):
    base_url: str = "http://127.0.0.1:8080/v1"
    model: str = "auto"
    privacy: Literal["normal", "strict"] = "normal"
    concurrency: int = Field(default=2, ge=1, le=16)
    max_tasks: int | None = Field(default=None, ge=1)
    max_rounds: int | None = Field(default=None, ge=1)
    max_steps: int | None = Field(default=None, ge=1)
    max_calls: int | None = Field(default=None, ge=1)
    max_tokens: int = Field(default=4096, ge=256, le=32768)
    request_timeout: float = Field(default=1500, gt=0)
    max_seconds: float | None = Field(default=None, gt=0)
    # On by default: a swarm that cannot run its code can only claim it works.
    # NOT a sandbox — code runs as the user (see tools.py).
    allow_python: bool = True
    # Internet for that code (never localhost). On at the user's request.
    allow_network: bool = True
    # Collective: how many independent members answer a collective question,
    # and whether each must come from a model family the others did not.
    collective_width: int = Field(default=3, ge=1, le=5)
    diverse_models: bool = True
    review_panel: bool = True
    proposals: bool = True
    board: bool = True
    debate: bool = True
    debate_rounds: int = Field(default=1, ge=1, le=2)
    memory: bool = True
    # coordinator: a coordinator decides every step (v2); pipeline: the
    # fixed phases of v1, kept for resuming old runs and for comparison.
    # swarm: stigmergy (v3) — agents coordinate through marks on the board;
    # coordinator: a council decides each step (v2); pipeline: fixed phases (v1).
    engine: Literal["swarm", "coordinator", "pipeline"] = Field(
        default_factory=lambda: os.getenv("WAYPOST_SWARM_ENGINE", "swarm"))


class Task(StrictModel):
    id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,48}$")
    role: str = Field(min_length=1, max_length=200)
    instruction: str = Field(min_length=1, max_length=12000)
    depends_on: list[str] = Field(default_factory=list)


class Plan(StrictModel):
    acceptance: list[str] = Field(min_length=1)
    tasks: list[Task] = Field(min_length=1)

    @model_validator(mode="after")
    def valid_graph(self):
        ids = {t.id for t in self.tasks}
        if ids & {"synthesis", "audit"}:
            raise ValueError("synthesis and audit are reserved task IDs")
        if len(ids) != len(self.tasks):
            raise ValueError("task IDs must be unique")
        done: set[str] = set()
        for t in self.tasks:
            if not set(t.depends_on) <= ids:
                raise ValueError(f"unknown dependencies for {t.id}")
        while len(done) < len(ids):
            ready = {t.id for t in self.tasks if t.id not in done and set(t.depends_on) <= done}
            if not ready:
                raise ValueError("dependency graph contains a cycle")
            done |= ready
        return self


class SubtaskSpec(StrictModel):
    id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,48}$")
    goal: str = Field(min_length=1, max_length=4000)
    kind: Literal["build", "check", "research", "analyze", "write", "decide"] = "build"
    depends_on: list[str] = Field(default_factory=list)


class Action(StrictModel):
    """One agent step. tool/final everywhere; the stigmergic swarm also lets
    an agent split its task, mark it a dead end, or flag a conflict between
    two results it is integrating."""
    kind: Literal["tool", "final", "split", "dead_end", "conflict"]
    tool: str | None = None
    arguments: dict = Field(default_factory=dict)
    answer: str | None = None
    subtasks: list[SubtaskSpec] = Field(default_factory=list)
    between: list[str] = Field(default_factory=list)
    # On final: what kind of work this was, so it is checked the right way.
    work_kind: Literal["build", "research", "analyze", "write", "decide", "check"] | None = None

    @model_validator(mode="after")
    def valid_action(self):
        if self.kind == "tool" and not self.tool:
            raise ValueError("tool action requires a tool name")
        if self.kind in ("final", "dead_end") and not (self.answer and self.answer.strip()):
            raise ValueError(f"{self.kind} action requires a nonempty answer")
        if self.kind == "split" and not (2 <= len(self.subtasks) <= 6):
            raise ValueError("split requires 2 to 6 subtasks")
        if self.kind == "conflict" and len(self.between) != 2:
            raise ValueError("conflict names exactly two task ids in between")
        return self


class Review(StrictModel):
    passed: bool
    findings: list[str]
    repair: Plan | None = None

    @model_validator(mode="after")
    def repair_required(self):
        if not self.passed and self.repair is None:
            raise ValueError("a failed review requires a repair plan")
        if self.passed and self.repair is not None:
            raise ValueError("a passing review must not contain a repair plan")
        return self


class ProgressDecision(StrictModel):
    action: Literal["continue", "redirect", "replan", "needs_input"]
    reason: str = Field(min_length=1)
    guidance: str = ""


class ReviewConsensus(StrictModel):
    """What a review panel agrees on. Only findings that at least two
    reviewers raised (in substance, not in wording) are confirmed; the
    repair plan addresses those and nothing else."""
    confirmed: list[str]
    repair: Plan | None = None

    @model_validator(mode="after")
    def repair_matches_findings(self):
        if self.confirmed and self.repair is None:
            raise ValueError("confirmed findings require a repair plan")
        if not self.confirmed and self.repair is not None:
            raise ValueError("no confirmed findings means no repair plan")
        return self


class Critique(StrictModel):
    proposal: int = Field(ge=0)
    strengths: list[str]
    flaws: list[str]


class PlanChoice(StrictModel):
    """A judge's pick among independent plans: critique each, then choose
    one or merge their strong parts into a new plan."""
    critiques: list[Critique]
    chosen: int = Field(ge=0)
    merged: Plan | None = None


class DraftChoice(StrictModel):
    """A judge's pick among independent final deliverables."""
    critiques: list[Critique]
    chosen: int = Field(ge=0)
    merged: str | None = None


class Rebuttal(StrictModel):
    """A specialist's read of the others' results: where they contradict
    each other or this specialist's own findings, and what should change."""
    contradictions: list[str]
    corrections: list[str] = Field(default_factory=list)


# ------------------------------------------------------ coordinator (v2)


class TeamMember(StrictModel):
    role: str = Field(min_length=1, max_length=200)
    strengths: str = ""


class TaskLedger(StrictModel):
    """The coordinator's picture of the task (Magentic-One's task ledger):
    what is known, what is guessed, the open plan, and the team it wants."""
    acceptance: list[str] = Field(min_length=1)
    facts: list[str] = Field(default_factory=list)
    guesses: list[str] = Field(default_factory=list)
    plan: list[str] = Field(min_length=1)
    # At least two, so members have someone to volunteer against and a
    # second perspective exists (e.g. a builder and a critic).
    team: list[TeamMember] = Field(min_length=2, max_length=6)
    # How many model families decide each fork, by task complexity.
    council: int = Field(default=2, ge=2, le=5)


class RequestSpec(StrictModel):
    id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,48}$")
    need: str = Field(min_length=1, max_length=4000)
    why: str = ""
    done_when: str = ""
    # Requests this one needs finished first ("review" after "write").
    depends_on: list[str] = Field(default_factory=list)


class CoordinatorStep(StrictModel):
    """One coordinator turn (Magentic-One's progress ledger): judge where
    the work stands, then choose exactly one action."""
    is_done: bool
    progress: bool
    looping: bool
    reasoning: str = Field(min_length=1)
    action: Literal["post", "deliberate", "review", "finish", "replan"]
    requests: list[RequestSpec] = Field(default_factory=list)
    question: str | None = None
    width: int = Field(default=1, ge=1, le=5)

    @model_validator(mode="after")
    def action_arguments(self):
        if self.action == "post" and not self.requests:
            raise ValueError("post requires at least one request")
        if self.action == "deliberate" and not (self.question and self.question.strip()):
            # The reasoning already states the fork; refusing the step over a
            # missing field cost live runs a retry each time.
            self.question = self.reasoning
        return self


class Claim(StrictModel):
    request: str
    confidence: float = Field(ge=0, le=1)
    approach: str = Field(min_length=1)


class Volunteer(StrictModel):
    """A team member's answer to the open requests: the ones it takes."""
    claims: list[Claim] = Field(default_factory=list)


class Position(StrictModel):
    """A debate turn. Round 1: `stance` and `argument` are the member's own;
    later rounds: `supports` names the position (0-based) it now backs and
    `persuaded_by` says which argument moved it, if any."""
    stance: str = Field(min_length=1)
    argument: str = Field(min_length=1)
    confidence: float = Field(default=0.5, ge=0, le=1)
    supports: int | None = Field(default=None, ge=0)
    persuaded_by: str = ""


class DebateRuling(StrictModel):
    chosen: int = Field(ge=0)
    why: str = Field(min_length=1)


class Verification(StrictModel):
    """An independent agent's check of a result against outside evidence."""
    verified: bool
    problems: list[str] = Field(default_factory=list)
    evidence: str = ""


class QuorumVote(StrictModel):
    """An independent check of two conflicting results: which is right."""
    winner: str
    why: str = Field(min_length=1)


class Assignment(StrictModel):
    request: str
    role: str
    why: str = Field(min_length=1)


class Assignments(StrictModel):
    """Who takes each contested request, decided by comparing the
    volunteers' approaches — not their self-reported confidence."""
    assignments: list[Assignment]
