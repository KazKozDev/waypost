from __future__ import annotations

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
    allow_python: bool = False
    # Collective: how many independent members answer a collective question,
    # and whether each must come from a model family the others did not.
    collective_width: int = Field(default=3, ge=1, le=5)
    diverse_models: bool = True
    review_panel: bool = True


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


class Action(StrictModel):
    kind: Literal["tool", "final"]
    tool: str | None = None
    arguments: dict = Field(default_factory=dict)
    answer: str | None = None

    @model_validator(mode="after")
    def valid_action(self):
        if self.kind == "tool" and not self.tool:
            raise ValueError("tool action requires a tool name")
        if self.kind == "final" and not (self.answer and self.answer.strip()):
            raise ValueError("final action requires a nonempty answer")
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
