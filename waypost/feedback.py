"""Implicit feedback: learning from what the user does next.

The verifier can only judge what is mechanically checkable — valid JSON,
parseable code, a truncated answer, the wrong language. That is a
minority of the traffic. For chat, summarization, explanation there was
no quality signal at all, so the bandit learned nothing about the classes
where most requests live. It was rewarded for HTTP 200s and, after that
was fixed, for passing checks that those requests never trigger.

But the signal exists; it was simply never collected. The user's next
action is a verdict on the last answer:

  regeneration  — the same request again, in the same session. They were
                  not satisfied enough to move on.
  correction    — the next message opens with "no", "not like that",
                  "wrong". A rejection, stated.
  continuation  — an ordinary next message. Weak, but real: the answer
                  was good enough to build on.

Three deliberate limits, because implicit signals are easy to
over-believe:

*Weak rewards.* A correction is 0.15, not 0. "No, in Python" after a
correct answer is a change of mind, not a defect, and there is no way to
tell the two apart from the outside. The numbers are meant to be nudges
that accumulate over hundreds of requests, not verdicts on one.

*Explicit beats inferred.* A rating through /v1/feedback is recorded at
full weight and overrides whatever was inferred for that request.

*One verdict per answer.* The first signal for a request is the one that
counts, so a long session cannot keep re-punishing the same answer.
"""
from __future__ import annotations

import hashlib
import logging
import re
import threading
import time
from dataclasses import dataclass
from typing import Any

from .schemas import ChatMessage, ChatRequest

log = logging.getLogger("waypost.feedback")

# Rewards in [0,1], fed to the bandit exactly like a verifier verdict.
REWARDS = {
    "rated_good": 1.0,
    "rated_bad": 0.0,
    "corrected": 0.15,
    "regenerated": 0.25,
    "continued": 0.75,
}

# Openings that read as a rejection of what was just said. Deliberately
# narrow: matched only at the start of the message, and only when the
# message is short enough to be a reaction rather than a new task.
# A rejection may open with a bare "no," before saying what is wrong —
# "No, that's not what I meant". So each phrase below is matched with an
# optional lead-in rather than strictly at the start.
_LEAD = r"^(?:(?:нет|no|nope|неа)[,!.]?\s+)?"

CORRECTION_MARKERS = (
    # A bare "no" — the WHOLE message, not a word it happens to begin
    # with. "no problem, thanks!" is agreement, and reading that as a
    # rejection is the exact failure this list has to avoid.
    r"^(нет|no|nope|неа)[.!,…]*$",
    _LEAD + r"(не так|не то|неверно|неправильно|это не то|опять не|снова не)\b",
    _LEAD + r"(not (quite|right|what|it)|wrong|incorrect|that'?s not)\b",
    _LEAD + r"(это|так|то) не (то|так)\b",
    r"^(ты )?(ошибся|ошибаешься|неправ)\b",
    r"^(попробуй|давай|переделай) (ещё|еще|снова|заново|по-другому)\b",
    r"^(try again|do it again|redo|rewrite it)\b",
)
_CORRECTION_RE = [re.compile(p, re.IGNORECASE) for p in CORRECTION_MARKERS]
# Beyond this a message is a new task that happens to start with "no".
CORRECTION_MAX_CHARS = 120


@dataclass
class Answered:
    """What we last told this session, and what it cost."""

    request_id: str
    offering_key: str
    task_class: str
    messages_hash: str
    ts: float


@dataclass
class Signal:
    request_id: str
    offering_key: str
    task_class: str
    kind: str
    reward: float
    source: str = "implicit"


def messages_hash(messages: list[ChatMessage]) -> str:
    """Identity of a request, for spotting a regeneration."""
    h = hashlib.sha256()
    for m in messages:
        h.update(m.role.encode())
        h.update(str(m.content).encode("utf-8", "replace"))
    return h.hexdigest()[:32]


def _last_user_text(req: ChatRequest) -> str:
    for m in reversed(req.messages):
        if m.role == "user":
            if isinstance(m.content, str):
                return m.content.strip()
            if isinstance(m.content, list):
                return " ".join(
                    b.get("text", "") for b in m.content if isinstance(b, dict)
                ).strip()
            return ""
    return ""


def looks_like_correction(text: str) -> bool:
    if not text or len(text) > CORRECTION_MAX_CHARS:
        return False
    return any(rx.search(text) for rx in _CORRECTION_RE)


class FeedbackCollector:
    def __init__(
        self,
        telemetry: Any,
        bandit: Any | None = None,
        *,
        limit: int = 4096,
        session_ttl_s: float = 3600.0,
        enabled: bool = True,
    ):
        self.telemetry = telemetry
        self.bandit = bandit
        self.limit = limit
        # Past this the next message is a new conversation, not a
        # reaction: judging an hour-old answer by it would be noise.
        self.session_ttl_s = session_ttl_s
        self.enabled = enabled
        self._last: dict[str, Answered] = {}
        self._judged: set[str] = set()
        self._lock = threading.Lock()
        self._counts: dict[str, int] = {}

    # ------------------------------------------------------------ memory
    def remember(
        self,
        session_id: str | None,
        request_id: str,
        offering_key: str,
        task_class: str,
        messages: list[ChatMessage],
    ) -> None:
        if not self.enabled or not session_id:
            return
        with self._lock:
            if len(self._last) >= self.limit:
                self._last.clear()
                self._judged.clear()
            self._last[session_id] = Answered(
                request_id=request_id,
                offering_key=offering_key,
                task_class=task_class,
                messages_hash=messages_hash(messages),
                ts=time.time(),
            )

    # ------------------------------------------------------------ infer
    def infer(self, req: ChatRequest) -> Signal | None:
        """Read this request as a verdict on the previous answer."""
        if not self.enabled or not req.session_id:
            return None
        with self._lock:
            prev = self._last.get(req.session_id)
            if prev is None or prev.request_id in self._judged:
                return None
            if time.time() - prev.ts > self.session_ttl_s:
                self._last.pop(req.session_id, None)
                return None

        if messages_hash(req.messages) == prev.messages_hash:
            kind = "regenerated"
        elif looks_like_correction(_last_user_text(req)):
            kind = "corrected"
        elif len(req.messages) > 1:
            kind = "continued"
        else:
            return None

        return Signal(
            request_id=prev.request_id,
            offering_key=prev.offering_key,
            task_class=prev.task_class,
            kind=kind,
            reward=REWARDS[kind],
        )

    # ----------------------------------------------------------- record
    def record(self, signal: Signal) -> bool:
        """Apply a signal once. Returns False if this answer was judged."""
        if not self.enabled:
            return False
        with self._lock:
            if signal.request_id in self._judged and signal.source != "explicit":
                return False
            self._judged.add(signal.request_id)
            if len(self._judged) > self.limit * 2:
                self._judged.clear()
            self._counts[signal.kind] = self._counts.get(signal.kind, 0) + 1

        if self.bandit is not None and signal.offering_key:
            self.bandit.update(signal.task_class, signal.offering_key, signal.reward)
        try:
            self.telemetry.log_feedback(
                request_id=signal.request_id,
                offering=signal.offering_key,
                task_class=signal.task_class,
                kind=signal.kind,
                reward=signal.reward,
                source=signal.source,
            )
        except Exception as exc:  # noqa: BLE001 — feedback must not break a request
            log.debug("feedback not logged: %s", exc)
        log.info(
            "FEEDBACK %s %s → reward %.2f (%s)",
            signal.kind,
            signal.offering_key,
            signal.reward,
            signal.source,
        )
        return True

    def rate(self, request_id: str, rating: str, session_id: str | None = None) -> bool:
        """Explicit rating from a client. Overrides anything inferred."""
        kind = "rated_good" if rating in ("good", "up", "1", "positive") else "rated_bad"
        target: Answered | None = None
        with self._lock:
            if session_id and (prev := self._last.get(session_id)):
                if prev.request_id == request_id:
                    target = prev
            if target is None:
                for answered in self._last.values():
                    if answered.request_id == request_id:
                        target = answered
                        break
        if target is None:
            # The answer is gone from memory but the rating is still worth
            # storing: the training set outlives the session cache.
            self.telemetry.log_feedback(
                request_id=request_id,
                offering="",
                task_class="",
                kind=kind,
                reward=REWARDS[kind],
                source="explicit",
            )
            return False
        return self.record(
            Signal(
                request_id=target.request_id,
                offering_key=target.offering_key,
                task_class=target.task_class,
                kind=kind,
                reward=REWARDS[kind],
                source="explicit",
            )
        )

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "enabled": self.enabled,
                "sessions": len(self._last),
                "signals": dict(self._counts),
            }
