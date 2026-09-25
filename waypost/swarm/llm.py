"""Swarms custom-LLM adapter. No direct provider fallback is configured here."""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
import httpx

from ..families import model_family
from .models import SwarmConfig
from .store import RunStore


class BudgetExceeded(RuntimeError):
    pass


class RunInterrupted(RuntimeError):
    pass


class Budget:
    def __init__(self, config: SwarmConfig, state: dict, store: RunStore):
        self.config, self.state, self.store = config, state, store
        self.lock = threading.RLock()
        self.started = time.monotonic()
        self.previous_seconds = state.get("elapsed_seconds", 0)

    def remaining(self):
        if self.config.max_seconds is None:
            return float("inf")
        return self.config.max_seconds - self.previous_seconds - (time.monotonic() - self.started)

    def check(self):
        control = self.store.control()
        if control.get("interrupt"):
            raise RunInterrupted("Interrupted by user")
        if control.get("paused"):
            with self.lock:
                if self.state.get("status") != "paused":
                    self.state["status"] = "paused"
                    self.checkpoint()
                    self.store.event("paused")
            while control.get("paused"):
                time.sleep(0.2)
                control = self.store.control()
                if control.get("interrupt"):
                    raise RunInterrupted("Interrupted by user")
            with self.lock:
                self.state["status"] = "running"
                self.checkpoint()
                self.store.event("resumed")
        if self.remaining() <= 0:
            raise BudgetExceeded("Execution time budget exhausted")

    def reserve(self):
        with self.lock:
            self.check()
            if self.config.max_calls is not None and self.state["calls"] >= self.config.max_calls:
                raise BudgetExceeded("LLM call budget exhausted")
            self.state["calls"] += 1
            self.checkpoint()

    def checkpoint(self):
        with self.lock:
            self.state["elapsed_seconds"] = self.previous_seconds + time.monotonic() - self.started
            self.store.save(self.state)


class EmptyAnswer(ValueError):
    pass


class Answer(str):
    """Model text that remembers which model family wrote it — the
    collective needs that to ask the next member someone else."""
    family: str | None = None

    def __new__(cls, text: str, family: str | None = None):
        value = super().__new__(cls, text)
        value.family = family
        return value


class WaypostLLM:
    def __init__(self, config: SwarmConfig, budget: Budget, store: RunStore,
                 session: str, system: str, client: httpx.Client | None = None,
                 schema: dict | None = None, avoid_families: list[str] | None = None):
        self.config, self.budget, self.store = config, budget, store
        self.session, self.system, self.client = session, system, client
        self.schema = schema
        self.avoid_families = avoid_families
        self.last_family: str | None = None
        self.last_response: str | None = None
        self.last_error: Exception | None = None

    def run(self, task: str | None = None, **kwargs) -> str:
        self.last_response = None
        self.last_error = None
        try:
            try:
                return self._run(task, kwargs.get("messages"))
            except EmptyAnswer as exc:
                # A 200 with no text is a bad sample, not a routing failure:
                # one fresh draw is cheaper than failing the whole run.
                self.store.event("llm_error", session=self.session, status=200,
                                 error=f"{exc} — повтор")
                return self._run(task, kwargs.get("messages"))
        except Exception as exc:
            self.last_error = exc
            raise

    def _run(self, task: str | None, messages: list[dict] | None = None) -> str:
        # Swarms 15 may call custom LLMs with task=None and messages=[...].
        # Its prompt is in messages; sending None as user content fails Waypost validation.
        conversation = [{"role": "system", "content": self.system}]
        if messages:
            conversation.extend({"role": m["role"], "content": m["content"]}
                                for m in messages if m.get("role") in {"user", "assistant"}
                                and isinstance(m.get("content"), str))
        elif isinstance(task, str):
            conversation.append({"role": "user", "content": task})
        if len(conversation) == 1:
            raise ValueError("Swarms supplied no text prompt")
        self.budget.reserve()
        payload = {
            "model": self.config.model,
            "messages": conversation,
            "max_tokens": self.config.max_tokens, "temperature": 0.2,
            "response_format": {"type": "json_object"},
            "stream": False, "session_id": self.session,
            "latency_class": "batch", "no_cache": True,
            "idempotency_key": str(uuid.uuid4()),
        }
        if self.schema is not None:
            # Lets Waypost's verifier reject valid JSON with the wrong
            # fields and move to another model, instead of the swarm
            # finding out after the fact.
            payload["output_schema"] = self.schema
        if self.avoid_families:
            payload["avoid_families"] = self.avoid_families
        if self.config.privacy == "strict":
            payload["privacy"] = "strict"
        timeout = min(self.config.request_timeout, self.budget.remaining())
        headers = {"Authorization": "Bearer " + os.getenv("WAYPOST_API_KEY", "unused")}
        # A router call can take minutes while Waypost walks its ladder; say
        # it started, or the UI shows nothing until it ends.
        self.store.event("llm_request", session=self.session)
        started = time.monotonic()
        try:
            # No implicit HTTP retries: Waypost owns upstream retry/quota accounting.
            if self.client is not None:
                response = self.client.post(self.config.base_url.rstrip("/") + "/chat/completions",
                                            json=payload, headers=headers, timeout=timeout)
            else:
                with httpx.Client(trust_env=False) as client:
                    response = client.post(self.config.base_url.rstrip("/") + "/chat/completions",
                                           json=payload, headers=headers, timeout=timeout)
        except httpx.HTTPError as exc:
            self.store.event("llm_error", session=self.session, status=None,
                             error=f"{type(exc).__name__}: {exc}",
                             seconds=round(time.monotonic() - started, 1))
            raise
        if response.is_error:
            try:
                body = response.json()
            except ValueError:
                body = {}
            error = body.get("error") if isinstance(body, dict) else None
            message = error.get("message") if isinstance(error, dict) else None
            self.store.event("llm_error", session=self.session, status=response.status_code,
                             error=message or response.text[:500],
                             router=body.get("router", {}) if isinstance(body, dict) else {},
                             seconds=round(time.monotonic() - started, 1))
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            router = data.get("router", {})
            raise ValueError(f"Waypost returned no choices (provider={router.get('provider')}, model={router.get('model')})")
        choice = choices[0]
        if choice.get("finish_reason") == "length":
            raise ValueError("Waypost response was truncated; increase max_tokens")
        content = choice["message"].get("content")
        if not isinstance(content, str) or not content.strip():
            raise EmptyAnswer("Waypost returned no text")
        model = data.get("router", {}).get("model")
        self.last_family = model_family(model) if model else None
        self.store.event("llm_response", session=self.session,
                         router=data.get("router", {}), usage=data.get("usage", {}),
                         seconds=round(time.monotonic() - started, 1))
        self.last_response = content
        return content


class SwarmsBackend:
    def __init__(self, config: SwarmConfig, budget: Budget, store: RunStore):
        # Import only when the optional engine is actually run.
        os.environ["SWARMS_TELEMETRY_ON"] = "false"
        os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
        try:
            from swarms import Agent
        except ModuleNotFoundError as exc:
            if exc.name == "swarms":
                raise RuntimeError("Install the optional engine: pip install -e '.[swarm]'") from exc
            raise
        self.agent_class = Agent
        self.config, self.budget, self.store = config, budget, store

    def ask(self, role: str, system: str, prompt: str, schema: dict | None = None,
            avoid_families: list[str] | None = None) -> str:
        llm = WaypostLLM(self.config, self.budget, self.store,
                         f"{self.store.directory.name}:{role}", system, schema=schema,
                         avoid_families=avoid_families)
        agent = self.agent_class(
            agent_name=role, system_prompt=system, llm=llm,
            model_name="openai/auto", max_loops=1, retry_attempts=1,
            output_type="final", autosave=False, print_on=False,
            streaming_on=False, dynamic_temperature_enabled=False,
            context_compression=False, dynamic_context_window=False,
            reasoning_prompt_on=False,
        )
        try:
            agent.run(prompt)
        except Exception:
            if llm.last_error is not None:
                raise llm.last_error
            raise
        # Read the actual adapter response, never an error string swallowed by Agent.
        if llm.last_error is not None:
            raise llm.last_error
        if llm.last_response is None:
            raise RuntimeError("Swarms did not invoke the Waypost adapter")
        return Answer(llm.last_response, llm.last_family)


def parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    result = json.loads(text)
    if not isinstance(result, dict):
        raise ValueError("Expected one JSON object")
    return result
