"""Atomic checkpoints and an append-only execution journal; single runner per run."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import threading


class RunStore:
    def __init__(self, directory: Path):
        self.directory = directory.resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.workspace = self.directory / "workspace"
        self.workspace.mkdir(exist_ok=True)
        self._lock = threading.RLock()

    @contextmanager
    def exclusive(self):
        with (self.directory / "run.lock").open("a") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("This run is already active in another process") from exc
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def load(self) -> dict:
        return json.loads((self.directory / "state.json").read_text())

    def save(self, state: dict):
        with self._lock:
            tmp = self.directory / "state.json.tmp"
            tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2))
            tmp.replace(self.directory / "state.json")

    def event(self, kind: str, **data):
        with self._lock, (self.directory / "events.jsonl").open("a") as handle:
            handle.write(json.dumps({"time": datetime.now(timezone.utc).isoformat(),
                                     "event": kind, **data}, ensure_ascii=False) + "\n")

    def control(self) -> dict:
        path = self.directory / "control.json"
        if not path.exists():
            return {"paused": False, "interrupt": False, "messages": []}
        return json.loads(path.read_text())

    def update_control(self, *, paused: bool | None = None,
                       interrupt: bool | None = None, message: str | None = None,
                       take_messages: bool = False) -> dict:
        """Exchange control with a running engine across processes."""
        with (self.directory / "control.lock").open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                value = self.control()
                taken = list(value.get("messages", [])) if take_messages else []
                if take_messages:
                    value["messages"] = []
                if paused is not None:
                    value["paused"] = paused
                if interrupt is not None:
                    value["interrupt"] = interrupt
                if message is not None:
                    value.setdefault("messages", []).append({
                        "time": datetime.now(timezone.utc).isoformat(), "text": message})
                tmp = self.directory / "control.json.tmp"
                tmp.write_text(json.dumps(value, ensure_ascii=False))
                tmp.replace(self.directory / "control.json")
                return {**value, "taken_messages": taken}
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)
