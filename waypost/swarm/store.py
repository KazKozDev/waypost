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
