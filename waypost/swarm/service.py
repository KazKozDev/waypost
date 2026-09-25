"""Local UI control plane for persistent swarm runs."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid

from .store import RunStore


RUN_ID = re.compile(r"^[0-9a-f]{32}$")


class SwarmService:
    def __init__(self, root: Path, base_url: str):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.base_url = base_url
        self._processes: dict[int, subprocess.Popen] = {}

    def directory(self, run_id: str) -> Path:
        if not RUN_ID.fullmatch(run_id):
            raise ValueError("Invalid run ID")
        directory = self.root / run_id
        if not directory.is_dir():
            raise FileNotFoundError("Swarm run not found")
        return directory

    def _launch(self, directory: Path, *, resume: bool = False,
                model: str = "auto", privacy: str = "normal", allow_python: bool = False):
        if importlib.util.find_spec("swarms") is None:
            raise RuntimeError("Swarms is not installed. Run: pip install -e '.[swarm]'")
        command = [sys.executable, "-m", "waypost.swarm"]
        if resume:
            command += ["resume", str(directory), "--acknowledge-interrupted-tools"]
        else:
            command += ["run", "--task-file", str(directory / "task.txt"),
                        "--run-dir", str(directory), "--base-url", self.base_url,
                        "--model", model, "--privacy", privacy]
            if allow_python:
                command.append("--allow-python")
        with (directory / "runner.log").open("a") as output:
            process = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[2],
                                       stdout=output, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, start_new_session=True)
        self._processes[process.pid] = process
        (directory / "process.json").write_text(json.dumps({"pid": process.pid}))

    def create(self, task: str, *, model: str = "auto", privacy: str = "normal",
               allow_python: bool = False) -> str:
        if not task.strip():
            raise ValueError("Task must not be empty")
        if len(task) > 200_000:
            raise ValueError("Task is too long")
        if privacy not in {"normal", "strict"}:
            raise ValueError("Invalid privacy mode")
        if importlib.util.find_spec("swarms") is None:
            raise RuntimeError("Swarms is not installed. Run: pip install -e '.[swarm]'")
        run_id = uuid.uuid4().hex
        directory = self.root / run_id
        directory.mkdir()
        (directory / "task.txt").write_text(task)
        RunStore(directory).event("user_task", text=task)
        self._launch(directory, model=model, privacy=privacy, allow_python=allow_python)
        return run_id

    def _running(self, directory: Path) -> bool:
        path = directory / "process.json"
        if not path.exists():
            return False
        try:
            pid = json.loads(path.read_text())["pid"]
            if pid in self._processes:
                return self._processes[pid].poll() is None
            os.kill(pid, 0)
            return True
        except (ValueError, KeyError, OSError):
            return False

    def status(self, run_id: str) -> dict:
        directory = self.directory(run_id)
        state_path = directory / "state.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        control = RunStore(directory).control()
        status = state.get("status", "starting")
        if status in {"starting", "running", "paused"} and not self._running(directory):
            status = "interrupted"
        return {"id": run_id, "status": status, "phase": state.get("phase"),
                "round": state.get("round"), "calls": state.get("calls", 0),
                "task": state.get("task") or (directory / "task.txt").read_text(),
                "draft": state.get("draft", ""), "error": state.get("error"),
                "paused": bool(control.get("paused")),
                "pending_messages": control.get("messages", []),
                "running": self._running(directory),
                "monitor": state.get("monitor", [])[-5:],
                "artifacts": [str(p.relative_to(directory / "workspace")) for p in
                              sorted((directory / "workspace" / "artifacts").rglob("*")) if p.is_file()]}

    def list_runs(self) -> list[dict]:
        directories = sorted((p for p in self.root.iterdir() if p.is_dir() and RUN_ID.fullmatch(p.name)),
                             key=lambda p: p.stat().st_mtime, reverse=True)
        return [self.status(path.name) for path in directories[:30]]

    def events(self, run_id: str, offset: int = 0) -> dict:
        if offset < 0:
            raise ValueError("Invalid event offset")
        path = self.directory(run_id) / "events.jsonl"
        if not path.exists():
            return {"events": [], "next_offset": 0}
        lines = path.read_text().splitlines()
        events = []
        for line in lines[offset:offset + 100]:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("event") == "tool_started":
                item["arguments"] = {k: (v[:500] + "…" if isinstance(v, str) and len(v) > 500 else v)
                                     for k, v in item.get("arguments", {}).items()}
            if item.get("event") == "tool_finished":
                item["observation"] = str(item.get("observation", ""))[:1500]
            if item.get("event") == "invalid_output":
                item.pop("output", None)
            events.append(item)
        return {"events": events, "next_offset": min(len(lines), offset + 100)}

    def message(self, run_id: str, text: str):
        if not text.strip():
            raise ValueError("Message must not be empty")
        directory = self.directory(run_id)
        store = RunStore(directory)
        store.update_control(message=text)
        store.event("user_message_queued", text=text)
        # An explicit interruption stays stopped while the user writes a
        # correction; the Continue control starts it again.
        if (not self._running(directory) and not store.control().get("paused")
                and self.status(run_id)["status"] != "interrupted"):
            self._launch(directory, resume=True)

    def pause(self, run_id: str):
        directory = self.directory(run_id)
        RunStore(directory).update_control(paused=True)
        RunStore(directory).event("pause_requested")

    def interrupt(self, run_id: str):
        directory = self.directory(run_id)
        RunStore(directory).update_control(paused=False, interrupt=True)
        RunStore(directory).event("interrupt_requested")

    def resume(self, run_id: str):
        directory = self.directory(run_id)
        RunStore(directory).update_control(paused=False, interrupt=False)
        RunStore(directory).event("resume_requested")
        if not self._running(directory):
            self._launch(directory, resume=True)
