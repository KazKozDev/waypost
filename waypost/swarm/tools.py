"""Workspace tools. Python runs only inside an OS sandbox (macOS Seatbelt):
no network, writes only to the agent's own output directory, no reads of
the user's home outside the run workspace. No sandbox, no execution."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading


class WorkspaceTools:
    _write_lock = threading.RLock()

    def __init__(self, root: Path, output_prefix: str, allow_python: bool = False):
        self.root = root.resolve()
        self.output = self._path(output_prefix)
        self.output.mkdir(parents=True, exist_ok=True)
        self.allow_python = allow_python

    def _path(self, name: str) -> Path:
        path = (self.root / name).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Path must stay inside the run workspace")
        return path

    def describe(self, read_only: bool = False) -> str:
        tools = {
            "list_files": {"path": "relative directory, default ."},
            "read_file": {"path": "relative UTF-8 file", "offset": "optional character offset"},
            "write_file": {"path": "relative filename inside your output directory", "content": "UTF-8 text"},
        }
        if self.allow_python:
            tools["run_python"] = {"code": "Python source; cwd is your output directory. Sandboxed: no "
                                           "network, writes only in your output directory"}
        if read_only:
            tools.pop("write_file")
        return json.dumps({"tools": tools, "your_output_directory": str(self.output.relative_to(self.root)),
                           "read_limit_chars": 20000, "write_limit_chars": 200000})

    def execute(self, name: str, arguments: dict, timeout: float = 30) -> str:
        if name == "list_files":
            path = self._path(arguments.get("path", "."))
            return json.dumps([str(p.relative_to(self.root)) + ("/" if p.is_dir() else "")
                               for p in sorted(path.iterdir())][:500])
        if name == "read_file":
            path = self._path(arguments["path"])
            if not path.exists():
                raise ValueError(f"File not found: {arguments['path']} — call list_files to see what exists")
            if not path.is_file():
                raise ValueError(f"{arguments['path']} is a directory; call list_files on it")
            if path.stat().st_size > 2_000_000:
                raise ValueError("File is larger than 2 MB")
            offset = int(arguments.get("offset", 0))
            if offset < 0:
                raise ValueError("offset must be nonnegative")
            text = path.read_text()
            return json.dumps({"content": text[offset:offset + 20000], "total_chars": len(text),
                               "offset": offset}, ensure_ascii=False)
        if name == "write_file":
            requested = Path(arguments["path"])
            prefix = self.output.relative_to(self.root)
            # Agents sometimes echo the advertised workspace-relative output
            # directory. Accept that spelling as well as a bare filename.
            base = self.root if requested.parts[:len(prefix.parts)] == prefix.parts else self.output
            path = (base / requested).resolve()
            if not path.is_relative_to(self.output):
                raise ValueError("Writes must stay inside your output directory")
            content = arguments["content"]
            if not isinstance(content, str) or len(content) > 200000:
                raise ValueError("content must be text of at most 200000 characters")
            with self._write_lock:
                # Another agent's file of the same name is worth knowing about,
                # not a reason to refuse: the refusal trapped two synthesizers
                # in a read-write loop for 97 calls on a live run.
                others = []
                output_parts = prefix.parts
                if len(output_parts) >= 3 and output_parts[0] == "artifacts" and output_parts[1].startswith("r"):
                    relative_path = path.relative_to(self.output)
                    for sibling in self.output.parent.iterdir():
                        if sibling != self.output and sibling.is_dir() and (sibling / relative_path).is_file():
                            others.append(str((sibling / relative_path).relative_to(self.root)))
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
            result = {"written": str(path.relative_to(self.root))}
            if others:
                result["note"] = "other agents have a file with this name: " + ", ".join(others)
            return json.dumps(result)
        if name == "run_python" and self.allow_python:
            return self._python(arguments["code"], min(timeout, 30))
        raise ValueError(f"Unknown or disabled tool: {name}")

    SANDBOX_EXEC = "/usr/bin/sandbox-exec"

    @staticmethod
    def _sb_path(path) -> str:
        return '"' + str(path).replace("\\", "\\\\").replace('"', '\\"') + '"'

    def _sandbox_profile(self) -> str:
        """Allow ordinary work; deny what could hurt the user. Everything the
        code starts inherits the profile. Seatbelt applies the last matching
        rule, so each deny comes before its narrower allow."""
        home = Path.home().resolve()
        readable = {self.root, Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve(),
                    Path(sys.executable).resolve().parent}
        q = self._sb_path
        lines = ["(version 1)", "(allow default)", "(deny network*)",
                 "(deny file-write*)",
                 f"(allow file-write* (subpath {q(self.output)}) (literal \"/dev/null\") (literal \"/dev/tty\"))",
                 f"(deny file-read* (subpath {q(home)}))"]
        lines += [f"(allow file-read* (subpath {q(p)}))" for p in sorted(readable, key=str)]
        return "\n".join(lines)

    def _python(self, code: str, timeout: float) -> str:
        if not isinstance(code, str) or len(code) > 50000:
            raise ValueError("Python code must be text of at most 50000 characters")
        if sys.platform != "darwin" or not os.access(self.SANDBOX_EXEC, os.X_OK):
            # Fail closed: agent-written code never runs unconfined.
            raise ValueError("Python execution is unavailable: no OS sandbox on this machine")
        # No provider keys inherited, and HOME/TMPDIR point into the output dir.
        env = {k: os.environ[k] for k in ("PATH", "LANG") if k in os.environ}
        env.update(HOME=str(self.output), TMPDIR=str(self.output))
        command = [self.SANDBOX_EXEC, "-p", self._sandbox_profile(), sys.executable, "-I", "-c", code]
        with tempfile.TemporaryFile() as output:
            proc = subprocess.Popen(command, cwd=self.output,
                                    env=env, stdout=output, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            timed_out = False
            try:
                proc.wait(timeout=max(0.01, timeout))
            except subprocess.TimeoutExpired:
                timed_out = True
            finally:
                # Also remove background descendants after the main process exits.
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
            output.seek(0)
            result = output.read(20001)
        return json.dumps({"exit_code": proc.returncode, "timed_out": timed_out,
                           "output": result[:20000].decode(errors="replace"),
                           "truncated": len(result) > 20000})
