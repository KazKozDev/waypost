"""Workspace tools. Python runs only inside an OS sandbox (macOS Seatbelt):
writes only to the agent's own output directory, no reads of the user's
home outside the run workspace, never this machine's own services
(localhost); the internet is allowed unless turned off. No sandbox, no
execution."""
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

    def __init__(self, root: Path, output_prefix: str, allow_python: bool = False,
                 allow_network: bool = False):
        self.root = root.resolve()
        self.output = self._path(output_prefix)
        self.output.mkdir(parents=True, exist_ok=True)
        # Where code keeps its mess, outside the deliverables: HOME, temp and
        # caches per agent; installed packages once for the whole swarm.
        self.scratch = self.root / ".scratch" / self.output.relative_to(self.root)
        self.lib = self.root / ".lib"
        self.allow_python = allow_python
        self.allow_network = allow_network

    def _path(self, name: str) -> Path:
        path = (self.root / name).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Path must stay inside the run workspace")
        return path

    def describe(self, read_only: bool = False) -> str:
        tools = {
            "list_files": {"path": "directory; default: your own output directory ('/' lists the workspace)"},
            "read_file": {"path": "relative UTF-8 file", "offset": "optional character offset"},
            "write_file": {"path": "relative filename inside your output directory", "content": "UTF-8 text"},
        }
        if self.allow_python:
            tools["run_python"] = {"code": "Python source; cwd is your output directory. Sandboxed: writes "
                                           "only in your output directory; " + (
                                               "internet allowed (not localhost); `pip install <pkg>` goes "
                                               "to the swarm's shared library and is importable at once by "
                                               "every agent" if self.allow_network
                                               else "no network")}
        if read_only:
            tools.pop("write_file")
        return json.dumps({"tools": tools, "your_output_directory": str(self.output.relative_to(self.root)),
                           "read_limit_chars": 20000, "write_limit_chars": 200000})

    def _resolve(self, name: str) -> Path:
        """A path as the agent means it. write_file("x.py") lands in the
        agent's own directory, so read_file("x.py") must find it there: a
        live run looped for 20 calls reading and listing a file it had just
        written, because reads resolved from the workspace root."""
        own = (self.output / name).resolve()
        if name not in ("", ".") and own.is_relative_to(self.root) and own.exists():
            return own
        return self._path(name)

    def execute(self, name: str, arguments: dict, timeout: float = 30) -> str:
        if name == "list_files":
            requested = arguments.get("path")
            # Default: your own directory — that is what you are working in.
            path = (self.output if requested in (None, "", ".") else
                    self.root if requested == "/" else self._resolve(requested))
            if not path.is_dir():
                raise ValueError(f"Not a directory: {requested}")
            entries = [p for p in sorted(path.iterdir()) if not (path == self.root and p.name.startswith("."))]
            return json.dumps({"directory": str(path.relative_to(self.root)) if path != self.root else ".",
                               "entries": [str(p.relative_to(self.root)) + ("/" if p.is_dir() else "")
                                           for p in entries][:500]})
        if name == "read_file":
            path = self._resolve(arguments["path"])
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
        # The internet when allowed; this machine's own services never — the
        # router, Ollama and anything else listening on localhost stay out.
        network = ('(deny network-outbound (remote ip "localhost:*"))' if self.allow_network
                   else "(deny network*)")
        lines = ["(version 1)", "(allow default)", network,
                 "(deny file-write*)",
                 f"(allow file-write* (subpath {q(self.output)}) (subpath {q(self.scratch)}) "
                 f"(subpath {q(self.lib)}) (literal \"/dev/null\") (literal \"/dev/tty\"))",
                 f"(deny file-read* (subpath {q(home)}))",
                 # Seeing that a path exists (stat/realpath) is not reading it:
                 # a Python in a venv under the home walks its own path on
                 # startup and died of this. Contents and listings stay denied.
                 "(allow file-read-metadata)"]
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
        self.scratch.mkdir(parents=True, exist_ok=True)
        self.lib.mkdir(parents=True, exist_ok=True)
        # pip reads PIP_* from the environment even under -I: installs land in
        # the shared swarm library, caches in scratch — never in the result.
        env.update(HOME=str(self.scratch), TMPDIR=str(self.scratch), PIP_TARGET=str(self.lib),
                   PIP_CACHE_DIR=str(self.scratch / "pip-cache"), PIP_DISABLE_PIP_VERSION_CHECK="1",
                   SWARM_LIB=str(self.lib))
        # -I drops the working directory and ignores PYTHONPATH: put the
        # agent's own directory (its modules) and the shared library back.
        # Without the first, `import hello_world` failed on a file the agent
        # had just written, and a hello-world run lost an attempt to it.
        code = (f"import sys as _s; _s.path[:0] = [{str(self.output)!r}, {str(self.lib)!r}]; del _s\n"
                + code)
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
