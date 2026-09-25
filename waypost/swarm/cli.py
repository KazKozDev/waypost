from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import uuid

from .engine import SwarmEngine
from .models import SwarmConfig
from .store import RunStore


def config_from_args(args) -> SwarmConfig:
    """Flags override defaults; a config field without a flag (the
    collective settings) keeps its default instead of crashing the run."""
    return SwarmConfig(**{field: getattr(args, field) for field in SwarmConfig.model_fields
                          if hasattr(args, field)})


def main(argv=None):
    parser = argparse.ArgumentParser(description="Autonomous Swarms agents through Waypost")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Solve a task")
    task = run.add_mutually_exclusive_group(required=True)
    task.add_argument("--task")
    task.add_argument("--task-file", type=Path)
    run.add_argument("--run-dir", type=Path)
    run.add_argument("--input", type=Path, action="append", default=[], help="Copy a file into workspace/inputs (repeatable)")
    run.add_argument("--base-url", default=os.getenv("WAYPOST_BASE_URL", "http://127.0.0.1:8080/v1"))
    run.add_argument("--model", default="auto")
    run.add_argument("--privacy", choices=["normal", "strict"], default="normal")
    run.add_argument("--concurrency", type=int, default=2)
    run.add_argument("--max-calls", type=int, default=None, help="Optional safety cap; unlimited by default")
    run.add_argument("--max-rounds", type=int, default=None, help="Optional safety cap; unlimited by default")
    run.add_argument("--max-steps", type=int, default=None, help="Optional safety cap; unlimited by default")
    run.add_argument("--max-tasks", type=int, default=None, help="Optional safety cap; unlimited by default")
    run.add_argument("--max-seconds", type=float, default=None, help="Optional safety cap; unlimited by default")
    run.add_argument("--max-tokens", type=int, default=4096)
    run.add_argument("--request-timeout", type=float, default=1500)
    run.add_argument("--allow-python", action="store_true", help="Allow arbitrary Python as your OS user; this is NOT sandboxed")
    resume = commands.add_parser("resume", help="Continue a failed/interrupted run")
    resume.add_argument("run_dir", type=Path)
    resume.add_argument("--acknowledge-interrupted-tools", action="store_true")
    resume.add_argument("--base-url", help="Waypost router to use instead of the saved one")
    status = commands.add_parser("status")
    status.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "status":
            state = RunStore(args.run_dir).load()
        elif args.command == "resume":
            state = SwarmEngine(args.run_dir).run(resume=True,
                acknowledge_interrupted_tools=args.acknowledge_interrupted_tools,
                base_url=args.base_url)
        else:
            directory = args.run_dir or Path("var/swarm") / uuid.uuid4().hex
            if (directory / "state.json").exists():
                raise ValueError("Run directory already contains a run; use resume")
            names = [p.name for p in args.input]
            if len(names) != len(set(names)):
                raise ValueError("Input file names must be unique")
            for source in args.input:
                if not source.is_file():
                    raise ValueError(f"Not an input file: {source}")
            config = config_from_args(args)
            engine = SwarmEngine(directory, config)
            inputs = engine.store.workspace / "inputs"
            inputs.mkdir(exist_ok=True)
            for source in args.input:
                shutil.copy2(source, inputs / source.name)
            task_text = args.task if args.task is not None else args.task_file.read_text()
            print(f"Run: {directory.resolve()}", file=sys.stderr, flush=True)
            state = engine.run(task_text)
        summary = {key: state.get(key) for key in ("status", "phase", "round", "calls", "elapsed_seconds", "error")}
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if state["status"] == "completed" or args.command == "status" else 2
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
