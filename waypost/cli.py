"""CLI for waypost.

    waypost                    # start the server (as before)
    waypost keys               # interactively enter all keys
    waypost keys set NAME VAL  # set one key
    waypost keys list          # which keys are set (masked)
    waypost keys status        # which providers are available
    waypost models             # models from the manifest
    waypost discover           # auto-discovery of free models
    waypost probe              # probing (TTFT, limits, capability)
    waypost stats              # stats of the running server
    waypost report             # telemetry report (--json/--csv)
    waypost train-head --data  # train the L1 classifier head

Keys are written to .env — the same file the server reads. No separate
database: a single source of truth.
"""
from __future__ import annotations

import argparse
import asyncio
import getpass
import json
import sys
from pathlib import Path

import httpx

# Variable name → a hint where to get the key. Free tiers only: there
# are no paid providers in the manifest.
KEYS = [
    ("OPENROUTER_API_KEY", "openrouter.ai/workspaces/default/keys"),
    ("GROQ_API_KEY", "console.groq.com/keys"),
    ("CEREBRAS_API_KEY", "cloud.cerebras.ai"),
    ("SILICONFLOW_API_KEY", "cloud.siliconflow.cn"),
    ("ZHIPU_API_KEY", "open.bigmodel.cn"),
    ("NVIDIA_API_KEY", "build.nvidia.com/settings/api-keys"),
    ("MISTRAL_API_KEY", "console.mistral.ai/api-keys"),
    ("GEMINI_API_KEY", "aistudio.google.com/app/apikey"),
]

ENV_PATH = Path(".env")


# ------------------------------------------------------------- .env utils


def _read_env() -> dict[str, str]:
    if not ENV_PATH.exists():
        return {}
    out: dict[str, str] = {}
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        out[k.strip()] = v.strip()
    return out


def _write_env(env: dict[str, str]) -> None:
    lines = (
        ENV_PATH.read_text(encoding="utf-8").splitlines() if ENV_PATH.exists() else []
    )
    seen: set[str] = set()
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            k = stripped.partition("=")[0].strip()
            if k in env:
                out.append(f"{k}={env[k]}")
                seen.add(k)
                continue
        out.append(line)
    for k, v in env.items():
        if k not in seen:
            out.append(f"{k}={v}")
    ENV_PATH.write_text("\n".join(out) + "\n", encoding="utf-8")


def _mask(v: str) -> str:
    if not v:
        return "(not set)"
    return v[:4] + "…" + v[-4:] if len(v) > 8 else "••••"


# ---------------------------------------------------------------- keys


def cmd_keys_set(args: argparse.Namespace) -> None:
    env = _read_env()
    env[args.name] = args.value
    _write_env(env)
    print(f"{args.name} written to {ENV_PATH}")


def cmd_keys_list(_: argparse.Namespace) -> None:
    from . import keychain

    env = _read_env()
    for name, _ in KEYS:
        value, source = env.get(name, ""), ".env"
        if not value and (kc := keychain.get(name)):
            value, source = kc, "keychain"
        extra = [
            f"{name}_{i}"
            for i in range(2, 6)
            if env.get(f"{name}_{i}") or keychain.get(f"{name}_{i}")
        ]
        suffix = f"  +{len(extra)} extra key(s)" if extra else ""
        print(
            f"  {name:<22} {_mask(value):<16} {source if value else '':<9}" f"{suffix}"
        )


def cmd_keys_keychain(args: argparse.Namespace) -> None:
    """Move a key to Keychain: .env is plaintext secrets next to the
    code, Keychain solves exactly that problem."""
    from . import keychain

    if not keychain.available():
        print("Keychain is available on macOS only — staying on .env")
        return
    env = _read_env()
    names = [args.name] if args.name else [n for n, _ in KEYS if env.get(n)]
    moved = 0
    for name in names:
        value = env.get(name)
        if not value:
            print(f"  {name}: no value in .env")
            continue
        if keychain.put(name, value):
            moved += 1
            if args.purge:
                env.pop(name, None)
            print(
                f"  {name}: moved to Keychain"
                + (" and removed from .env" if args.purge else "")
            )
        else:
            print(f"  {name}: failed to write")
    if args.purge and moved:
        _write_env(env)
    print(f"keys moved: {moved}. Lookup order: " f"environment variable → Keychain")


def cmd_keys_status(_: argparse.Namespace) -> None:
    from .config import Settings
    from .registry import Registry

    reg = Registry.from_manifest(Settings().manifest_path)
    for name, _ in KEYS:
        usable = any(
            o.provider == name.split("_")[0].lower() and o.usable for o in reg.all()
        )
        print(f"  {name:<22} {'available' if usable else 'no key'}")


def cmd_keys_interactive(_: argparse.Namespace) -> None:
    env = _read_env()
    print("Enter keys (empty value to skip).")
    for name, where in KEYS:
        current = env.get(name, "")
        prompt = f"{name} [{where}]"
        if current:
            prompt += f" (current {_mask(current)})"
        val = getpass.getpass(f"  {prompt}: ").strip()
        if val:
            env[name] = val
    _write_env(env)
    print(f"Done. Keys saved to {ENV_PATH}")


# --------------------------------------------------------------- models


def cmd_models(args: argparse.Namespace) -> None:
    from .config import Settings
    from .registry import Registry

    st = Settings()
    reg = Registry.from_manifest(st.manifest_path)
    free_only = st.free_only and not getattr(args, "all", False)
    hidden = 0
    for o in sorted(reg.all(), key=lambda x: (x.provider, x.tier.value)):
        if free_only and not o.free:
            hidden += 1
            continue
        mark = " (local)" if o.is_local else ""
        money = "free" if o.free else "PAID"
        print(
            f"  {o.key:<55} [{o.tier.value}] ctx={o.ctx_window}"
            f" {money}/{o.free_source}{mark}"
        )
    if hidden:
        print(f"\n  hidden paid: {hidden} (show: waypost models --all)")


# ------------------------------------------------------------- discover


def cmd_discover(args: argparse.Namespace) -> None:
    from .config import Settings
    from .discovery import discover_provider
    from .providers.openai_compat import OpenAICompatAdapter
    from .registry import Registry

    settings = Settings()
    registry = Registry.from_manifest(settings.manifest_path)
    seen: set[str] = set()
    providers = []
    for o in registry.all():
        if o.is_local or o.provider in seen:
            continue
        seen.add(o.provider)
        if args.provider and o.provider != args.provider:
            continue
        if o.provider != "openrouter" and not o.api_key:
            continue
        providers.append(o)

    if not providers:
        print("no providers to poll (need keys: waypost keys)")
        return

    async def _run():
        async with httpx.AsyncClient() as client:
            adapter = OpenAICompatAdapter(client)
            for p in providers:
                r = await discover_provider(adapter, registry, p)
                print(f"\n== {r['provider']} ({r.get('status')}) ==")
                if r.get("detail"):
                    print(f"   {r['detail']}")
                    continue
                if r.get("gone"):
                    print(f"   gone: {', '.join(r['gone'])}")
                if r.get("added"):
                    print(f"   added: {', '.join(r['added'])}")
                for o in sorted(
                    (
                        x
                        for x in registry.all()
                        if x.provider == p.provider and not x.is_local
                    ),
                    key=lambda x: x.model_id,
                ):
                    mark = " (auto)" if o.weight < 1.0 else ""
                    print(
                        f"   - {o.model_id}  [{o.tier.value}] ctx={o.ctx_window}{mark}"
                    )

    asyncio.run(_run())


# ----------------------------------------------------------------- probe


def cmd_probe(args: argparse.Namespace) -> None:
    from scripts.probe import main as probe_main

    argv = ["probe"]
    if args.provider:
        argv.append(f"--provider={args.provider}")
    if args.tier:
        argv.append(f"--tier={args.tier}")
    if args.all:
        argv.append("--all")
    if args.json:
        argv.append("--json")
    if args.apply:
        argv.append("--apply")
    sys.argv = argv
    asyncio.run(probe_main())


# ----------------------------------------------------------------- stats


def cmd_stats(args: argparse.Namespace) -> None:
    from .config import Settings

    base = f"http://{Settings().host}:{Settings().port}"
    r = httpx.get(f"{base}/v1/stats", timeout=5.0)
    if r.status_code != 200:
        print(f"server not responding at {base} ({r.status_code})")
        return
    data = r.json()
    print("Quotas:")
    for key, buckets in data.get("quota", {}).items():
        print(f"  {key:<40} " + " ".join(f"{k}={v}" for k, v in buckets.items()))
    print(
        f"Cache: exact hit_rate={data.get('cache_hit_rate')}, "
        f"semantic={data.get('semantic_hit_rate')}"
    )
    print("Last 24h:")
    for row in data.get("last_24h", [])[:10]:
        print(
            f"  {row['offering']:<40} attempts={row['attempts']} "
            f"ok={row['success_rate']} lat={row['avg_latency_ms']}ms"
        )


# ------------------------------------------------------- jobs and batches


def _server_base() -> str:
    from .config import Settings

    s = Settings()
    return f"http://{s.host}:{s.port}"


def _get(path: str, **kw):
    r = httpx.get(_server_base() + path, timeout=10.0, **kw)
    r.raise_for_status()
    return r.json()


def cmd_jobs(args: argparse.Namespace) -> None:
    if args.run:
        r = httpx.post(f"{_server_base()}/v1/jobs/{args.run}/run", timeout=300.0)
        print(json.dumps(r.json(), ensure_ascii=False, indent=2)[:2000])
        return
    for name, j in _get("/v1/jobs")["jobs"].items():
        state = "on" if j["enabled"] else "off"
        every = (
            f"{j['interval_s'] / 3600:.1f}h"
            if j["interval_s"] >= 3600
            else f"{j['interval_s'] / 60:.0f}m"
        )
        print(
            f"  {name:<12} {state:<5} every {every:<7} "
            f"runs={j['runs']} failures={j['failures']}"
            + (f"  {j['last_error']}" if j["last_error"] else "")
        )


def cmd_batch(args: argparse.Namespace) -> None:
    base = _server_base()
    if args.batch_cmd == "submit":
        with open(args.file, encoding="utf-8") as f:
            payload = {"input_jsonl": f.read()}
        if args.window:
            payload["completion_window_h"] = args.window
        r = httpx.post(f"{base}/v1/batches", json=payload, timeout=30.0)
        b = r.json()
        print(
            f"{b.get('id')}  {b.get('status')}  "
            f"items: {b.get('request_counts', {}).get('total')}"
        )
    elif args.batch_cmd == "status":
        b = _get(f"/v1/batches/{args.id}")
        print(f"{b['id']}  {b['status']}  {b['request_counts']}")
    elif args.batch_cmd == "output":
        for row in _get(f"/v1/batches/{args.id}/output")["data"]:
            print(json.dumps(row, ensure_ascii=False))
    elif args.batch_cmd == "cancel":
        r = httpx.post(f"{base}/v1/batches/{args.id}/cancel", timeout=10.0)
        print(r.json().get("status"))
    else:
        for b in _get("/v1/batches")["data"]:
            print(f"  {b['id']}  {b['status']:<10} {b['request_counts']}")


# ------------------------------------------------------------ train-head


def cmd_train_head(args: argparse.Namespace) -> None:
    from scripts.train_head import main as train_main

    sys.argv = ["train-head"] + ([f"--data={args.data}"] if args.data else [])
    train_main()


# ---------------------------------------------------------------- main


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="waypost")
    sub = ap.add_subparsers(dest="cmd")

    k = sub.add_parser("keys", help="manage API keys")
    ksub = k.add_subparsers(dest="keys_cmd")
    ksub.add_parser("list", help="which keys are set")
    ksub.add_parser("status", help="which providers are available")
    p_set = ksub.add_parser("set", help="set one key")
    p_set.add_argument("name")
    p_set.add_argument("value")
    p_kc = ksub.add_parser("keychain", help="move keys from .env to macOS Keychain")
    p_kc.add_argument("name", nargs="?", help="one key; without a name — all")
    p_kc.add_argument(
        "--purge", action="store_true", help="remove moved keys from .env"
    )

    m = sub.add_parser("models", help="models from the manifest")
    m.add_argument(
        "--all", action="store_true", help="show paid too (hidden by default)"
    )

    d = sub.add_parser("discover", help="auto-discovery of free models")
    d.add_argument("--provider")

    p = sub.add_parser("probe", help="probe providers")
    p.add_argument("--provider")
    p.add_argument(
        "-t",
        "--tier",
        choices=["S", "M", "L", "s", "m", "l"],
        help="probe only this tier (S, M, L)",
    )
    p.add_argument(
        "--all", action="store_true", help="include offerings with missing API keys"
    )
    p.add_argument("--json", action="store_true", help="output JSON")
    p.add_argument("--apply", action="store_true", help="merge results into registry")

    sub.add_parser("stats", help="stats of the running server")

    t = sub.add_parser("train-head", help="train the L1 classifier head")
    t.add_argument("--data", help="JSONL with labels")

    j = sub.add_parser("jobs", help="control plane background jobs")
    j.add_argument("--run", help="run a job immediately")

    b = sub.add_parser("batch", help="offline jobs (batch API)")
    bsub = b.add_subparsers(dest="batch_cmd")
    b_submit = bsub.add_parser("submit", help="submit a JSONL file")
    b_submit.add_argument("file")
    b_submit.add_argument("--window", type=float, help="window in hours")
    for name in ("status", "output", "cancel"):
        sp = bsub.add_parser(name)
        sp.add_argument("id")
    bsub.add_parser("list", help="list jobs")

    rep = sub.add_parser("report", help="telemetry report")
    rep.add_argument("--json", action="store_true")
    rep.add_argument("--csv", action="store_true")
    rep.add_argument("--days", type=float, default=7.0)
    return ap


def main() -> None:
    from .config import load_env

    load_env()  # keys from .env → os.environ
    args = build_parser().parse_args()

    if args.cmd == "keys":
        if args.keys_cmd == "set":
            cmd_keys_set(args)
        elif args.keys_cmd == "list":
            cmd_keys_list(args)
        elif args.keys_cmd == "status":
            cmd_keys_status(args)
        elif args.keys_cmd == "keychain":
            cmd_keys_keychain(args)
        else:
            cmd_keys_interactive(args)
    elif args.cmd == "models":
        cmd_models(args)
    elif args.cmd == "discover":
        cmd_discover(args)
    elif args.cmd == "probe":
        cmd_probe(args)
    elif args.cmd == "stats":
        cmd_stats(args)
    elif args.cmd == "jobs":
        cmd_jobs(args)
    elif args.cmd == "batch":
        cmd_batch(args)
    elif args.cmd == "train-head":
        cmd_train_head(args)
    elif args.cmd == "report":
        from .report import main as report_main

        sys.argv = (
            ["report"]
            + (["--json"] if args.json else [])
            + (["--csv"] if args.csv else [])
            + [f"--days={args.days}"]
        )
        report_main()
    else:
        # No args — as before, start the server.
        from .server import main as server_main

        server_main()


if __name__ == "__main__":
    main()
