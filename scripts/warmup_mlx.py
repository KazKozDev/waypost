#!/usr/bin/env python3
"""Warmup and health probe for MLX LM Server.

Sends a minimal prompt to prime the Metal GPU context and confirm
the OpenAI-compatible endpoint is responding.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
import urllib.error


def warmup_server(
    host: str = "127.0.0.1",
    port: int = 8081,
    model: str = "mlx-community/Qwen3.6-27B-4bit",
    timeout_s: float = 30.0,
) -> bool:
    url = f"http://{host}:{port}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 4,
        "temperature": 0.0,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            elapsed = (time.perf_counter() - t0) * 1000
            print(
                f"[warmup_mlx] Server at {url} responded in {elapsed:.1f}ms: "
                f"{body.get('choices', [{}])[0].get('message', {}).get('content', '').strip()}"
            )
            return True
    except (urllib.error.URLError, TimeoutError, ConnectionRefusedError) as exc:
        print(f"[warmup_mlx] Failed to reach server at {url}: {exc}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Warmup MLX LM server")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=8081, help="Server port")
    parser.add_argument(
        "--timeout", type=float, default=30.0, help="Timeout in seconds"
    )
    args = parser.parse_args()

    ok = warmup_server(host=args.host, port=args.port, timeout_s=args.timeout)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
