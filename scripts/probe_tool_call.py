#!/usr/bin/env python3
"""Send one tool-calling request at a running Waypost and print the raw reply.

A hand probe for the local-tool path: what the router actually puts on the
wire is easier to judge from the unparsed body than from a client that has
already normalized it. --stream switches to SSE and prints frames as they
arrive, which is where chunking bugs show up.

Uses httpx (already a dependency) rather than requests, so it runs in the
project venv with nothing extra installed.
"""
from __future__ import annotations

import argparse
import json

import httpx

WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--prompt", default="погугли курс евро")
    ap.add_argument("--model", default="auto")
    ap.add_argument("--stream", action="store_true", help="use SSE and print frames")
    ap.add_argument("--timeout", type=float, default=180.0)
    args = ap.parse_args()

    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "tools": [WEB_SEARCH_TOOL],
        "stream": args.stream,
    }
    endpoint = f"{args.url.rstrip('/')}/v1/chat/completions"

    try:
        if args.stream:
            with httpx.stream(
                "POST", endpoint, json=payload, timeout=args.timeout
            ) as r:
                print(r.status_code)
                for line in r.iter_lines():
                    if line:
                        print(line)
        else:
            r = httpx.post(endpoint, json=payload, timeout=args.timeout)
            print(r.status_code)
            try:
                print(json.dumps(r.json(), ensure_ascii=False, indent=2))
            except ValueError:
                print(r.text)
    except httpx.HTTPError as exc:
        print(f"request failed: {exc}")
        print(f"is waypost running at {args.url}?")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
