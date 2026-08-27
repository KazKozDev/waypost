#!/usr/bin/env python3
"""Interactive chat client for manual testing of waypost.

Connects to the server over the OpenAI-compatible API and shows what the
router decided (provider, model, cache, latency).

    python -m scripts.chat
    python -m scripts.chat --url http://127.0.0.1:8080/v1 --stream

In-chat commands:
    /quit, /exit        exit
    /clear              clear history
    /model NAME         set the model (auto — the router)
    /privacy strict|default
    /temp N             temperature (0..2)
    /stream on|off      streaming
    /help               list commands
"""
from __future__ import annotations

import argparse
import json

import httpx


class Chat:
    def __init__(self, url: str, model: str, stream: bool):
        self.url = url.rstrip("/")
        self.model = model
        self.stream = stream
        self.privacy = "default"
        self.temperature = 0.0
        self.messages: list[dict] = []

    # ------------------------------------------------------------ api
    def _payload(self) -> dict:
        return {
            "model": self.model,
            "messages": self.messages,
            "temperature": self.temperature,
            "privacy": self.privacy,
            "stream": self.stream,
        }

    def _send(self) -> dict:
        r = httpx.post(
            f"{self.url}/chat/completions", json=self._payload(), timeout=180.0
        )
        if r.status_code != 200:
            body = r.json()
            err = body.get("error", {}).get("message", r.text)
            raise RuntimeError(f"HTTP {r.status_code}: {err}")
        return r.json()

    def _stream(self) -> str:
        content = ""
        with httpx.stream(
            "POST", f"{self.url}/chat/completions", json=self._payload(), timeout=180.0
        ) as r:
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}: {r.read().decode()}")
            for line in r.iter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                delta = (
                    (chunk.get("choices") or [{}])[0].get("delta", {}).get("content")
                )
                if delta:
                    content += delta
                    print(delta, end="", flush=True)
        print()
        return content

    # ------------------------------------------------------------ loop
    def run(self) -> None:
        print(
            f"waypost chat → {self.url}  (model: {self.model}, "
            f"stream: {'on' if self.stream else 'off'})"
        )
        print("Type a message or /help. /quit to exit.\n")
        while True:
            try:
                line = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line:
                continue
            if line.startswith("/"):
                if self._command(line):
                    break
                continue

            self.messages.append({"role": "user", "content": line})
            try:
                if self.stream:
                    content = self._stream()
                    meta = None
                else:
                    body = self._send()
                    content = body["choices"][0]["message"]["content"]
                    meta = body.get("router", {})
                    print(content)
                self.messages.append({"role": "assistant", "content": content})
                if meta:
                    self._show_meta(meta)
            except RuntimeError as exc:
                print(f"  ! {exc}")
                self.messages.pop()  # do not accumulate a failed request

    def _show_meta(self, meta: dict) -> None:
        parts = [
            f"provider={meta.get('provider')}",
            f"model={meta.get('model')}",
            f"task={meta.get('task_class')}",
            f"tier={meta.get('complexity_tier')}",
            f"src={meta.get('classifier_source')}",
            f"cache={meta.get('cache')}",
            f"attempts={meta.get('attempts')}",
            f"{meta.get('latency_ms')}ms",
        ]
        print(f"  [router] {' | '.join(parts)}")

    def _command(self, line: str) -> bool:
        cmd, _, arg = line.partition(" ")
        arg = arg.strip()
        if cmd in ("/quit", "/exit"):
            return True
        if cmd == "/clear":
            self.messages.clear()
            print("  history cleared")
        elif cmd == "/model":
            self.model = arg or "auto"
            print(f"  model: {self.model}")
        elif cmd == "/privacy":
            self.privacy = arg or "default"
            print(f"  privacy: {self.privacy}")
        elif cmd == "/temp":
            try:
                self.temperature = float(arg)
                print(f"  temperature: {self.temperature}")
            except ValueError:
                print("  ! number 0..2 expected")
        elif cmd == "/stream":
            self.stream = arg != "off"
            print(f"  stream: {'on' if self.stream else 'off'}")
        elif cmd == "/help":
            print(
                "  /quit /clear /model NAME /privacy strict|default "
                "/temp N /stream on|off"
            )
        else:
            print(f"  ! unknown command: {cmd}")
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description="interactive waypost chat")
    ap.add_argument("--url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--model", default="auto")
    ap.add_argument("--stream", action="store_true")
    args = ap.parse_args()
    Chat(args.url, args.model, args.stream).run()


if __name__ == "__main__":
    main()
