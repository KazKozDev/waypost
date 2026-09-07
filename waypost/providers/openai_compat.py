"""Adapter for OpenAI-compatible endpoints.

Covers OpenRouter, Groq, Gemini (openai-compatible mode), together, local
mlx_lm.server and ollama. A custom protocol only makes sense to write for
a provider that requires it.

The adapter's main job is to normalize not the request but the ERRORS.
Each provider has its own 429 and quota-refusal format; without a single
classification, fallback turns into guesswork.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from dataclasses import dataclass
from enum import Enum
from typing import Any, AsyncIterator

import httpx

from ..prefix import build_payload
from ..registry import Offering
from ..schemas import ChatRequest


class Verdict(str, Enum):
    OK = "ok"
    RETRY = "retry"  # 5xx, timeout — the same model once more
    SWITCH = "switch"  # 429, quota — the next candidate at once
    FATAL = "fatal"  # 400, invalid request — the plan cannot save it


@dataclass
class ProviderError(Exception):
    verdict: Verdict
    status: int
    message: str
    retry_after_s: float = 60.0

    def __str__(self) -> str:
        return f"[{self.verdict}] {self.status}: {self.message}"


def parse_retry_after(value: str | None, default: float = 60.0) -> float:
    """Retry-After is either delta-seconds or an HTTP-date (RFC 9110).

    Only the first form was handled; a date fell through to the default,
    so the router waited a flat minute where the provider had said five
    seconds — or five minutes.
    """
    if not value:
        return default
    raw = value.strip()
    try:
        return max(0.0, float(raw))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return default
    if when is None:
        return default
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    delta = (when - datetime.now(timezone.utc)).total_seconds()
    # A date in the past means "retry now", not "wait a minute".
    return max(0.0, min(delta, 86_400.0))


def classify_error(
    status: int, body: str, headers: httpx.Headers | None = None
) -> ProviderError:
    retry_after = parse_retry_after(headers.get("retry-after") if headers else None)

    low = body.lower()
    if status == 429 or "rate limit" in low or "quota" in low:
        return ProviderError(Verdict.SWITCH, status, body[:300], retry_after)
    if status in (401, 403):
        return ProviderError(Verdict.SWITCH, status, "auth failed", 3600.0)
    if (
        status in (404, 410)
        or "end of life" in low
        or "no longer available" in low
        or "decommissioned" in low
        or "deprecated" in low
        or "gone" in low
    ):
        return ProviderError(Verdict.SWITCH, status, "model retired or not found", 86400.0)
    # "Model not found" or "tools not supported" also arrives as a 400
    # (xAI, Cerebras, Groq, some hosts). This is not a request defect
    # but an incompatibility with a specific provider/model — move to
    # the next model, do not stop the ladder.
    if any(
        p in low
        for p in (
            "model not found",
            "model does not exist",
            "unknown model",
            "invalid model",
            "no such model",
            "model not supported",
            "tool calling is not supported",
            "tool calling` is not supported",
            "tools are not supported",
            "tool_choice",
            "does not support tools",
            "tools is not supported",
            "function calling is not supported",
            "not support tool",
        )
    ):
        return ProviderError(Verdict.SWITCH, status, body[:300], 1.0)
    if status >= 500 or status == 408:
        return ProviderError(Verdict.RETRY, status, body[:300], 1.0)
    if status >= 400:
        return ProviderError(Verdict.FATAL, status, body[:300])
    return ProviderError(Verdict.OK, status, "")


class OpenAICompatAdapter:
    def __init__(self, client: httpx.AsyncClient):
        self.client = client

    def _headers(
        self,
        o: Offering,
        api_key: str | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        key = api_key if api_key is not None else o.api_key
        if key:
            h["Authorization"] = f"Bearer {key}"
        if idempotency_key:
            # Providers that understand it will not run the repeat twice;
            # the rest will ignore the unknown header.
            h["Idempotency-Key"] = idempotency_key
        return h

    def _url(self, o: Offering) -> str:
        return o.base_url.rstrip("/") + "/chat/completions"

    async def complete(
        self,
        o: Offering,
        req: ChatRequest,
        timeout: float,
        *,
        api_key: str | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        payload = build_payload(req, o, stream=False)
        try:
            r = await self.client.post(
                self._url(o),
                json=payload,
                headers=self._headers(o, api_key, idempotency_key),
                timeout=timeout,
            )
        except (httpx.TimeoutException, httpx.ConnectError) as exc:
            raise ProviderError(Verdict.RETRY, 408, str(exc), 1.0) from exc

        if r.status_code >= 400:
            raise classify_error(r.status_code, r.text, r.headers)
        return r.json()

    async def stream(
        self,
        o: Offering,
        req: ChatRequest,
        timeout: float,
        *,
        api_key: str | None = None,
        idempotency_key: str | None = None,
    ) -> AsyncIterator[bytes]:
        """SSE is proxied as-is.

        An important consequence: after the first chunk sent to the
        client, fallback is impossible. So errors that happen BEFORE the
        first byte are raised as ordinary ones — they can still be
        survived.
        """
        payload = build_payload(req, o, stream=True)
        stream_timeout = httpx.Timeout(connect=30.0, read=300.0, write=60.0, pool=30.0)
        try:
            payloads = [payload]
            if "stream_options" in payload:
                fallback = dict(payload)
                fallback.pop("stream_options", None)
                payloads.append(fallback)
            for index, candidate_payload in enumerate(payloads):
                async with self.client.stream(
                    "POST",
                    self._url(o),
                    json=candidate_payload,
                    headers=self._headers(o, api_key, idempotency_key),
                    timeout=stream_timeout,
                ) as r:
                    if r.status_code >= 400:
                        body = (await r.aread()).decode("utf-8", "replace")
                        # Some compatible APIs reject the optional OpenAI
                        # stream_options field. Retry once without it before
                        # exposing the provider error to the routing ladder.
                        if index == 0 and len(payloads) > 1 and r.status_code == 400:
                            continue
                        raise classify_error(r.status_code, body, r.headers)
                    async for chunk in r.aiter_bytes():
                        yield chunk
                    return
        except (httpx.TimeoutException, httpx.ConnectError) as exc:
            raise ProviderError(Verdict.RETRY, 408, str(exc), 1.0) from exc

    async def list_models(
        self, o: Offering, timeout: float = 30.0
    ) -> list[dict[str, Any]]:
        """Source B from the discovery spec."""
        url = o.base_url.rstrip("/") + "/models"
        r = await self.client.get(url, headers=self._headers(o), timeout=timeout)
        if r.status_code >= 400:
            raise classify_error(r.status_code, r.text, r.headers)
        return r.json().get("data", [])


def parse_usage(body: dict[str, Any]) -> tuple[int, int]:
    u = body.get("usage") or {}
    return int(u.get("prompt_tokens", 0)), int(u.get("completion_tokens", 0))


def sse_event(payload: dict[str, Any]) -> bytes:
    return b"data: " + json.dumps(payload, ensure_ascii=False).encode() + b"\n\n"
