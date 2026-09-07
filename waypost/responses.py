"""Compatibility bridge between OpenAI Responses and Chat Completions.

Waypost's routing core speaks Chat Completions internally.  This module keeps
the protocol translation at ingress/egress so Responses clients use the same
policy, cache, router and provider fallback path as every other client.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from .schemas import ChatRequest


def _text_content(content: Any) -> str | list[dict[str, Any]] | None:
    if content is None or isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)

    converted: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            converted.append({"type": "text", "text": str(part)})
            continue
        kind = part.get("type")
        if kind in ("input_text", "output_text", "text"):
            converted.append({"type": "text", "text": str(part.get("text", ""))})
        elif kind == "input_image":
            image = part.get("image_url") or part.get("file_id")
            if image:
                converted.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image,
                            "detail": part.get("detail", "auto"),
                        },
                    }
                )
        # input_file and input_audio have no portable Chat Completions shape.
        # Rejecting the whole request would prevent mixed text clients from
        # working, so unsupported blocks are omitted at this compatibility edge.
    return converted


def _response_tools(tools: Any) -> list[dict[str, Any]] | None:
    if not tools:
        return None
    converted = []
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            raise ValueError("Waypost Responses currently supports function tools only")
        if isinstance(tool.get("function"), dict):
            converted.append(tool)
            continue
        function = {
            key: tool[key]
            for key in ("name", "description", "parameters", "strict")
            if key in tool
        }
        if not function.get("name"):
            raise ValueError("Responses function tool is missing name")
        converted.append({"type": "function", "function": function})
    return converted


def _tool_choice(choice: Any) -> Any:
    if not isinstance(choice, dict):
        return choice
    if choice.get("type") == "function" and choice.get("name"):
        return {"type": "function", "function": {"name": choice["name"]}}
    return choice


def _response_format(text: Any) -> dict[str, Any] | None:
    if not isinstance(text, dict) or not isinstance(text.get("format"), dict):
        return None
    fmt = dict(text["format"])
    kind = fmt.get("type")
    if kind == "text":
        return None
    if kind == "json_schema":
        return {
            "type": "json_schema",
            "json_schema": {
                key: fmt[key]
                for key in ("name", "description", "schema", "strict")
                if key in fmt
            },
        }
    if kind == "json_object":
        return {"type": "json_object"}
    raise ValueError(f"unsupported Responses text format: {kind}")


def responses_to_chat(payload: dict[str, Any]) -> ChatRequest:
    """Convert a Responses create request to Waypost's internal request."""
    messages: list[dict[str, Any]] = []
    instructions = payload.get("instructions")
    if instructions:
        messages.append({"role": "developer", "content": instructions})

    raw_input = payload.get("input")
    if isinstance(raw_input, str):
        messages.append({"role": "user", "content": raw_input})
    elif isinstance(raw_input, list):
        for item in raw_input:
            if isinstance(item, str):
                messages.append({"role": "user", "content": item})
                continue
            if not isinstance(item, dict):
                continue
            kind = item.get("type", "message")
            if kind in ("message", "easy_input_message"):
                messages.append(
                    {
                        "role": item.get("role", "user"),
                        "content": _text_content(item.get("content")),
                    }
                )
            elif kind == "function_call":
                call_id = item.get("call_id") or item.get("id") or f"call_{uuid.uuid4().hex}"
                messages.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": item.get("name", ""),
                                    "arguments": item.get("arguments", "{}"),
                                },
                            }
                        ],
                    }
                )
            elif kind == "function_call_output":
                output = item.get("output", "")
                if not isinstance(output, str):
                    output = json.dumps(output, ensure_ascii=False)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": item.get("call_id", ""),
                        "content": output,
                    }
                )
            # Reasoning and references are provider-specific context items. They
            # cannot be represented faithfully in Chat Completions and are ignored.
    elif raw_input is not None:
        raise ValueError("Responses input must be a string or an array")

    if not messages:
        raise ValueError("Responses request must include input or instructions")

    return ChatRequest(
        model=payload.get("model", "auto"),
        messages=messages,
        temperature=payload.get("temperature", 1.0),
        top_p=payload.get("top_p"),
        max_tokens=payload.get("max_output_tokens"),
        stream=bool(payload.get("stream", False)),
        tools=_response_tools(payload.get("tools")),
        tool_choice=_tool_choice(payload.get("tool_choice")),
        response_format=_response_format(payload.get("text")),
        privacy=payload.get("privacy", "default"),
        latency_class=payload.get("latency_class", "interactive"),
        profile=payload.get("profile", "auto"),
        session_id=payload.get("session_id"),
        no_cache=bool(payload.get("no_cache", False)),
        thinking_mode=bool(payload.get("thinking_mode", False)),
        idempotency_key=payload.get("idempotency_key"),
    )


def _output_items(message: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    output: list[dict[str, Any]] = []
    text = message.get("content") or ""
    if text:
        output.append(
            {
                "id": f"msg_{uuid.uuid4().hex}",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": text,
                        "annotations": [],
                        "logprobs": [],
                    }
                ],
            }
        )
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        output.append(
            {
                "id": f"fc_{uuid.uuid4().hex}",
                "type": "function_call",
                "status": "completed",
                "call_id": call.get("id") or f"call_{uuid.uuid4().hex}",
                "name": function.get("name", ""),
                "arguments": function.get("arguments", "{}"),
            }
        )
    return output, text


def chat_to_response(
    body: dict[str, Any], request_payload: dict[str, Any], *, response_id: str | None = None
) -> dict[str, Any]:
    """Convert a completed Chat Completions body to a Responses object."""
    response_id = response_id or f"resp_{uuid.uuid4().hex}"
    choice = (body.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    output, output_text = _output_items(message)
    finish_reason = choice.get("finish_reason")
    incomplete = finish_reason in ("length", "max_tokens")
    usage = body.get("usage") or {}
    now = int(time.time())
    result = {
        "id": response_id,
        "object": "response",
        "created_at": body.get("created", now),
        "completed_at": now,
        "status": "incomplete" if incomplete else "completed",
        "error": None,
        "incomplete_details": {"reason": "max_output_tokens"} if incomplete else None,
        "instructions": request_payload.get("instructions"),
        "max_output_tokens": request_payload.get("max_output_tokens"),
        "model": body.get("model", request_payload.get("model", "auto")),
        "output": output,
        "output_text": output_text,
        "parallel_tool_calls": request_payload.get("parallel_tool_calls", True),
        "previous_response_id": request_payload.get("previous_response_id"),
        "reasoning": request_payload.get("reasoning"),
        "store": request_payload.get("store", False),
        "temperature": request_payload.get("temperature", 1.0),
        "text": request_payload.get("text", {"format": {"type": "text"}}),
        "tool_choice": request_payload.get("tool_choice", "auto"),
        "tools": request_payload.get("tools", []),
        "top_p": request_payload.get("top_p", 1.0),
        "truncation": request_payload.get("truncation", "disabled"),
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": usage.get("completion_tokens", 0),
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": usage.get("total_tokens", 0),
        },
    }
    if "router" in body:
        result["router"] = body["router"]
    return result


def responses_sse(event: dict[str, Any]) -> bytes:
    return b"event: " + event["type"].encode() + b"\n" + b"data: " + json.dumps(
        event, ensure_ascii=False, separators=(",", ":")
    ).encode() + b"\n\n"


@dataclass
class ResponsesStreamTranslator:
    """Incrementally translate Chat Completions SSE into Responses SSE."""

    request_payload: dict[str, Any]
    response_id: str = field(default_factory=lambda: f"resp_{uuid.uuid4().hex}")
    message_id: str = field(default_factory=lambda: f"msg_{uuid.uuid4().hex}")
    created_at: int = field(default_factory=lambda: int(time.time()))
    buffer: bytes = b""
    sequence: int = 0
    text: str = ""
    model: str = "auto"
    usage: dict[str, Any] = field(default_factory=dict)
    finish_reason: str | None = None
    message_started: bool = False
    tool_calls: dict[int, dict[str, Any]] = field(default_factory=dict)
    finished: bool = False

    def _event(self, kind: str, **fields: Any) -> bytes:
        self.sequence += 1
        return responses_sse({"type": kind, **fields, "sequence_number": self.sequence})

    def _base_response(self, status: str, output: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        body = {
            "id": self.response_id,
            "object": "response",
            "created_at": self.created_at,
            "status": status,
            "error": None,
            "incomplete_details": None,
            "instructions": self.request_payload.get("instructions"),
            "max_output_tokens": self.request_payload.get("max_output_tokens"),
            "model": self.model,
            "output": output or [],
            "parallel_tool_calls": self.request_payload.get("parallel_tool_calls", True),
            "previous_response_id": self.request_payload.get("previous_response_id"),
            "reasoning": self.request_payload.get("reasoning"),
            "store": self.request_payload.get("store", False),
            "temperature": self.request_payload.get("temperature", 1.0),
            "text": self.request_payload.get("text", {"format": {"type": "text"}}),
            "tool_choice": self.request_payload.get("tool_choice", "auto"),
            "tools": self.request_payload.get("tools", []),
            "top_p": self.request_payload.get("top_p", 1.0),
            "truncation": self.request_payload.get("truncation", "disabled"),
            "usage": None,
        }
        return body

    def start(self) -> list[bytes]:
        response = self._base_response("in_progress")
        return [
            self._event("response.created", response=response),
            self._event("response.in_progress", response=response),
        ]

    def _start_message(self) -> list[bytes]:
        if self.message_started:
            return []
        self.message_started = True
        item = {
            "id": self.message_id,
            "type": "message",
            "status": "in_progress",
            "role": "assistant",
            "content": [],
        }
        part = {"type": "output_text", "text": "", "annotations": [], "logprobs": []}
        return [
            self._event("response.output_item.added", output_index=0, item=item),
            self._event(
                "response.content_part.added",
                item_id=self.message_id,
                output_index=0,
                content_index=0,
                part=part,
            ),
        ]

    def _chat_event(self, data: dict[str, Any]) -> list[bytes]:
        events: list[bytes] = []
        self.model = data.get("model") or self.model
        if data.get("usage"):
            self.usage = data["usage"]
        for choice in data.get("choices") or []:
            delta = choice.get("delta") or {}
            if delta.get("content") is not None:
                text = str(delta["content"])
                events.extend(self._start_message())
                self.text += text
                events.append(
                    self._event(
                        "response.output_text.delta",
                        item_id=self.message_id,
                        output_index=0,
                        content_index=0,
                        delta=text,
                        logprobs=[],
                    )
                )
            for call in delta.get("tool_calls") or []:
                index = int(call.get("index", 0))
                state = self.tool_calls.setdefault(
                    index,
                    {
                        "id": f"fc_{uuid.uuid4().hex}",
                        "call_id": call.get("id") or f"call_{uuid.uuid4().hex}",
                        "name": "",
                        "arguments": "",
                        "started": False,
                    },
                )
                if call.get("id"):
                    state["call_id"] = call["id"]
                function = call.get("function") or {}
                state["name"] += function.get("name") or ""
                arguments = function.get("arguments") or ""
                if not state["started"]:
                    state["started"] = True
                    item = {
                        "id": state["id"],
                        "type": "function_call",
                        "status": "in_progress",
                        "call_id": state["call_id"],
                        "name": state["name"],
                        "arguments": "",
                    }
                    events.append(
                        self._event(
                            "response.output_item.added",
                            output_index=index + (1 if self.message_started else 0),
                            item=item,
                        )
                    )
                state["arguments"] += arguments
                if arguments:
                    events.append(
                        self._event(
                            "response.function_call_arguments.delta",
                            item_id=state["id"],
                            output_index=index + (1 if self.message_started else 0),
                            delta=arguments,
                        )
                    )
            if choice.get("finish_reason"):
                self.finish_reason = choice["finish_reason"]
        return events

    def feed(self, chunk: bytes) -> list[bytes]:
        self.buffer += chunk.replace(b"\r\n", b"\n")
        events: list[bytes] = []
        while b"\n\n" in self.buffer:
            raw, self.buffer = self.buffer.split(b"\n\n", 1)
            data_lines = [line[5:].strip() for line in raw.splitlines() if line.startswith(b"data:")]
            if not data_lines:
                continue
            payload = b"\n".join(data_lines)
            if payload == b"[DONE]":
                events.extend(self.finish())
                continue
            try:
                events.extend(self._chat_event(json.loads(payload)))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
        return events

    def finish(self) -> list[bytes]:
        if self.finished:
            return []
        self.finished = True
        events: list[bytes] = []
        output: list[dict[str, Any]] = []
        if self.message_started:
            part = {"type": "output_text", "text": self.text, "annotations": [], "logprobs": []}
            item = {
                "id": self.message_id,
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [part],
            }
            events.extend(
                [
                    self._event(
                        "response.output_text.done",
                        item_id=self.message_id,
                        output_index=0,
                        content_index=0,
                        text=self.text,
                        logprobs=[],
                    ),
                    self._event(
                        "response.content_part.done",
                        item_id=self.message_id,
                        output_index=0,
                        content_index=0,
                        part=part,
                    ),
                    self._event("response.output_item.done", output_index=0, item=item),
                ]
            )
            output.append(item)
        for index, state in sorted(self.tool_calls.items()):
            output_index = index + (1 if self.message_started else 0)
            item = {
                "id": state["id"],
                "type": "function_call",
                "status": "completed",
                "call_id": state["call_id"],
                "name": state["name"],
                "arguments": state["arguments"],
            }
            events.append(
                self._event(
                    "response.function_call_arguments.done",
                    item_id=state["id"],
                    output_index=output_index,
                    arguments=state["arguments"],
                )
            )
            events.append(self._event("response.output_item.done", output_index=output_index, item=item))
            output.append(item)
        incomplete = self.finish_reason in ("length", "max_tokens")
        response = self._base_response("incomplete" if incomplete else "completed", output)
        response["completed_at"] = int(time.time())
        response["incomplete_details"] = {"reason": "max_output_tokens"} if incomplete else None
        response["output_text"] = self.text
        response["usage"] = {
            "input_tokens": self.usage.get("prompt_tokens", 0),
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": self.usage.get("completion_tokens", 0),
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": self.usage.get("total_tokens", 0),
        }
        kind = "response.incomplete" if incomplete else "response.completed"
        events.append(self._event(kind, response=response))
        return events
