"""Context compression.

Fewer tokens — more requests before the limit, and on a free tier that is
a direct saving. But compression has three constraints, each of which
easily turns a win into a loss, so they are baked into the code rather
than left to discretion:

  1. Only the DYNAMIC zone is compressed — retrieved context and old
     history. The system prompt and tool descriptions are never touched:
     a changed prefix = a provider-side cache miss, and that is usually
     more expensive than the tokens saved.
  2. Code, precise instructions and structured data are not compressed.
     There are measurements where LLMLingua-2 dropped accuracy from
     99.5% to 77% for 10% tokens — a pure loss.
  3. It only kicks in above a length threshold. On short context the
     encoder overhead eats the benefit entirely.

Modes:
  off       — default. Until you have your own measurements, this is the
              right mode.
  safe      — deterministic cleanup: duplicate paragraphs, extra spaces,
              repeated separators. Nothing is "rewritten".
  llmlingua — LLMLingua-2 (token classification over XLM-RoBERTa).
              Requires the llmlingua package; without it the mode falls
              back to safe.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

from .classify import estimate_tokens
from .schemas import ChatRequest

log = logging.getLogger("waypost.compress")

# Signs that the text must not be touched in any mode.
STRUCTURED_RE = re.compile(
    r"```|^\s*[{\[]|\bdef \b|\bclass \b|</\w+>|\|\s*-{3,}\s*\||^\s*\d+\.\s", re.M
)
LLMLINGUA_MODEL = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"


@dataclass
class CompressionStats:
    tokens_before: int = 0
    tokens_after: int = 0
    messages_touched: int = 0
    mode: str = "off"

    @property
    def saved(self) -> int:
        return max(0, self.tokens_before - self.tokens_after)

    @property
    def ratio(self) -> float:
        if not self.tokens_before:
            return 1.0
        return round(self.tokens_after / self.tokens_before, 3)

    def as_dict(self) -> dict:
        return {
            "mode": self.mode,
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
            "saved": self.saved,
            "ratio": self.ratio,
            "messages": self.messages_touched,
        }


def is_compressible(text: str) -> bool:
    """Code, JSON, tables and numbered instructions — do not touch."""
    return not STRUCTURED_RE.search(text)


def truncate_tool_output(text: str, max_lines: int = 60, max_chars: int = 3000) -> str:
    """Smart compaction of verbose agent tool/terminal outputs."""
    lines = text.splitlines()
    if len(lines) > max_lines:
        head_count = max(10, max_lines * 2 // 3)
        tail_count = max(5, max_lines - head_count)
        head = lines[:head_count]
        tail = lines[-tail_count:]
        omitted = len(lines) - head_count - tail_count
        return (
            "\n".join(head)
            + f"\n\n[... waypost: compressed {omitted} lines of tool output ...]\n\n"
            + "\n".join(tail)
        )
    if len(text) > max_chars:
        half = max_chars // 2
        return text[:half] + "\n[... waypost: truncated ...]\n" + text[-half:]
    return text


def safe_clean(text: str) -> str:
    """Deterministic cleanup. The meaning does not change — only
    duplicates and junk whitespace are removed."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    paragraphs = [p.strip() for p in text.split("\n\n")]
    seen: set[str] = set()
    kept: list[str] = []
    for p in paragraphs:
        if not p:
            continue
        # Duplicate paragraphs — a typical RAG problem: the same chunk
        # arrives from two sources.
        marker = p.lower()
        if marker in seen:
            continue
        seen.add(marker)
        kept.append(p)
    return "\n\n".join(kept).strip()


@lru_cache(maxsize=1)
def _llmlingua(model_name: str = LLMLINGUA_MODEL):
    try:
        from llmlingua import PromptCompressor  # noqa: PLC0415

        return PromptCompressor(model_name=model_name, use_llmlingua2=True)
    except Exception:  # noqa: BLE001
        return None


class Compressor:
    def __init__(
        self,
        *,
        mode: Literal["off", "safe", "llmlingua", "smart", "auto"] = "off",
        min_tokens: int = 3000,
        rate: float = 0.6,
        model_name: str = LLMLINGUA_MODEL,
    ):
        self.mode = mode
        self.min_tokens = min_tokens
        self.rate = rate
        self.model_name = model_name

    @property
    def backend(self) -> str:
        if self.mode != "llmlingua":
            return self.mode
        return "llmlingua" if _llmlingua(self.model_name) else "llmlingua:missing"

    def compress(self, req: ChatRequest) -> CompressionStats:
        """Mutates the request in place. Returns stats — without them the
        main question cannot be answered: did it pay off."""
        effective_mode = self.mode
        stats = CompressionStats(mode=effective_mode)
        if effective_mode == "off":
            return stats

        # Dynamic zone: everything except system blocks. The last user
        # message is also left untouched — that is the question itself.
        dynamic = [m for m in req.messages if m.role not in ("system", "developer")]
        if dynamic:
            dynamic = dynamic[:-1]

        total = sum(
            estimate_tokens(m.content) for m in dynamic if isinstance(m.content, str)
        )
        stats.tokens_before = total
        stats.tokens_after = total
        if total < self.min_tokens and effective_mode != "smart":
            # Below the threshold, compressing costs more than not.
            return stats

        if effective_mode == "auto":
            effective_mode = "smart"

        compressor = (
            _llmlingua(self.model_name) if effective_mode == "llmlingua" else None
        )
        after = 0
        for m in dynamic:
            if not isinstance(m.content, str):
                continue

            # For agent tool outputs, apply smart observation compaction
            if m.role in ("tool", "function"):
                compacted = truncate_tool_output(m.content)
                if compacted != m.content:
                    m.content = compacted
                    stats.messages_touched += 1

            if not is_compressible(m.content):
                after += estimate_tokens(m.content)
                continue
            original = m.content
            if compressor is not None:
                try:
                    result = compressor.compress_prompt(
                        original, rate=self.rate, force_tokens=["\n", "?", "."]
                    )
                    m.content = result.get("compressed_prompt") or original
                except Exception as exc:  # noqa: BLE001
                    log.warning("llmlingua failed, staying on safe: %s", exc)
                    m.content = safe_clean(original)
            else:
                m.content = safe_clean(original)
            if m.content != original:
                stats.messages_touched += 1
            after += estimate_tokens(m.content)
        stats.tokens_after = after
        return stats
