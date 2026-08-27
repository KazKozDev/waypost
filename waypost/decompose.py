"""Task and Multimodal Decomposition for Waypost v5 (Spec Section I.3).

Decomposes separable multimodal queries (e.g. OCR / description + downstream text reasoning):
1. Detects whether a multimodal request is separable or deictic/coupled.
2. Formulates sub-tasks:
   - Extraction stage: vision model transforms visual/audio into structured text.
   - Reasoning stage: text-only router evaluates the full free pool of 56 models.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from .schemas import ChatRequest

# Deictic patterns indicate the question cannot be separated from direct visual references
DEICTIC_PATTERNS = (
    r"\bhere\b",
    r"\bthis part\b",
    r"\bcircled in\b",
    r"\bpointed at\b",
    r"\barrows?\b",
    r"\bhighlighted in\b",
    r"\bleft side of\b",
    r"\bright side of\b",
    r"\bshown here\b",
    r"\bна этом месте\b",
    r"\bобведено\b",
    r"\bстрелочк\w+\b",
    r"\bвыделенн\w+\b",
)


@dataclass
class DecomposedTask:
    is_separable: bool
    stage: Literal["single_shot", "extraction_first", "transcription_first"]
    extraction_prompt: str | None = None
    reasoning_prompt: str | None = None


def is_deictic_prompt(text: str) -> bool:
    """Check if the user prompt relies on deictic references to the image."""
    low = text.lower()
    return any(re.search(pat, low) for pat in DEICTIC_PATTERNS)


def analyze_multimodal_decomposition(req: ChatRequest) -> DecomposedTask:
    """Analyzes a request to decide if multimodal decomposition should be applied."""
    has_image = False
    has_audio = False
    user_text = ""

    for m in req.messages:
        if m.role == "user":
            if isinstance(m.content, str):
                user_text += " " + m.content
            elif isinstance(m.content, list):
                for b in m.content:
                    if isinstance(b, dict):
                        b_type = b.get("type", "")
                        if b_type in ("image_url", "image"):
                            has_image = True
                        elif b_type in ("input_audio", "audio"):
                            has_audio = True
                        elif b_type == "text":
                            user_text += " " + b.get("text", "")

    user_text = user_text.strip()

    if not has_image and not has_audio:
        return DecomposedTask(is_separable=False, stage="single_shot")

    # If deictic, keep end-to-end vision routing
    if has_image and is_deictic_prompt(user_text):
        return DecomposedTask(is_separable=False, stage="single_shot")

    if has_image:
        return DecomposedTask(
            is_separable=True,
            stage="extraction_first",
            extraction_prompt="Describe in detail all visible text, diagrams, code, and key elements from the image.",
            reasoning_prompt=user_text or "Analyze the extracted image content.",
        )

    if has_audio:
        return DecomposedTask(
            is_separable=True,
            stage="transcription_first",
            extraction_prompt="Transcribe the audio accurately.",
            reasoning_prompt=user_text or "Respond to the transcribed audio.",
        )

    return DecomposedTask(is_separable=False, stage="single_shot")


def get_vision_token_budget(task_class: str) -> int:
    """Returns the explicit vision token budget based on task type (Spec I.2).

    - classification / short caption: 70–140
    - scene description / QA: 280
    - diagrams / large text: 560
    - dense OCR / tables: 1120
    - video frames: 70
    """
    task = task_class.lower()
    if any(k in task for k in ("ocr", "table", "document", "dense")):
        return 1120
    if any(k in task for k in ("diagram", "code", "chart", "math", "schema")):
        return 560
    if any(k in task for k in ("classification", "video", "frame")):
        return 70
    return 280
