"""Screen-perception contracts (one observation per screen chunk)."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from .base import ContractModel

# Keyed into the screen chunk cache, so this must move whenever the screen
# prompt does — otherwise a prompt change is silently replayed against
# observations the old prompt produced. Camera avoids the trap by keying off
# PERCEPTION_PROMPT_VERSION directly; screen carries its own tag, so the two
# have to be bumped together.
SCREEN_PERCEPTION_VERSION = "screen-perc-v5"

ScreenTernary = Literal["yes", "no", "UNKNOWN"]
ForegroundAppClass = Literal[
    "exam_ide",
    "browser",
    "notes",
    "ai_chat",
    "messaging",
    "os_desktop",
    "other",
    "UNKNOWN",
]


class ScreenObservation(ContractModel):
    chunk_id: str
    sequence: int
    section_id: str | None = None
    start_ms: int
    end_ms: int
    exam_ui_visible: ScreenTernary = "UNKNOWN"
    foreground_app_class: ForegroundAppClass = "UNKNOWN"
    external_resource_labels: list[str] = Field(default_factory=list)
    ai_assistant_ui_visible: ScreenTernary = "UNKNOWN"
    secondary_workspace_visible: ScreenTernary = "UNKNOWN"
    fullscreen_exam_likely: ScreenTernary = "UNKNOWN"
    paste_cue_visible: ScreenTernary = "UNKNOWN"
    pasted_text_excerpt: str | None = None
    visible_question_ref: str | None = None
    confidence: float = 0.0
    quality_caveat: str | None = None
    video_available: bool = False


class ScreenPerceptionBundle(ContractModel):
    candidate_id: str
    assessment_id: str
    produced_at: str
    screen_perception_version: str
    observations: list[ScreenObservation]
    total_chunks: int
    covered_chunks: int
    session_start_ms: int
    duration_ms: int
