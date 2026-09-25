"""Reviewer-facing wording: no internal codes in text people read.
"""

from __future__ import annotations

import re

TERMS: dict[str, str] = {
    "suspicious_eye_movement": "the candidate looked away from the screen",
    "external_help": "another person was interacting with the candidate",
    "phone_usage": "a phone was visible",
    "no_candidate": "the candidate was not in frame",
    "external_resource_open": "a non-exam app or site was open on screen",
    "secondary_workspace_visible": "a second screen was in use",
    "external_paste": "text was pasted in from outside the exam",
    "mass_paste": "several large blocks of text were pasted in",
    "discussing_solution": "discussing the solution",
    "receiving_dictation": "receiving dictation",
    "reciting_answer_choices": "reciting the answer choices",
    "asking_for_answer": "asking for an answer",
    "technical_exam_help": "asking for technical help",
    "MCQ_ANSWER_SELECTED": "an answer was selected",
    "MCQ_ANSWER_CHANGED": "an answer was changed",
    "TYPING_STARTED": "the candidate started typing",
    "TYPING_STOPPED": "the candidate stopped typing",
    "LARGE_PASTE": "a large block of text was pasted",
    "PASTE": "text was pasted",
    "COPY": "text was copied",
    "LARGE_TEXT_INSERTION": "a large block of text was inserted",
    "TEXT_CORRECTION": "the candidate corrected their text",
    "CODE_SUBMISSION": "code was submitted",
    "ANSWER_SUBMITTED": "an answer was submitted",
    "WINDOW_BLUR": "the candidate left the exam window",
    "WINDOW_FOCUS": "the candidate returned to the exam window",
    "TAB_SWITCH": "the candidate switched tabs",
    "FULLSCREEN_EXIT": "the candidate left fullscreen",
    "SCREEN_FULLSCREEN_LOST": "the exam left fullscreen",
    "SCREEN_PASTE": "a paste was seen on screen",
    "SCREEN_EXTERNAL_PASTE": "text was pasted from outside the exam",
    "SCREEN_EXTERNAL_RESOURCE": "a non-exam site or app was open",
    "SCREEN_AI_ASSISTANT_UI": "an AI assistant was open on screen",
    "SCREEN_SECONDARY_WORKSPACE": "a second screen was in use",
    "SECOND_MONITOR_DETECTED": "a second monitor was detected",
    "RIGHT_CLICK": "the candidate right-clicked",
    "ACTIVITY_GAP": "there was a gap in activity",
}

_BY_LENGTH = sorted(TERMS, key=len, reverse=True)
_UPPER_CODE = re.compile(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b")
_SNAKE_CODE = re.compile(r"\b[a-z]+(?:_[a-z]+)+\b")
_MS = re.compile(r"\b(\d+)\s?ms\b")


def seconds(ms: int | str) -> str:
    s = round(int(ms) / 1000)
    return f"{s} second{'' if s == 1 else 's'}"


def term(token: str) -> str:
    return TERMS.get(token, token.replace("_", " ").lower())


def plain_text(text: str | None) -> str | None:
    """Replace internal codes and millisecond counts with plain words."""

    if not text:
        return text
    out = text
    for token in _BY_LENGTH:
        out = re.sub(rf"\b{re.escape(token)}\b", TERMS[token], out)
    out = _UPPER_CODE.sub(lambda m: m[0].replace("_", " ").lower(), out)
    out = _SNAKE_CODE.sub(lambda m: m[0].replace("_", " "), out)
    return _MS.sub(lambda m: seconds(m[1]), out)
