from __future__ import annotations

from dataclasses import dataclass, field

from ..machine_facts.kinds import MachineFactKind
from .contracts import (
    CorrelationConfig,
    TimelineEvent,
)
from .sqb import is_verified_question_boundary

PASTE_CLUSTER_GAP_MS = 5_000

BLUR_KINDS = {MachineFactKind.WINDOW_BLUR.value, MachineFactKind.TAB_SWITCH.value}
INPUT_AFTER_GAZE_KINDS = {
    MachineFactKind.LARGE_PASTE.value,
    MachineFactKind.TEXT_CORRECTION.value,
    MachineFactKind.TYPING_STARTED.value,
    MachineFactKind.TYPING_STOPPED.value,
}
GAZE_EVENT_TYPES = {"suspicious_eye_movement"}
SPEECH_INTEGRITY_EVENT_TYPES = {
    "whisper_or_dictation",
    "off_camera_voice_coaching",
    "device_call_during_attempt",
    "av_speech_without_lips",
    "discussing_solution_audio",
}

SECOND_PERSON_EVENT_TYPES = {"external_help", "multiple_faces"}
ANSWER_SPEECH_EVENT_TYPES = {"discussing_solution_audio"}
ANSWER_SPEECH_CLASSES = {
    "asking_for_answer",
    "receiving_dictation",
    "discussing_solution",
    "reciting_answer_choices",
}


@dataclass
class PasteCluster:
    representative: TimelineEvent
    members: list[TimelineEvent] = field(default_factory=list)
    t_ms: int = 0
    end_ms: int = 0
    section_id: str | None = None
    question_number: int | None = None
    question_id: str | None = None
    total_chars_added: int = 0
    evidence_refs: list[str] = field(default_factory=list)


@dataclass
class FastCorrectQuestion:
    question_number: int
    question_id: str | None
    section_id: str | None
    window_ms: tuple[int, int]
    correct: bool
    fast: bool
    speed_basis: str
    time_spent_seconds: float | None = None
    score: float | None = None
    max_score: float | None = None
    evidence_refs: list[str] = field(default_factory=list)


@dataclass
class FastMcqBurst:
    section_id: str | None
    section_title: str | None
    start_ms: int
    end_ms: int
    answer_count: int
    median_gap_ms: int
    evidence_refs: list[str] = field(default_factory=list)


def same_section(a: str | None, b: str | None) -> bool:
    if not a or not b:
        return True
    return a == b


def intervals_overlap(a0: int, a1: int, b0: int, b1: int, pad_ms: int) -> bool:
    return a0 - pad_ms <= b1 and b0 <= a1 + pad_ms


def paste_size(event: TimelineEvent) -> int:
    chars = event.detail.get("charsAdded")
    return int(chars) if isinstance(chars, (int, float)) else 0


def perception_episodes_by_kind(timeline: list[TimelineEvent], kind: str) -> list[TimelineEvent]:
    return [e for e in timeline if e.source == "perception_episode" and e.kind == kind]


def perception_episodes_by_kinds(
    timeline: list[TimelineEvent], kinds: set[str]
) -> list[TimelineEvent]:
    return [e for e in timeline if e.source == "perception_episode" and e.kind in kinds]


def is_answer_speech(event: TimelineEvent) -> bool:
    return event.detail.get("speechContentClass") in ANSWER_SPEECH_CLASSES


def cluster_large_pastes(pastes: list[TimelineEvent]) -> list[PasteCluster]:
    sorted_pastes = sorted(
        (
            p
            for p in pastes
            if p.detail.get("pasteOrigin") != "internal" and p.detail.get("revertedEdit") is not True
        ),
        key=lambda p: p.t_ms,
    )
    clusters: list[PasteCluster] = []
    for paste in sorted_pastes:
        el = paste.detail.get("elementId") if isinstance(paste.detail.get("elementId"), str) else None
        prev = clusters[-1] if clusters else None
        prev_el = None
        if prev and isinstance(prev.representative.detail.get("elementId"), str):
            prev_el = prev.representative.detail["elementId"]
        can_merge = (
            prev is not None
            and paste.t_ms - prev.end_ms <= PASTE_CLUSTER_GAP_MS
            and (el is None or prev_el is None or el == prev_el)
        )
        if can_merge and prev is not None:
            prev.members.append(paste)
            prev.end_ms = max(prev.end_ms, paste.end_ms or paste.t_ms)
            prev.total_chars_added += paste_size(paste)
            prev.evidence_refs.append(paste.evidence_ref)
            if paste_size(paste) > paste_size(prev.representative):
                prev.representative = paste
                prev.section_id = paste.section_id or prev.section_id
                prev.question_number = paste.question_number or prev.question_number
                prev.question_id = paste.question_id or prev.question_id
        else:
            clusters.append(
                PasteCluster(
                    representative=paste,
                    members=[paste],
                    t_ms=paste.t_ms,
                    end_ms=paste.end_ms or paste.t_ms,
                    section_id=paste.section_id,
                    question_number=paste.question_number,
                    question_id=paste.question_id,
                    total_chars_added=paste_size(paste),
                    evidence_refs=[paste.evidence_ref],
                )
            )
    return clusters


def section_score_time_stats(baseline, section_title: str | None) -> dict:
    if not section_title:
        return {}
    stats = baseline.cohort_candidate_stats or []
    score = next((s for s in stats if s.ref_id == f"section_score:{section_title}"), None)
    time = next((s for s in stats if s.ref_id == f"section_time:{section_title}"), None)
    return {
        "score_z": score.z_score if score else None,
        "time_z": time.z_score if time else None,
        "score_ref": score.ref_id if score else None,
        "time_ref": time.ref_id if time else None,
    }


def speech_episodes(timeline: list[TimelineEvent], kinds: set[str]) -> list[TimelineEvent]:
    return [e for e in timeline if e.source == "perception_episode" and e.kind in kinds]


def speech_summary_from_event(event: TimelineEvent) -> str:
    d = event.detail or {}
    summary = d.get("conversationSummaryEn") or d.get("speechSummary") or ""
    return f" Summary: {summary}" if summary else ""
