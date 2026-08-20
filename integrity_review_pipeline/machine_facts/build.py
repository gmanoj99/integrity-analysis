"""Build the Scope 1 MachineFacts bundle from timeline + rrweb evidence."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from ..contracts.evidence import ExamMode
from ..contracts.timeline import ActivityAnchor, MasterTimeline, SessionSection
from .chunk_analysis import analyze_keystroke_chunk
from .contracts import (
    ClientReportedEvent,
    MachineFact,
    MachineFactCategoryCounts,
    MachineFactsBundle,
    MachineFactsSummary,
    RrwebChunkEvents,
)
from .kinds import LIFECYCLE_KIND_MAP, MachineFactKind
from .paste_utils import mark_paste_delete_pairs, refine_paste_origins
from .typing_extraction import extract_rrweb_typing_facts, resolve_section_id

MAX_SESSION_DURATION_MS = 8 * 60 * 60 * 1000

INPUT_KINDS = {
    MachineFactKind.COPY.value,
    MachineFactKind.PASTE.value,
    MachineFactKind.LARGE_TEXT_INSERTION.value,
    MachineFactKind.LARGE_PASTE.value,
    MachineFactKind.TEXT_CORRECTION.value,
    MachineFactKind.TYPING_STARTED.value,
    MachineFactKind.TYPING_STOPPED.value,
    MachineFactKind.RIGHT_CLICK.value,
    MachineFactKind.MCQ_ANSWER_SELECTED.value,
    MachineFactKind.MCQ_ANSWER_CHANGED.value,
    MachineFactKind.AUTOFORMAT.value,
}
FACE_KINDS = {
    MachineFactKind.FACE_NOT_VISIBLE_WARNING.value,
    MachineFactKind.FACE_WARNING_DISMISSED.value,
    MachineFactKind.CAMERA_BLOCKED.value,
}
ENVIRONMENT_KINDS = {
    MachineFactKind.MIC_DISABLED.value,
    MachineFactKind.SECOND_MONITOR_DETECTED.value,
    MachineFactKind.NOISE_DETECTED.value,
}
SCORING_KINDS = {
    MachineFactKind.SECTION_SCORE.value,
    MachineFactKind.QUESTION_ATTEMPTED.value,
    MachineFactKind.CODE_SUBMISSION.value,
}


def _extract_lifecycle_facts(
    anchors: list[ActivityAnchor],
    sections: list[SessionSection],
) -> list[MachineFact]:
    facts: list[MachineFact] = []
    for anchor in anchors:
        kind = LIFECYCLE_KIND_MAP.get(anchor.type)
        if kind is None:
            continue
        section = next(
            (item for item in sections if item.section_id == anchor.section_id),
            None,
        )
        detail: dict[str, Any] = {
            "activityType": anchor.type,
            "order": anchor.order,
        }
        if anchor.section_id:
            detail["sectionId"] = anchor.section_id
        if section and section.section_type:
            detail["sectionType"] = section.section_type
        if section and section.label:
            detail["sectionTitle"] = section.label
        if anchor.end_reason:
            detail["endReason"] = anchor.end_reason
        facts.append(
            MachineFact(
                id=str(uuid.uuid4()),
                start_offset_ms=anchor.session_offset_ms,
                end_offset_ms=anchor.session_offset_ms,
                raw_timestamp_ms=anchor.epoch_ms,
                kind=kind.value,
                source="application_event",
                evidence_source="client_reported",
                attribution="candidate",
                confidence=1.0,
                detail=detail,
            )
        )
    return facts


def _hydrate_client_events(
    events: list[ClientReportedEvent],
    session_start_ms: int,
    sections: list[SessionSection],
) -> list[MachineFact]:
    facts: list[MachineFact] = []
    for row in events:
        session_offset_ms = row.timestamp_ms - session_start_ms
        if session_offset_ms < -60_000 or session_offset_ms > MAX_SESSION_DURATION_MS:
            continue
        section_id = resolve_section_id(max(0, session_offset_ms), sections)
        source = (
            "client_reported"
            if row.signal_source == "client_reported"
            else "video_cv"
            if row.signal_source == "platform_cv"
            else "rrweb_deterministic"
        )
        detail = dict(row.detail)
        if section_id:
            detail.setdefault("sectionId", section_id)
        if row.source_ref:
            detail["sourceRef"] = row.source_ref
        facts.append(
            MachineFact(
                id=str(uuid.uuid4()),
                start_offset_ms=max(0, session_offset_ms),
                end_offset_ms=max(0, session_offset_ms),
                raw_timestamp_ms=row.timestamp_ms,
                kind=row.kind,
                source=source,
                evidence_source=row.evidence_type,
                attribution="unclear",
                confidence=1.0,
                detail=detail,
            )
        )
    return facts


def _backfill_chunk_facts(
    facts: list[MachineFact], session_start_ms: int, sections: list[SessionSection]
) -> None:
    for fact in facts:
        if fact.start_offset_ms != 0 or fact.raw_timestamp_ms is None:
            continue
        session_offset_ms = max(0, fact.raw_timestamp_ms - session_start_ms)
        fact.start_offset_ms = session_offset_ms
        fact.end_offset_ms = session_offset_ms
        section_id = resolve_section_id(session_offset_ms, sections)
        if section_id and "sectionId" not in fact.detail:
            fact.detail["sectionId"] = section_id


def _compute_summary(
    facts: list[MachineFact], sections: list[SessionSection]
) -> MachineFactsSummary:
    labels = {section.section_id: section.label for section in sections}
    by_kind: dict[str, int] = {}
    by_section: dict[str, int] = {}
    categories = MachineFactCategoryCounts()
    for fact in facts:
        by_kind[fact.kind] = by_kind.get(fact.kind, 0) + 1
        section_id = fact.detail.get("sectionId", "unknown")
        label = labels.get(section_id, section_id)
        by_section[label] = by_section.get(label, 0) + 1
        if fact.kind in {
            MachineFactKind.ASSESSMENT_STARTED.value,
            MachineFactKind.ALL_SECTIONS_COMPLETED.value,
        }:
            categories.assessment_lifecycle += 1
        elif fact.kind.startswith("SECTION_"):
            categories.section_lifecycle += 1
        elif fact.kind in {
            MachineFactKind.WINDOW_BLUR.value,
            MachineFactKind.WINDOW_FOCUS.value,
            MachineFactKind.TAB_SWITCH.value,
            MachineFactKind.FULLSCREEN_EXIT.value,
        }:
            categories.focus_and_visibility += 1
        elif fact.kind in FACE_KINDS:
            categories.face_and_camera += 1
        elif fact.kind in INPUT_KINDS:
            categories.input_and_clipboard += 1
        elif fact.kind in ENVIRONMENT_KINDS:
            categories.environment += 1
        elif fact.kind == MachineFactKind.ACTIVITY_GAP.value:
            categories.activity_gaps += 1
        elif fact.kind in SCORING_KINDS:
            categories.scoring_and_submissions += 1
        else:
            categories.other += 1
    return MachineFactsSummary(
        total_facts=len(facts),
        by_kind=by_kind,
        by_section=by_section,
        categories=categories,
    )


def build_machine_facts(
    timeline: MasterTimeline,
    *,
    exam_mode: ExamMode,
    rrweb_chunks: list[RrwebChunkEvents] | None = None,
    client_reported_events: list[ClientReportedEvent] | None = None,
    screen_synthetic_facts: list[MachineFact] | None = None,
) -> MachineFactsBundle:
    """Compute-once Scope 1 bundle. Screen and rrweb DOM paths are mutually exclusive."""
    session_start_ms = timeline.session_start_ms
    sections = timeline.sections
    all_facts: list[MachineFact] = []
    limitations: list[str] = []

    if timeline.canonical_timeline.source == "activity_logs":
        all_facts.extend(
            _extract_lifecycle_facts(timeline.canonical_timeline.anchors, sections)
        )
    else:
        limitations.append(
            "No activity logs available — lifecycle facts (ASSESSMENT_STARTED, "
            "SECTION_STARTED, etc.) are absent. Session offsets are derived from "
            "artifact timestamps only."
        )

    if client_reported_events:
        all_facts.extend(
            _hydrate_client_events(client_reported_events, session_start_ms, sections)
        )

    had_monaco = False

    if exam_mode == ExamMode.RRWEB and rrweb_chunks:
        chunk_rows = [(chunk.chunk_id, chunk.sequence, chunk.events) for chunk in rrweb_chunks]
        for _, _, events in sorted(chunk_rows, key=lambda item: item[1]):
            chunk_facts = analyze_keystroke_chunk(events)
            _backfill_chunk_facts(chunk_facts, session_start_ms, sections)
            all_facts.extend(chunk_facts)
        typing_facts, had_monaco = extract_rrweb_typing_facts(
            chunks=chunk_rows,
            session_start_ms=session_start_ms,
            sections=sections,
        )
        all_facts.extend(typing_facts)
    elif exam_mode == ExamMode.SCREEN:
        limitations.append(
            "Exam mode is screen — rrweb keystroke typing facts are skipped; "
            "use screen perception findings and synthetic SCREEN_* facts instead."
        )
    else:
        limitations.append("No rrweb keystroke chunks supplied for DOM machine facts.")

    if screen_synthetic_facts and exam_mode == ExamMode.SCREEN:
        all_facts.extend(screen_synthetic_facts)
    elif screen_synthetic_facts and exam_mode != ExamMode.SCREEN:
        limitations.append("Ignored screen synthetic facts because exam mode is not screen.")

    all_facts.sort(key=lambda fact: fact.start_offset_ms)
    refine_paste_origins(all_facts)
    mark_paste_delete_pairs(all_facts)

    if had_monaco:
        limitations.append(
            "TYPING_STARTED / TYPING_STOPPED / LARGE_PASTE / TEXT_CORRECTION facts "
            'are absent for editor chunks whose rrweb Input events have text=""; '
            "per-keystroke behavior is unrecoverable from rrweb for those sections."
        )

    limitations.extend(
        [
            "LARGE_TEXT_INSERTION (DOM-mutation-based) detection is disabled because "
            "styled-components CSS rewrites produce false positives. Use PASTE and "
            "LARGE_PASTE instead.",
            "Typing analysis covers visible standard input fields only. MCQ controls "
            "produce MCQ_ANSWER_SELECTED / MCQ_ANSWER_CHANGED; test fixtures and UI "
            "labels are ignored; autoformat is excluded from typing and paste counters. "
            "rrweb captures DOM state snapshots, not keydown/keyup timing.",
            "TEST_RESULT_OBSERVED is an in-session DOM observation only; counts may be "
            "UNKNOWN when separately-rendered text is not captured within 3 seconds.",
            "Question navigation facts are unavailable because the platform does not "
            "emit question-level rrweb custom events.",
            "Topin-derived coding outcomes, difficulty, scores, cohort analysis, and "
            "plagiarism are intentionally excluded from this standalone Scope 1 port.",
        ]
    )

    return MachineFactsBundle(
        candidate_id=timeline.candidate_id,
        assessment_id=timeline.assessment_id,
        produced_at=datetime.now(tz=UTC).isoformat(),
        session_start_ms=session_start_ms,
        duration_ms=timeline.duration_ms,
        facts=all_facts,
        summary=_compute_summary(all_facts, sections),
        limitations=limitations,
        exam_mode=exam_mode.value,
    )
