"""Unified master-clock timeline for correlation detectors."""

from __future__ import annotations

from ..findings.contracts import EvidenceFinding
from ..machine_facts.contracts import MachineFact
from ..machine_facts.kinds import MachineFactKind
from .contracts import SyntheticQuestionBoundary, TimelineEvent
from .helpers import BLUR_KINDS

TIMELINE_FACT_KINDS = {
    MachineFactKind.WINDOW_BLUR.value,
    MachineFactKind.WINDOW_FOCUS.value,
    MachineFactKind.TAB_SWITCH.value,
    MachineFactKind.LARGE_PASTE.value,
    MachineFactKind.PASTE.value,
    MachineFactKind.TEXT_CORRECTION.value,
    MachineFactKind.ACTIVITY_GAP.value,
    MachineFactKind.TYPING_STARTED.value,
    MachineFactKind.TYPING_STOPPED.value,
    MachineFactKind.SECTION_STARTED.value,
    MachineFactKind.SECTION_COMPLETED.value,
    MachineFactKind.SECTION_TERMINATED.value,
    MachineFactKind.SECTION_TIMEOUT.value,
    MachineFactKind.CODE_SUBMISSION.value,
    MachineFactKind.QUESTION_ATTEMPTED.value,
    MachineFactKind.MCQ_ANSWER_SELECTED.value,
    MachineFactKind.MCQ_ANSWER_CHANGED.value,
    MachineFactKind.TEST_RESULT_OBSERVED.value,
    MachineFactKind.SCREEN_EXTERNAL_PASTE.value,
    MachineFactKind.SCREEN_EXTERNAL_RESOURCE.value,
    MachineFactKind.SCREEN_AI_ASSISTANT_UI.value,
    MachineFactKind.SCREEN_SECONDARY_WORKSPACE.value,
    MachineFactKind.SCREEN_FULLSCREEN_LOST.value,
}

INPUT_KINDS = {
    MachineFactKind.LARGE_PASTE.value,
    MachineFactKind.TEXT_CORRECTION.value,
    MachineFactKind.TYPING_STARTED.value,
    MachineFactKind.TYPING_STOPPED.value,
}


def build_correlation_timeline(
    facts: list[MachineFact],
    boundaries: list[SyntheticQuestionBoundary],
    video_findings: list[EvidenceFinding],
    screen_findings: list[EvidenceFinding] | None = None,
) -> list[TimelineEvent]:
    events: list[TimelineEvent] = []
    for fact in facts:
        if fact.kind not in TIMELINE_FACT_KINDS:
            continue
        detail = dict(fact.detail)
        section_id = detail.get("sectionId") if isinstance(detail.get("sectionId"), str) else None
        section_title = detail.get("sectionTitle") if isinstance(detail.get("sectionTitle"), str) else None
        question_number = detail.get("questionNumber") if isinstance(detail.get("questionNumber"), int) else None
        question_id = detail.get("questionId") if isinstance(detail.get("questionId"), str) else None
        is_submission = fact.kind == MachineFactKind.CODE_SUBMISSION.value
        events.append(
            TimelineEvent(
                t_ms=fact.start_offset_ms,
                end_ms=fact.end_offset_ms if fact.end_offset_ms != fact.start_offset_ms else None,
                source="submission" if is_submission else "fact",
                kind=fact.kind,
                section_id=section_id,
                section_title=section_title,
                question_number=question_number,
                question_id=question_id,
                detail=detail,
                evidence_ref=f"fact:{fact.id}",
            )
        )

    for boundary in boundaries:
        events.append(
            TimelineEvent(
                t_ms=boundary.start_offset_ms,
                end_ms=boundary.end_offset_ms,
                source="sqb",
                kind="SQB_WINDOW",
                section_id=boundary.section_id,
                question_number=boundary.question_number,
                question_id=boundary.question_id,
                detail={
                    "timingUnavailable": boundary.timing_unavailable,
                    "endIsFallback": boundary.end_is_fallback,
                    "authoritativeTimeSpentSeconds": boundary.authoritative_time_spent_seconds,
                },
                evidence_ref=f"sqb:s={boundary.section_id}:q={boundary.question_number}",
            )
        )

    for finding in video_findings:
        start, end = finding.timestamp_window_ms
        speech = finding.speech_proof
        detail = {
            "severity": finding.severity,
            "conversationSummaryEn": speech.conversation_summary_en if speech else None,
            "speechContentClass": speech.speech_content_class if speech else None,
            "secondPersonLooksLike": None,
        }
        events.append(
            TimelineEvent(
                t_ms=start,
                end_ms=end,
                source="perception_episode",
                kind=finding.event_type,
                detail=detail,
                evidence_ref=finding.evidence_ref or f"video:{finding.id}",
            )
        )

    for finding in screen_findings or []:
        start, end = finding.timestamp_window_ms
        qn = finding.screen_proof.question_number if finding.screen_proof else None
        events.append(
            TimelineEvent(
                t_ms=start,
                end_ms=end,
                source="screen_episode",
                kind=finding.event_type,
                question_number=qn,
                detail={
                    "pastedExcerpt": finding.screen_proof.pasted_excerpt if finding.screen_proof else None
                },
                evidence_ref=finding.evidence_ref or f"screen:{finding.id}",
            )
        )

    events.sort(key=lambda event: (event.t_ms, event.end_ms or event.t_ms))
    return events
