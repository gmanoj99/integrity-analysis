from __future__ import annotations

from datetime import UTC, datetime

from ..machine_facts.contracts import MachineFact
from ..machine_facts.kinds import MachineFactKind
from .contracts import SyntheticQuestionBoundary

SECTION_END_KINDS = {
    MachineFactKind.SECTION_COMPLETED.value,
    MachineFactKind.SECTION_TERMINATED.value,
    MachineFactKind.SECTION_TIMEOUT.value,
    MachineFactKind.SECTION_QUIT_BY_USER.value,
}

SQB_ANNOTATABLE_KINDS = {
    MachineFactKind.LARGE_PASTE.value,
    MachineFactKind.TEXT_CORRECTION.value,
    MachineFactKind.ACTIVITY_GAP.value,
}


def derive_boundaries_from_facts(
    facts: list[MachineFact],
    *,
    candidate_id: str,
    assessment_id: str,
    duration_ms: int,
) -> list[SyntheticQuestionBoundary]:
    section_start_ms: dict[str, int] = {}
    section_end_ms: dict[str, int] = {}

    for fact in facts:
        section_id = fact.detail.get("sectionId")
        if not isinstance(section_id, str):
            continue
        if fact.kind == MachineFactKind.SECTION_STARTED.value:
            existing = section_start_ms.get(section_id)
            if existing is None or fact.start_offset_ms < existing:
                section_start_ms[section_id] = fact.start_offset_ms
        if fact.kind in SECTION_END_KINDS:
            existing = section_end_ms.get(section_id)
            if existing is None or fact.start_offset_ms > existing:
                section_end_ms[section_id] = fact.start_offset_ms

    attempted_by_section: dict[str, dict[int, dict]] = {}
    submission_by_section: dict[str, dict[int, dict]] = {}

    for fact in facts:
        section_id = fact.detail.get("sectionId")
        if not isinstance(section_id, str):
            continue
        if fact.kind == MachineFactKind.QUESTION_ATTEMPTED.value:
            qn = fact.detail.get("questionNumber")
            if not isinstance(qn, int):
                continue
            qid = fact.detail.get("questionId") if isinstance(fact.detail.get("questionId"), str) else None
            attempted_by_section.setdefault(section_id, {}).setdefault(
                qn, {"question_id": qid, "count": 0}
            )["count"] += 1
        if fact.kind == MachineFactKind.CODE_SUBMISSION.value:
            qn = fact.detail.get("questionNumber")
            if not isinstance(qn, int):
                continue
            tsp = fact.detail.get("timeSpentSeconds")
            time_spent = tsp if isinstance(tsp, (int, float)) and tsp > 0 else None
            submission_by_section.setdefault(section_id, {}).setdefault(
                qn,
                {
                    "start_offset_ms": fact.start_offset_ms,
                    "count": 0,
                    "time_spent_seconds": time_spent,
                },
            )["count"] += 1

    boundaries: list[SyntheticQuestionBoundary] = []

    for section_id, q_attempt_map in attempted_by_section.items():
        sec_start = section_start_ms.get(section_id)
        if sec_start is None:
            continue

        q_submission_map = submission_by_section.get(section_id, {})

        if any(entry["count"] > 1 for entry in q_attempt_map.values()):
            continue
        if any(entry["count"] > 1 for entry in q_submission_map.values()):
            continue

        sorted_qns = sorted(q_attempt_map.keys())
        if not sorted_qns:
            continue

        sec_end = section_end_ms.get(section_id, duration_ms)
        submitted_qns = sorted(
            (qn for qn in sorted_qns if qn in q_submission_map),
            key=lambda qn: q_submission_map[qn]["start_offset_ms"],
        )

        window_by_qn: dict[int, dict] = {}
        prev_window_end_ms = sec_start
        for question_number in submitted_qns:
            submission = q_submission_map[question_number]
            end_offset_ms = min(max(submission["start_offset_ms"], sec_start), sec_end)
            time_spent_ms = (
                round(submission["time_spent_seconds"] * 1000)
                if submission.get("time_spent_seconds")
                else 0
            )
            if time_spent_ms > 0:
                start_offset_ms = max(sec_start, prev_window_end_ms, end_offset_ms - time_spent_ms)
                start_from_time_spent = True
            else:
                start_offset_ms = max(sec_start, prev_window_end_ms)
                start_from_time_spent = False
            if end_offset_ms < start_offset_ms:
                end_offset_ms = start_offset_ms
            prev_window_end_ms = end_offset_ms
            window_by_qn[question_number] = {
                "start_offset_ms": start_offset_ms,
                "end_offset_ms": end_offset_ms,
                "start_from_time_spent": start_from_time_spent,
                "time_spent_seconds": submission.get("time_spent_seconds"),
            }

        for question_number in sorted_qns:
            attempted = q_attempt_map[question_number]
            derived = window_by_qn.get(question_number)
            qid = attempted.get("question_id")
            if derived:
                boundaries.append(
                    SyntheticQuestionBoundary(
                        section_id=section_id,
                        question_number=question_number,
                        question_id=qid,
                        start_offset_ms=derived["start_offset_ms"],
                        end_offset_ms=derived["end_offset_ms"],
                        end_is_fallback=False,
                        start_from_time_spent=derived["start_from_time_spent"] or None,
                        authoritative_time_spent_seconds=derived.get("time_spent_seconds"),
                    )
                )
            else:
                boundaries.append(
                    SyntheticQuestionBoundary(
                        section_id=section_id,
                        question_number=question_number,
                        question_id=qid,
                        start_offset_ms=sec_start,
                        end_offset_ms=sec_start,
                        end_is_fallback=True,
                        timing_unavailable=True,
                    )
                )

    boundaries.sort(key=lambda b: (b.section_id, b.question_number))
    return boundaries


def annotate_facts_with_question_boundaries(
    facts: list[MachineFact],
    boundaries: list[SyntheticQuestionBoundary],
) -> None:
    if not boundaries:
        return
    by_section: dict[str, list[SyntheticQuestionBoundary]] = {}
    for boundary in boundaries:
        by_section.setdefault(boundary.section_id, []).append(boundary)

    for fact in facts:
        if fact.kind not in SQB_ANNOTATABLE_KINDS:
            continue
        section_id = fact.detail.get("sectionId")
        if not isinstance(section_id, str):
            continue
        section_bounds = by_section.get(section_id)
        if not section_bounds:
            continue
        annotatable = [
            b
            for b in section_bounds
            if not b.timing_unavailable and not b.end_is_fallback and b.end_offset_ms > b.start_offset_ms
        ]
        if not annotatable:
            continue
        t = fact.start_offset_ms
        match = next((b for b in annotatable if b.start_offset_ms <= t < b.end_offset_ms), None)
        if match is None:
            continue
        fact.detail["questionNumber"] = match.question_number
        if match.question_id:
            fact.detail["questionId"] = match.question_id


def is_verified_question_boundary(boundary: SyntheticQuestionBoundary) -> bool:
    return (
        not boundary.timing_unavailable
        and not boundary.end_is_fallback
        and boundary.end_offset_ms > boundary.start_offset_ms
    )
