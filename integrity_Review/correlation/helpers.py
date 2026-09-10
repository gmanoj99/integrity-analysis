"""Shared correlation helpers — port of correlationEngine.ts utilities."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..machine_facts.contracts import MachineFact
from ..machine_facts.kinds import MachineFactKind
from .contracts import (
    CorrelationConfig,
    DetectCtx,
    SyntheticQuestionBoundary,
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
VIDEO_OVERLAP_EVENT_TYPES = {"phone_usage", "multiple_faces", "external_help"}
GAZE_EVENT_TYPES = {"suspicious_eye_movement"}
SPEECH_INTEGRITY_EVENT_TYPES = {
    "whisper_or_dictation",
    "off_camera_voice_coaching",
    "device_call_during_attempt",
    "av_speech_without_lips",
    "discussing_solution_audio",
}
WHISPER_DICTATION_TYPES = {
    "whisper_or_dictation",
    "off_camera_voice_coaching",
    "discussing_solution_audio",
}
AV_MISMATCH_TYPES = {"av_speech_without_lips"}
CAMERA_ABSENT_TYPES = {"no_candidate", "left_examination_area", "left_seat_body_present"}

# A second person the classifier flagged as helping, and speech the classifier
# judged to be about the exam itself. ``discussing_solution_audio`` is the only
# speech event type any classifier emits today, so the class carried in
# ``detail["speechContentClass"]`` is what separates coaching from chatter.
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


def gap_duration_ms(event: TimelineEvent) -> int:
    duration = event.detail.get("durationMs")
    if isinstance(duration, (int, float)) and duration > 0:
        return int(duration)
    if event.end_ms is not None and event.end_ms > event.t_ms:
        return event.end_ms - event.t_ms
    return 0


def paste_size(event: TimelineEvent) -> int:
    chars = event.detail.get("charsAdded")
    return int(chars) if isinstance(chars, (int, float)) else 0


def perception_episodes_by_kind(timeline: list[TimelineEvent], kind: str) -> list[TimelineEvent]:
    return [e for e in timeline if e.source == "perception_episode" and e.kind == kind]


def perception_episodes_by_kinds(
    timeline: list[TimelineEvent], kinds: set[str]
) -> list[TimelineEvent]:
    """``perception_episodes_by_kind`` for a set — one classifier concept often
    surfaces under more than one event type (a helper beside the candidate is
    ``external_help``; two faces in frame is ``multiple_faces``)."""

    return [e for e in timeline if e.source == "perception_episode" and e.kind in kinds]


def is_answer_speech(event: TimelineEvent) -> bool:
    """Speech the classifier judged to be about the exam's answers.

    The timeline carries one speech event type, so the episode's
    ``speechContentClass`` is what separates coaching from a room where people
    happen to be talking. An episode without a class is not assumed to qualify.
    """

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


def is_question_correct(detail: dict) -> bool:
    max_score = detail.get("maxScore") or detail.get("max_question_score")
    score = detail.get("latestScore") or detail.get("score")
    if isinstance(max_score, (int, float)) and max_score > 0 and isinstance(score, (int, float)):
        return score >= max_score
    passed = detail.get("passedTestCases")
    total = detail.get("totalTestCases")
    if isinstance(passed, int) and isinstance(total, int) and total > 0:
        return passed == total
    ev = detail.get("evaluationResult")
    if isinstance(ev, str):
        upper = ev.upper()
        return upper in {"CORRECT", "PASS"}
    return False


def build_fast_correct_questions(ctx: DetectCtx) -> list[FastCorrectQuestion]:
    facts, boundaries, baseline, config = ctx.facts, ctx.boundaries, ctx.baseline, ctx.config
    attempt_by_key: dict[str, MachineFact] = {}
    sub_by_key: dict[str, MachineFact] = {}
    for fact in facts:
        d = fact.detail
        qn = d.get("questionNumber") if isinstance(d.get("questionNumber"), int) else None
        qid = d.get("questionId") if isinstance(d.get("questionId"), str) else None
        key = qid or (f"n:{qn}" if qn is not None else "")
        if not key:
            continue
        if fact.kind == MachineFactKind.QUESTION_ATTEMPTED.value:
            attempt_by_key[key] = fact
        if fact.kind == MachineFactKind.CODE_SUBMISSION.value:
            sub_by_key[key] = fact

    question_acc_stats = [
        s
        for s in baseline.cohort_candidate_stats or []
        if s.ref_id.startswith("question_accuracy:")
    ]

    out: list[FastCorrectQuestion] = []
    for boundary in boundaries:
        if boundary.timing_unavailable or boundary.end_is_fallback:
            continue
        key = boundary.question_id or f"n:{boundary.question_number}"
        attempt = attempt_by_key.get(key) or attempt_by_key.get(f"n:{boundary.question_number}")
        sub = sub_by_key.get(key) or sub_by_key.get(f"n:{boundary.question_number}")
        detail = {**(attempt.detail if attempt else {}), **(sub.detail if sub else {})}
        correct = is_question_correct(detail)
        time_spent = (
            detail.get("timeSpentSeconds")
            if isinstance(detail.get("timeSpentSeconds"), (int, float))
            else boundary.authoritative_time_spent_seconds
        )
        window_sec = (boundary.end_offset_ms - boundary.start_offset_ms) / 1000
        effective_sec = time_spent if time_spent is not None else window_sec
        diff = str(detail.get("difficulty", "")).upper()
        abs_limit = config.hard_fast_max_seconds if diff == "HARD" else config.fast_correct_max_seconds
        speed_basis = "none"
        fast = False
        if isinstance(time_spent, (int, float)) and time_spent > 0 and time_spent <= abs_limit:
            fast = True
            speed_basis = "absolute"
        peer = next(
            (s for s in question_acc_stats if s.ref_id == f"question_accuracy:{boundary.question_id}"),
            None,
        ) if boundary.question_id else None
        if peer and peer.z_score is not None and peer.z_score >= 1.0 and effective_sec <= abs_limit * 1.25:
            fast = True
            speed_basis = "cohort"
        score = detail.get("latestScore") or detail.get("score")
        max_score = detail.get("maxScore")
        out.append(
            FastCorrectQuestion(
                question_number=boundary.question_number,
                question_id=boundary.question_id,
                section_id=boundary.section_id,
                window_ms=(boundary.start_offset_ms, boundary.end_offset_ms),
                correct=correct,
                fast=fast,
                speed_basis=speed_basis,
                time_spent_seconds=float(time_spent) if isinstance(time_spent, (int, float)) else None,
                score=float(score) if isinstance(score, (int, float)) else None,
                max_score=float(max_score) if isinstance(max_score, (int, float)) else None,
                evidence_refs=[
                    f"sqb:s={boundary.section_id}:q={boundary.question_number}",
                    *( [f"fact:{sub.id}"] if sub else [] ),
                    *( [f"fact:{attempt.id}"] if attempt else [] ),
                ],
            )
        )
    return out


def build_fast_mcq_answer_bursts(ctx: DetectCtx) -> list[FastMcqBurst]:
    timeline, config = ctx.timeline, ctx.config
    selects = sorted(
        (e for e in timeline if e.kind == MachineFactKind.MCQ_ANSWER_SELECTED.value),
        key=lambda e: e.t_ms,
    )
    if not selects:
        return []
    bursts: list[FastMcqBurst] = []
    cur = [selects[0]]
    for i in range(1, len(selects)):
        prev, nxt = selects[i - 1], selects[i]
        gap = nxt.t_ms - prev.t_ms
        if gap <= config.mcq_fast_gap_ms and same_section(prev.section_id, nxt.section_id):
            cur.append(nxt)
        else:
            if len(cur) >= config.mcq_fast_min_answers:
                bursts.append(_burst_from_selects(cur))
            cur = [nxt]
    if len(cur) >= config.mcq_fast_min_answers:
        bursts.append(_burst_from_selects(cur))
    return bursts


def _burst_from_selects(selects: list[TimelineEvent]) -> FastMcqBurst:
    gaps = [selects[i].t_ms - selects[i - 1].t_ms for i in range(1, len(selects))]
    gaps.sort()
    median_gap_ms = gaps[len(gaps) // 2] if gaps else 0
    return FastMcqBurst(
        section_id=selects[0].section_id,
        section_title=selects[0].section_title,
        start_ms=selects[0].t_ms,
        end_ms=selects[-1].t_ms,
        answer_count=len(selects),
        median_gap_ms=median_gap_ms,
        evidence_refs=[s.evidence_ref for s in selects[:4]],
    )


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


def difficulty_by_question(facts: list[MachineFact]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for fact in facts:
        if fact.kind != MachineFactKind.QUESTION_ATTEMPTED.value:
            continue
        d = fact.detail
        difficulty = str(d.get("difficulty", "")).upper()
        if not difficulty:
            continue
        qn = d.get("questionNumber") if isinstance(d.get("questionNumber"), int) else None
        qid = d.get("questionId") if isinstance(d.get("questionId"), str) else None
        key = qid or (f"n:{qn}" if qn is not None else "")
        if key:
            out[key] = {"difficulty": difficulty, "question_number": qn, "question_id": qid}
    return out


def find_question_at(
    boundaries: list[SyntheticQuestionBoundary],
    t_ms: int,
    section_id: str | None = None,
) -> SyntheticQuestionBoundary | None:
    for boundary in boundaries:
        if section_id and boundary.section_id != section_id:
            continue
        if boundary.start_offset_ms <= t_ms < boundary.end_offset_ms:
            return boundary
    return None


def submission_time_spent_seconds(event: TimelineEvent) -> float | None:
    tsp = event.detail.get("timeSpentSeconds")
    return float(tsp) if isinstance(tsp, (int, float)) and tsp >= 0 else None


def speech_episodes(timeline: list[TimelineEvent], kinds: set[str]) -> list[TimelineEvent]:
    return [e for e in timeline if e.source == "perception_episode" and e.kind in kinds]


def speech_summary_from_event(event: TimelineEvent) -> str:
    d = event.detail or {}
    summary = d.get("conversationSummaryEn") or d.get("speechSummary") or ""
    return f" Summary: {summary}" if summary else ""


def is_hard_question(ctx: DetectCtx, q: FastCorrectQuestion) -> bool:
    for fact in ctx.facts:
        if fact.kind != MachineFactKind.QUESTION_ATTEMPTED.value:
            continue
        d = fact.detail
        qid = d.get("questionId") if isinstance(d.get("questionId"), str) else None
        qn = d.get("questionNumber") if isinstance(d.get("questionNumber"), int) else None
        if q.question_id and qid == q.question_id:
            return str(d.get("difficulty", "")).upper() == "HARD"
        if qn == q.question_number:
            return str(d.get("difficulty", "")).upper() == "HARD"
    return False


def has_external_paste_in_window(
    timeline: list[TimelineEvent],
    window_ms: tuple[int, int],
    question_number: int | None = None,
) -> bool:
    pastes = cluster_large_pastes(
        [e for e in timeline if e.kind == MachineFactKind.LARGE_PASTE.value]
    )
    return any(
        p.t_ms >= window_ms[0]
        and p.t_ms <= window_ms[1]
        and (p.question_number is None or question_number is None or p.question_number == question_number)
        for p in pastes
    )


def external_paste_after_leave_rationale(base: str, q: FastCorrectQuestion | None) -> str:
    if q is None:
        return f"{base} — copy-paste from outside after leaving the exam tab."
    spent = q.time_spent_seconds or round((q.window_ms[1] - q.window_ms[0]) / 1000)
    return (
        f"Copy-paste from outside after leaving the exam tab, then fast+correct Q{q.question_number}"
        f" (~{spent}s, speed={q.speed_basis}). {base}"
    )
