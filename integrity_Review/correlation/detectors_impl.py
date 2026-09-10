"""Remaining correlation detectors — split from detectors.py for maintainability."""

from __future__ import annotations

from ..machine_facts.kinds import MachineFactKind
from .contracts import DetectCtx, RiskContribution, TimelineEvent, make_contribution
from .helpers import (
    ANSWER_SPEECH_EVENT_TYPES,
    AV_MISMATCH_TYPES,
    BLUR_KINDS,
    CAMERA_ABSENT_TYPES,
    GAZE_EVENT_TYPES,
    INPUT_AFTER_GAZE_KINDS,
    SECOND_PERSON_EVENT_TYPES,
    SPEECH_INTEGRITY_EVENT_TYPES,
    VIDEO_OVERLAP_EVENT_TYPES,
    WHISPER_DICTATION_TYPES,
    is_answer_speech,
    perception_episodes_by_kinds,
    build_fast_correct_questions,
    build_fast_mcq_answer_bursts,
    cluster_large_pastes,
    difficulty_by_question,
    external_paste_after_leave_rationale,
    find_question_at,
    gap_duration_ms,
    has_external_paste_in_window,
    intervals_overlap,
    is_hard_question,
    paste_size,
    perception_episodes_by_kind,
    same_section,
    section_score_time_stats,
    speech_episodes,
    speech_summary_from_event,
    submission_time_spent_seconds,
)
from .chains import dedupe_contributions


class _PhonePrecursor:
    def __init__(self, phone):
        self.t_ms = phone.end_ms or phone.t_ms
        self.end_ms = phone.end_ms
        self.kind = "phone_usage"
        self.section_id = phone.section_id
        self.evidence_ref = phone.evidence_ref


def detect_paste_after_blur(ctx: DetectCtx) -> list[RiskContribution]:
    timeline, config = ctx.timeline, ctx.config
    blurs = [e for e in timeline if e.kind in BLUR_KINDS]
    phone_ends = perception_episodes_by_kind(timeline, "phone_usage")
    precursors = sorted([*blurs, *[_PhonePrecursor(p) for p in phone_ends]], key=lambda e: e.t_ms)
    paste_clusters = cluster_large_pastes(
        [e for e in timeline if e.kind == MachineFactKind.LARGE_PASTE.value]
    )
    out: list[RiskContribution] = []
    used: set[str] = set()
    for precursor in precursors:
        for cluster in paste_clusters:
            key = "|".join(cluster.evidence_refs)
            if key in used or cluster.t_ms < precursor.t_ms:
                continue
            delta = cluster.t_ms - precursor.t_ms
            if delta > config.paste_after_blur_ms or not same_section(precursor.section_id, cluster.section_id):
                continue
            used.add(key)
            is_tab = precursor.kind == MachineFactKind.TAB_SWITCH.value
            is_phone = precursor.kind == "phone_usage"
            intensity = max(0.35, 1 - delta / config.paste_after_blur_ms)
            if is_tab:
                intensity = min(1.0, intensity + 0.1)
            if is_phone:
                intensity = min(1.0, intensity + 0.05)
            phone_near = None if is_phone else next(
                (
                    p
                    for p in phone_ends
                    if intervals_overlap(
                        p.t_ms,
                        p.end_ms or p.t_ms,
                        precursor.t_ms,
                        cluster.end_ms,
                        config.video_overlap_pad_ms,
                    )
                ),
                None,
            )
            n = len(cluster.members)
            label = "phone_usage" if is_phone else precursor.kind
            out.append(
                make_contribution(
                    "paste_after_blur",
                    tier=2,
                    intensity=intensity,
                    timing_basis="section_clock",
                    section_id=cluster.section_id or precursor.section_id,
                    question_number=cluster.question_number,
                    question_id=cluster.question_id,
                    time_range_ms=(precursor.t_ms, cluster.end_ms),
                    evidence_refs=[
                        precursor.evidence_ref,
                        *cluster.evidence_refs,
                        *( [phone_near.evidence_ref] if phone_near else [] ),
                    ][:4],
                    rationale=(
                        f"{label} → external LARGE_PASTE {delta}ms later"
                        + (" (phone visible during acquisition→paste chain)" if phone_near else "")
                        + (f" ({n} near-duplicate pastes clustered)" if n > 1 else "")
                        + f" (window {config.paste_after_blur_ms}ms)"
                    ),
                )
            )
    return out


def detect_idle_then_large_paste(ctx: DetectCtx) -> list[RiskContribution]:
    timeline, config = ctx.timeline, ctx.config
    gaps = [e for e in timeline if e.kind == MachineFactKind.ACTIVITY_GAP.value]
    paste_clusters = cluster_large_pastes(
        [e for e in timeline if e.kind == MachineFactKind.LARGE_PASTE.value]
    )
    out: list[RiskContribution] = []
    used: set[str] = set()
    for gap in gaps:
        dur = gap_duration_ms(gap)
        if dur < config.idle_gap_min_ms:
            continue
        gap_end = gap.end_ms or gap.t_ms + dur
        for cluster in paste_clusters:
            key = "|".join(cluster.evidence_refs)
            if key in used or cluster.t_ms < gap_end:
                continue
            delta = cluster.t_ms - gap_end
            if delta > config.idle_then_paste_ms or not same_section(gap.section_id, cluster.section_id):
                continue
            used.add(key)
            intensity = min(
                1.0,
                0.4 + (dur / (config.idle_gap_min_ms * 3)) * 0.4 + (1 - delta / config.idle_then_paste_ms) * 0.2,
            )
            n = len(cluster.members)
            out.append(
                make_contribution(
                    "idle_then_large_paste",
                    tier=2,
                    intensity=intensity,
                    timing_basis="section_clock",
                    section_id=cluster.section_id or gap.section_id,
                    question_number=cluster.question_number,
                    question_id=cluster.question_id,
                    time_range_ms=(gap.t_ms, cluster.end_ms),
                    evidence_refs=[gap.evidence_ref, *cluster.evidence_refs][:4],
                    rationale=(
                        f"ACTIVITY_GAP {dur}ms ended then LARGE_PASTE {delta}ms later"
                        + (f" ({n} near-duplicate pastes clustered)" if n > 1 else "")
                    ),
                )
            )
    return out


def detect_video_overlap(ctx: DetectCtx) -> list[RiskContribution]:
    timeline, boundaries, config = ctx.timeline, ctx.boundaries, ctx.config
    episodes = [
        e
        for e in timeline
        if e.source == "perception_episode" and e.kind in VIDEO_OVERLAP_EVENT_TYPES
    ]
    blurs = [e for e in timeline if e.kind in BLUR_KINDS]
    paste_clusters = cluster_large_pastes(
        [e for e in timeline if e.kind == MachineFactKind.LARGE_PASTE.value]
    )
    submissions = [
        e
        for e in timeline
        if e.kind == MachineFactKind.CODE_SUBMISSION.value or e.source == "submission"
    ]
    out: list[RiskContribution] = []
    for ep in episodes:
        ep_end = ep.end_ms or ep.t_ms
        for cluster in paste_clusters:
            if not intervals_overlap(
                ep.t_ms, ep_end, cluster.t_ms, cluster.end_ms, config.video_overlap_pad_ms
            ):
                continue
            precursor = next(
                (
                    b
                    for b in blurs
                    if b.t_ms <= cluster.t_ms
                    and cluster.t_ms - b.t_ms <= config.paste_after_blur_ms
                    and same_section(b.section_id, cluster.section_id)
                ),
                None,
            )
            if precursor is None:
                continue
            q = None if cluster.question_number is not None else find_question_at(
                boundaries, cluster.t_ms, cluster.section_id
            )
            qn = cluster.question_number or (q.question_number if q else None)
            n = len(cluster.members)
            out.append(
                make_contribution(
                    "video_overlap_paste",
                    tier=1 if qn is not None else 2,
                    intensity=0.8,
                    timing_basis="sqb_submission_window" if qn is not None else "section_clock",
                    section_id=cluster.section_id,
                    question_number=qn,
                    question_id=cluster.question_id or (q.question_id if q else None),
                    time_range_ms=(min(ep.t_ms, precursor.t_ms), max(ep_end, cluster.end_ms)),
                    evidence_refs=[precursor.evidence_ref, ep.evidence_ref, *cluster.evidence_refs][:4],
                    rationale=(
                        f"{precursor.kind} → external LARGE_PASTE with M0 {ep.kind} visible"
                        + (f" — {n} near-duplicate pastes clustered" if n > 1 else "")
                    ),
                )
            )
        for sub in submissions:
            if not intervals_overlap(
                ep.t_ms, ep_end, sub.t_ms, sub.end_ms or sub.t_ms, config.video_overlap_pad_ms
            ):
                continue
            q = None if sub.question_number is not None else find_question_at(
                boundaries, sub.t_ms, sub.section_id
            )
            qn = sub.question_number or (q.question_number if q else None)
            out.append(
                make_contribution(
                    "video_overlap_submission",
                    tier=1 if qn is not None else 2,
                    intensity=0.85,
                    timing_basis="sqb_submission_window" if qn is not None else "section_clock",
                    section_id=sub.section_id,
                    question_number=qn,
                    question_id=sub.question_id or (q.question_id if q else None),
                    time_range_ms=(min(ep.t_ms, sub.t_ms), max(ep_end, sub.t_ms)),
                    evidence_refs=[ep.evidence_ref, sub.evidence_ref],
                    rationale=f"M0 {ep.kind} overlaps CODE_SUBMISSION (±{config.video_overlap_pad_ms}ms)",
                )
            )
    return out


def detect_gaze_off_then_input(ctx: DetectCtx) -> list[RiskContribution]:
    timeline, config, boundaries = ctx.timeline, ctx.config, ctx.boundaries
    gazes = [e for e in timeline if e.source == "perception_episode" and e.kind in GAZE_EVENT_TYPES]
    inputs = [
        e
        for e in timeline
        if e.kind in INPUT_AFTER_GAZE_KINDS
        and e.detail.get("pasteOrigin") != "internal"
        and e.detail.get("revertedEdit") is not True
    ]
    out: list[RiskContribution] = []
    used_input: set[str] = set()
    used_gaze: set[str] = set()
    for gaze in gazes:
        if gaze.evidence_ref in used_gaze:
            continue
        gaze_end = gaze.end_ms or gaze.t_ms
        best = None
        for inp in inputs:
            if inp.evidence_ref in used_input or inp.t_ms < gaze_end:
                continue
            delta = inp.t_ms - gaze_end
            if delta > config.gaze_off_then_input_ms:
                continue
            if best is None or delta < best[1]:
                best = (inp, delta)
        if best is None:
            continue
        inp, delta = best
        used_gaze.add(gaze.evidence_ref)
        used_input.add(inp.evidence_ref)
        q = None if inp.question_number is not None else find_question_at(
            boundaries, inp.t_ms, inp.section_id
        )
        out.append(
            make_contribution(
                "gaze_off_then_input",
                tier=2,
                intensity=max(0.4, 1 - delta / config.gaze_off_then_input_ms),
                requires_corroboration=True,
                timing_basis="section_clock",
                section_id=inp.section_id or gaze.section_id,
                section_title=inp.section_title or gaze.section_title,
                question_number=inp.question_number or (q.question_number if q else None),
                question_id=inp.question_id or (q.question_id if q else None),
                time_range_ms=(gaze_end, inp.t_ms),
                evidence_refs=[gaze.evidence_ref, inp.evidence_ref],
                rationale=f"suspicious_eye_movement ended then {inp.kind} {delta}ms later",
            )
        )
    return out


def detect_improbable_hard_fast_paste(ctx: DetectCtx) -> list[RiskContribution]:
    timeline, boundaries, facts, config = ctx.timeline, ctx.boundaries, ctx.facts, ctx.config
    if not boundaries:
        return []
    diff_map = difficulty_by_question(facts)
    pastes = [
        e
        for e in timeline
        if e.kind == MachineFactKind.LARGE_PASTE.value
        and e.detail.get("pasteOrigin") != "internal"
        and e.detail.get("revertedEdit") is not True
    ]
    submissions = [e for e in timeline if e.kind == MachineFactKind.CODE_SUBMISSION.value]
    out: list[RiskContribution] = []
    for boundary in boundaries:
        key = boundary.question_id or f"n:{boundary.question_number}"
        meta = diff_map.get(key) or diff_map.get(f"n:{boundary.question_number}")
        if not meta or meta["difficulty"] != "HARD":
            continue
        window_sec = (boundary.end_offset_ms - boundary.start_offset_ms) / 1000
        sub = next(
            (
                s
                for s in submissions
                if s.question_number == boundary.question_number
                or (s.question_id and s.question_id == boundary.question_id)
                or boundary.start_offset_ms <= s.t_ms < boundary.end_offset_ms + 1
            ),
            None,
        )
        spent = submission_time_spent_seconds(sub) if sub else None
        effective_sec = spent if spent is not None else window_sec
        if effective_sec > config.hard_fast_max_seconds:
            continue
        window_pastes = [
            p
            for p in pastes
            if (
                boundary.start_offset_ms <= p.t_ms < boundary.end_offset_ms
                or p.question_number == boundary.question_number
                or (p.question_id and p.question_id == boundary.question_id)
            )
        ]
        if not window_pastes:
            continue
        near_sub = (
            [p for p in window_pastes if p.t_ms <= sub.t_ms and sub.t_ms - p.t_ms <= config.paste_near_submission_ms]
            if sub
            else window_pastes
        )
        chosen = (near_sub or window_pastes)[0]
        intensity = min(
            1.0,
            0.5 + (1 - effective_sec / config.hard_fast_max_seconds) * 0.35 + (0.15 if len(window_pastes) > 1 else 0),
        )
        out.append(
            make_contribution(
                "improbable_hard_fast_paste",
                tier=1,
                intensity=intensity,
                requires_corroboration=True,
                timing_basis="sqb_submission_window",
                section_id=boundary.section_id,
                question_number=boundary.question_number,
                question_id=boundary.question_id,
                time_range_ms=(boundary.start_offset_ms, boundary.end_offset_ms),
                evidence_refs=[
                    f"sqb:s={boundary.section_id}:q={boundary.question_number}",
                    chosen.evidence_ref,
                    *( [sub.evidence_ref] if sub else [] ),
                ][:4],
                rationale=(
                    f"HARD Q{boundary.question_number} finished in ~{round(effective_sec)}s "
                    f"(≤{config.hard_fast_max_seconds}s) with LARGE_PASTE before submission"
                ),
            )
        )
    return out


def detect_paste_on_coding_question(ctx: DetectCtx) -> list[RiskContribution]:
    timeline, config = ctx.timeline, ctx.config
    pastes = [
        e
        for e in timeline
        if e.kind == MachineFactKind.LARGE_PASTE.value
        and isinstance(e.question_number, int)
        and e.detail.get("pasteOrigin") != "internal"
    ]
    by_q: dict[int, list] = {}
    for paste in pastes:
        by_q.setdefault(paste.question_number, []).append(paste)
    out: list[RiskContribution] = []
    for qn, items in by_q.items():
        if len(items) < config.paste_cluster_min_count:
            biggest = max(paste_size(p) for p in items)
            if not (len(items) == 1 and biggest >= config.min_paste_size_for_tier1 * 4):
                continue
        t0 = min(p.t_ms for p in items)
        t1 = max(p.t_ms for p in items)
        out.append(
            make_contribution(
                "paste_on_coding_question",
                tier=1,
                intensity=min(1.0, 0.5 + len(items) * 0.2),
                timing_basis="sqb_submission_window",
                section_id=items[0].section_id,
                question_number=qn,
                question_id=items[0].question_id,
                time_range_ms=(t0, t1),
                evidence_refs=[p.evidence_ref for p in items[:4]],
                rationale=f"{len(items)} LARGE_PASTE event(s) on coding question {qn} (verified SQB window)",
            )
        )
    return out


def detect_phone_fast_correct(ctx: DetectCtx) -> list[RiskContribution]:
    timeline, config = ctx.timeline, ctx.config
    phones = perception_episodes_by_kind(timeline, "phone_usage")
    fast_correct = [q for q in build_fast_correct_questions(ctx) if q.fast and q.correct]
    out: list[RiskContribution] = []
    for phone in phones:
        phone_end = phone.end_ms or phone.t_ms
        for q in fast_correct:
            if not intervals_overlap(
                phone.t_ms, phone_end, q.window_ms[0], q.window_ms[1], config.video_overlap_pad_ms
            ):
                continue
            spent = q.time_spent_seconds or round((q.window_ms[1] - q.window_ms[0]) / 1000)
            out.append(
                make_contribution(
                    "phone_fast_correct",
                    tier=1,
                    intensity=min(1.0, 0.55 + (1 - spent / config.fast_correct_max_seconds) * 0.35),
                    timing_basis="sqb_submission_window",
                    section_id=q.section_id,
                    question_number=q.question_number,
                    question_id=q.question_id,
                    time_range_ms=(min(phone.t_ms, q.window_ms[0]), max(phone_end, q.window_ms[1])),
                    evidence_refs=[phone.evidence_ref, *q.evidence_refs][:4],
                    rationale=(
                        f"phone_usage overlaps fast+correct Q{q.question_number} (~{spent}s, "
                        f"score {q.score or '?'}/{q.max_score or '?'}, speed={q.speed_basis})"
                    ),
                )
            )
    return out


def detect_phone_fast_mcq_answers(ctx: DetectCtx) -> list[RiskContribution]:
    timeline, config, baseline = ctx.timeline, ctx.config, ctx.baseline
    phones = perception_episodes_by_kind(timeline, "phone_usage")
    bursts = build_fast_mcq_answer_bursts(ctx)
    out: list[RiskContribution] = []
    for phone in phones:
        phone_end = phone.end_ms or phone.t_ms
        for burst in bursts:
            if not intervals_overlap(
                phone.t_ms, phone_end, burst.start_ms, burst.end_ms, config.video_overlap_pad_ms
            ):
                continue
            stats = section_score_time_stats(baseline, burst.section_title)
            intensity = min(
                1.0,
                0.45
                + min(0.3, burst.answer_count * 0.05)
                + min(0.2, (config.mcq_fast_gap_ms - burst.median_gap_ms) / config.mcq_fast_gap_ms) * 0.2,
            )
            if stats.get("score_z") is not None and stats["score_z"] >= 1:
                intensity = min(1.0, intensity + 0.1)
            if stats.get("time_z") is not None and stats["time_z"] <= -1:
                intensity = min(1.0, intensity + 0.1)
            cohort_refs = [r for r in (stats.get("score_ref"), stats.get("time_ref")) if r]
            out.append(
                make_contribution(
                    "phone_fast_mcq_answers",
                    tier=1,
                    intensity=intensity,
                    timing_basis="section_clock",
                    section_id=burst.section_id,
                    section_title=burst.section_title,
                    time_range_ms=(min(phone.t_ms, burst.start_ms), max(phone_end, burst.end_ms)),
                    evidence_refs=[phone.evidence_ref, *burst.evidence_refs][:4],
                    cohort_ref_ids=cohort_refs or None,
                    rationale=(
                        f"phone_usage during rapid MCQ commits ({burst.answer_count} selects, "
                        f"median gap {burst.median_gap_ms}ms)"
                    ),
                )
            )
    return out


def detect_gaze_fast_correct(ctx: DetectCtx) -> list[RiskContribution]:
    timeline, config = ctx.timeline, ctx.config
    gazes = perception_episodes_by_kind(timeline, "suspicious_eye_movement")
    fast_correct = [q for q in build_fast_correct_questions(ctx) if q.fast and q.correct]
    external_pastes = cluster_large_pastes(
        [e for e in timeline if e.kind == MachineFactKind.LARGE_PASTE.value]
    )
    out: list[RiskContribution] = []
    for gaze in gazes:
        gaze_end = gaze.end_ms or gaze.t_ms
        for q in fast_correct:
            if not intervals_overlap(
                gaze.t_ms, gaze_end, q.window_ms[0], q.window_ms[1], config.video_overlap_pad_ms
            ):
                continue
            if any(
                p.t_ms >= q.window_ms[0]
                and p.t_ms <= q.window_ms[1]
                and (p.question_number is None or p.question_number == q.question_number)
                for p in external_pastes
            ):
                continue
            spent = q.time_spent_seconds or round((q.window_ms[1] - q.window_ms[0]) / 1000)
            out.append(
                make_contribution(
                    "gaze_fast_correct",
                    tier=1,
                    intensity=min(1.0, 0.5 + (1 - spent / config.fast_correct_max_seconds) * 0.35),
                    requires_corroboration=True,
                    timing_basis="sqb_submission_window",
                    section_id=q.section_id,
                    question_number=q.question_number,
                    question_id=q.question_id,
                    time_range_ms=(min(gaze.t_ms, q.window_ms[0]), max(gaze_end, q.window_ms[1])),
                    evidence_refs=[gaze.evidence_ref, *q.evidence_refs][:4],
                    rationale=(
                        f"suspicious_eye_movement during fast+correct Q{q.question_number} (~{spent}s) "
                        "with no external paste"
                    ),
                )
            )
    return out


def detect_gaze_with_device(ctx: DetectCtx) -> list[RiskContribution]:
    timeline, config = ctx.timeline, ctx.config
    gazes = perception_episodes_by_kind(timeline, "suspicious_eye_movement")
    phones = perception_episodes_by_kind(timeline, "phone_usage")
    out: list[RiskContribution] = []
    used: set[str] = set()
    for gaze in gazes:
        gaze_end = gaze.end_ms or gaze.t_ms
        for phone in phones:
            key = f"{gaze.evidence_ref}|{phone.evidence_ref}"
            if key in used:
                continue
            phone_end = phone.end_ms or phone.t_ms
            if not intervals_overlap(
                gaze.t_ms, gaze_end, phone.t_ms, phone_end, config.gaze_device_cooccur_ms
            ):
                continue
            used.add(key)
            out.append(
                make_contribution(
                    "gaze_with_device",
                    tier=1,
                    intensity=0.75,
                    timing_basis="section_clock",
                    time_range_ms=(min(gaze.t_ms, phone.t_ms), max(gaze_end, phone_end)),
                    evidence_refs=[gaze.evidence_ref, phone.evidence_ref],
                    rationale=(
                        f"suspicious_eye_movement co-occurs with phone_usage "
                        f"(within {config.gaze_device_cooccur_ms}ms)"
                    ),
                )
            )
    return out


def detect_gaze_then_answer_commit(ctx: DetectCtx) -> list[RiskContribution]:
    timeline, config, baseline = ctx.timeline, ctx.config, ctx.baseline
    gazes = perception_episodes_by_kind(timeline, "suspicious_eye_movement")
    answers = [
        e
        for e in timeline
        if e.kind
        in {MachineFactKind.MCQ_ANSWER_SELECTED.value, MachineFactKind.MCQ_ANSWER_CHANGED.value}
    ]
    blocking = {
        MachineFactKind.LARGE_PASTE.value,
        MachineFactKind.TYPING_STARTED.value,
        MachineFactKind.ACTIVITY_GAP.value,
    }
    out: list[RiskContribution] = []
    used_gaze: set[str] = set()
    for gaze in gazes:
        if gaze.evidence_ref in used_gaze:
            continue
        gaze_end = gaze.end_ms or gaze.t_ms
        best = None
        for ans in answers:
            if ans.t_ms < gaze_end:
                continue
            delta = ans.t_ms - gaze_end
            if delta > config.gaze_to_commit_ms:
                continue
            intervening = any(
                e.t_ms > gaze_end
                and e.t_ms < ans.t_ms
                and (
                    e.kind in blocking
                    or (e.source == "perception_episode" and e.kind != "suspicious_eye_movement")
                )
                for e in timeline
            )
            if intervening:
                continue
            if best is None or delta < best[1]:
                best = (ans, delta)
        if best is None:
            continue
        ans, delta = best
        used_gaze.add(gaze.evidence_ref)
        stats = section_score_time_stats(baseline, ans.section_title)
        intensity = max(0.35, 1 - delta / config.gaze_to_commit_ms)
        if stats.get("score_z") is not None and stats["score_z"] >= 1:
            intensity = min(1.0, intensity + 0.1)
        if stats.get("time_z") is not None and stats["time_z"] <= -1:
            intensity = min(1.0, intensity + 0.1)
        repeats = sum(
            1
            for g in gazes
            if g.evidence_ref != gaze.evidence_ref
            and any(
                a.t_ms >= (g.end_ms or g.t_ms) and a.t_ms - (g.end_ms or g.t_ms) <= config.gaze_to_commit_ms
                for a in answers
            )
        )
        if repeats >= 1:
            intensity = min(1.0, intensity + 0.1)
        cohort_refs = [r for r in (stats.get("score_ref"), stats.get("time_ref")) if r]
        out.append(
            make_contribution(
                "gaze_then_answer_commit",
                tier=2,
                intensity=intensity,
                requires_corroboration=True,
                timing_basis="section_clock",
                section_id=ans.section_id,
                section_title=ans.section_title,
                time_range_ms=(gaze.t_ms, ans.t_ms),
                evidence_refs=[gaze.evidence_ref, ans.evidence_ref],
                cohort_ref_ids=cohort_refs or None,
                rationale=(
                    f"suspicious_eye_movement ended then {ans.kind} {delta}ms later "
                    "(no intervening activity)"
                ),
            )
        )
    return out


def detect_speech_then_correct_hard(ctx: DetectCtx) -> list[RiskContribution]:
    timeline, config = ctx.timeline, ctx.config
    speeches = speech_episodes(timeline, SPEECH_INTEGRITY_EVENT_TYPES)
    if not speeches:
        return []
    hard_correct = [
        q
        for q in build_fast_correct_questions(ctx)
        if q.correct
        and (
            difficulty_by_question(ctx.facts).get(q.question_id or f"n:{q.question_number}", {}).get(
                "difficulty"
            )
            == "HARD"
            or is_hard_question(ctx, q)
        )
    ]
    out: list[RiskContribution] = []
    for speech in speeches:
        speech_end = speech.end_ms or speech.t_ms
        for q in hard_correct:
            delta = q.window_ms[0] - speech_end
            if delta < -config.video_overlap_pad_ms or delta > config.speech_then_correct_ms:
                continue
            out.append(
                make_contribution(
                    "speech_then_correct_hard",
                    tier=1,
                    intensity=min(1.0, 0.5 + (1 - max(0, delta) / config.speech_then_correct_ms) * 0.5),
                    timing_basis="sqb_submission_window",
                    section_id=q.section_id,
                    question_number=q.question_number,
                    question_id=q.question_id,
                    time_range_ms=(speech.t_ms, q.window_ms[1]),
                    evidence_refs=[speech.evidence_ref, *q.evidence_refs][:4],
                    rationale=(
                        f"{speech.kind} then correct HARD Q{q.question_number} within {max(0, delta)}ms."
                        + speech_summary_from_event(speech)
                    ),
                )
            )
    return out


def detect_whisper_then_paste(ctx: DetectCtx) -> list[RiskContribution]:
    timeline, config = ctx.timeline, ctx.config
    speeches = speech_episodes(timeline, WHISPER_DICTATION_TYPES)
    pastes = [
        e
        for e in timeline
        if e.kind in {MachineFactKind.LARGE_PASTE.value, MachineFactKind.PASTE.value}
        and e.detail.get("pasteOrigin") != "internal"
        and e.detail.get("revertedEdit") is not True
    ]
    out: list[RiskContribution] = []
    for speech in speeches:
        speech_end = speech.end_ms or speech.t_ms
        for paste in pastes:
            delta = paste.t_ms - speech_end
            if delta < 0 or delta > config.whisper_then_paste_ms:
                continue
            out.append(
                make_contribution(
                    "whisper_then_paste",
                    tier=1,
                    intensity=min(1.0, 0.55 + (1 - delta / config.whisper_then_paste_ms) * 0.45),
                    timing_basis="section_clock",
                    section_id=paste.section_id,
                    question_number=paste.question_number,
                    question_id=paste.question_id,
                    time_range_ms=(speech.t_ms, paste.t_ms),
                    evidence_refs=[speech.evidence_ref, paste.evidence_ref],
                    rationale=f"{speech.kind} then {paste.kind} {delta}ms later." + speech_summary_from_event(speech),
                )
            )
    return out


def detect_av_mismatch_then_input(ctx: DetectCtx) -> list[RiskContribution]:
    timeline, config = ctx.timeline, ctx.config
    speeches = speech_episodes(timeline, AV_MISMATCH_TYPES)
    inputs = [
        e
        for e in timeline
        if e.kind in INPUT_AFTER_GAZE_KINDS
        and not (
            e.kind == MachineFactKind.LARGE_PASTE.value
            and (e.detail.get("pasteOrigin") == "internal" or e.detail.get("revertedEdit") is True)
        )
    ]
    out: list[RiskContribution] = []
    for speech in speeches:
        speech_end = speech.end_ms or speech.t_ms
        for inp in inputs:
            delta = inp.t_ms - speech_end
            if delta < 0 or delta > config.av_mismatch_then_input_ms:
                continue
            out.append(
                make_contribution(
                    "av_mismatch_then_input",
                    tier=1,
                    intensity=min(1.0, 0.5 + (1 - delta / config.av_mismatch_then_input_ms) * 0.5),
                    timing_basis="section_clock",
                    section_id=inp.section_id,
                    question_number=inp.question_number,
                    question_id=inp.question_id,
                    time_range_ms=(speech.t_ms, inp.t_ms),
                    evidence_refs=[speech.evidence_ref, inp.evidence_ref],
                    rationale=f"av_speech_without_lips then {inp.kind} {delta}ms later." + speech_summary_from_event(speech),
                )
            )
    return out


def detect_gaze_off_then_speech_then_answer(ctx: DetectCtx) -> list[RiskContribution]:
    timeline, config = ctx.timeline, ctx.config
    gazes = perception_episodes_by_kind(timeline, "suspicious_eye_movement")
    speeches = speech_episodes(timeline, SPEECH_INTEGRITY_EVENT_TYPES)
    answers = [
        e
        for e in timeline
        if e.kind
        in {
            MachineFactKind.CODE_SUBMISSION.value,
            MachineFactKind.MCQ_ANSWER_SELECTED.value,
            MachineFactKind.LARGE_PASTE.value,
        }
        and (
            e.kind != MachineFactKind.LARGE_PASTE.value
            or (e.detail.get("pasteOrigin") != "internal" and e.detail.get("revertedEdit") is not True)
        )
    ]
    out: list[RiskContribution] = []
    for gaze in gazes:
        gaze_end = gaze.end_ms or gaze.t_ms
        for speech in speeches:
            speech_end = speech.end_ms or speech.t_ms
            if speech.t_ms < gaze.t_ms - config.gaze_device_cooccur_ms:
                continue
            if speech.t_ms > gaze_end + config.gaze_off_then_input_ms:
                continue
            for ans in answers:
                delta = ans.t_ms - speech_end
                if delta < 0 or delta > config.speech_then_correct_ms:
                    continue
                out.append(
                    make_contribution(
                        "gaze_off_then_speech_then_answer",
                        tier=1,
                        intensity=0.75,
                        timing_basis="section_clock",
                        section_id=ans.section_id,
                        question_number=ans.question_number,
                        question_id=ans.question_id,
                        time_range_ms=(gaze.t_ms, ans.t_ms),
                        evidence_refs=[gaze.evidence_ref, speech.evidence_ref, ans.evidence_ref],
                        rationale=f"gaze_off → {speech.kind} → {ans.kind} chain." + speech_summary_from_event(speech),
                    )
                )
    return out


def detect_blur_fast_correct(ctx: DetectCtx) -> list[RiskContribution]:
    timeline, config = ctx.timeline, ctx.config
    blurs = [e for e in timeline if e.kind in BLUR_KINDS]
    fast_correct = [q for q in build_fast_correct_questions(ctx) if q.fast and q.correct]
    out: list[RiskContribution] = []
    for blur in blurs:
        blur_end = blur.end_ms or blur.t_ms
        for q in fast_correct:
            delta_to_window = q.window_ms[0] - blur_end
            delta_to_submit = q.window_ms[1] - blur_end
            if delta_to_submit < 0 or delta_to_submit > config.blur_to_answer_ms:
                continue
            if delta_to_window > config.blur_to_answer_ms:
                continue
            if has_external_paste_in_window(timeline, q.window_ms, q.question_number):
                pastes = [
                    p
                    for p in cluster_large_pastes(
                        [e for e in timeline if e.kind == MachineFactKind.LARGE_PASTE.value]
                    )
                    if p.t_ms >= q.window_ms[0]
                    and p.t_ms <= q.window_ms[1]
                    and (p.question_number is None or p.question_number == q.question_number)
                ]
                paste = pastes[0] if pastes else None
                if paste is None:
                    continue
                spent = q.time_spent_seconds or round((q.window_ms[1] - q.window_ms[0]) / 1000)
                out.append(
                    make_contribution(
                        "paste_after_blur",
                        tier=1,
                        intensity=0.95,
                        timing_basis="sqb_submission_window",
                        section_id=q.section_id,
                        question_number=q.question_number,
                        question_id=q.question_id,
                        time_range_ms=(
                            min(blur.t_ms, paste.t_ms, q.window_ms[0]),
                            max(blur_end, paste.end_ms, q.window_ms[1]),
                        ),
                        evidence_refs=[blur.evidence_ref, *paste.evidence_refs, *q.evidence_refs][:5],
                        rationale=external_paste_after_leave_rationale(
                            f"{blur.kind} → external LARGE_PASTE → correct Q{q.question_number} in ~{spent}s",
                            q,
                        ),
                    )
                )
                continue
            spent = q.time_spent_seconds or round((q.window_ms[1] - q.window_ms[0]) / 1000)
            out.append(
                make_contribution(
                    "blur_fast_correct",
                    tier=1,
                    intensity=min(1.0, 0.55 + (1 - spent / config.fast_correct_max_seconds) * 0.35),
                    requires_corroboration=True,
                    timing_basis="sqb_submission_window",
                    section_id=q.section_id,
                    question_number=q.question_number,
                    question_id=q.question_id,
                    time_range_ms=(min(blur.t_ms, q.window_ms[0]), max(blur_end, q.window_ms[1])),
                    evidence_refs=[blur.evidence_ref, *q.evidence_refs][:4],
                    rationale=(
                        f"{blur.kind} → correct Q{q.question_number} in ~{spent}s (speed={q.speed_basis}) "
                        "with no external paste"
                    ),
                )
            )
    return out


def detect_camera_absent_fast_correct(ctx: DetectCtx) -> list[RiskContribution]:
    timeline, config = ctx.timeline, ctx.config
    absences = [
        e for e in timeline if e.source == "perception_episode" and e.kind in CAMERA_ABSENT_TYPES
    ]
    fast_correct = [q for q in build_fast_correct_questions(ctx) if q.fast and q.correct]
    out: list[RiskContribution] = []
    for absence in absences:
        abs_end = absence.end_ms or absence.t_ms
        for q in fast_correct:
            if not intervals_overlap(
                absence.t_ms, abs_end, q.window_ms[0], q.window_ms[1], config.video_overlap_pad_ms
            ):
                continue
            if has_external_paste_in_window(timeline, q.window_ms, q.question_number):
                pastes = [
                    p
                    for p in cluster_large_pastes(
                        [e for e in timeline if e.kind == MachineFactKind.LARGE_PASTE.value]
                    )
                    if p.t_ms >= q.window_ms[0]
                    and p.t_ms <= q.window_ms[1]
                    and (p.question_number is None or p.question_number == q.question_number)
                ]
                paste = pastes[0] if pastes else None
                if paste is None:
                    continue
                spent = q.time_spent_seconds or round((q.window_ms[1] - q.window_ms[0]) / 1000)
                out.append(
                    make_contribution(
                        "paste_after_blur",
                        tier=1,
                        intensity=0.9,
                        timing_basis="sqb_submission_window",
                        section_id=q.section_id,
                        question_number=q.question_number,
                        question_id=q.question_id,
                        time_range_ms=(
                            min(absence.t_ms, paste.t_ms, q.window_ms[0]),
                            max(abs_end, paste.end_ms, q.window_ms[1]),
                        ),
                        evidence_refs=[absence.evidence_ref, *paste.evidence_refs, *q.evidence_refs][:5],
                        rationale=external_paste_after_leave_rationale(
                            f"{absence.kind} co-occurs with external paste then correct Q{q.question_number} (~{spent}s)",
                            q,
                        ),
                    )
                )
                continue
            spent = q.time_spent_seconds or round((q.window_ms[1] - q.window_ms[0]) / 1000)
            out.append(
                make_contribution(
                    "camera_absent_fast_correct",
                    tier=1,
                    intensity=min(1.0, 0.55 + (1 - spent / config.fast_correct_max_seconds) * 0.35),
                    requires_corroboration=True,
                    timing_basis="sqb_submission_window",
                    section_id=q.section_id,
                    question_number=q.question_number,
                    question_id=q.question_id,
                    time_range_ms=(min(absence.t_ms, q.window_ms[0]), max(abs_end, q.window_ms[1])),
                    evidence_refs=[absence.evidence_ref, *q.evidence_refs][:4],
                    rationale=(
                        f"{absence.kind} during fast+correct Q{q.question_number} (~{spent}s, speed={q.speed_basis}) "
                        "with no external paste"
                    ),
                )
            )
    return out


def detect_out_of_order_submission_hop(ctx: DetectCtx) -> list[RiskContribution]:
    boundaries, config, facts = ctx.boundaries, ctx.config, ctx.facts
    by_section: dict[str, list] = {}
    for boundary in boundaries:
        if boundary.timing_unavailable:
            continue
        by_section.setdefault(boundary.section_id, []).append(boundary)
    out: list[RiskContribution] = []
    for section_id, items in by_section.items():
        ordered = sorted(items, key=lambda b: b.end_offset_ms)
        if len(ordered) < 3:
            continue
        reversals = sum(
            1 for i in range(1, len(ordered)) if ordered[i].question_number < ordered[i - 1].question_number
        )
        if reversals < config.hop_min_reversals:
            continue
        seq = "→".join(str(b.question_number) for b in ordered)
        sub_facts = [
            f
            for f in facts
            if f.kind == MachineFactKind.CODE_SUBMISSION.value and f.detail.get("sectionId") == section_id
        ]
        out.append(
            make_contribution(
                "out_of_order_submission_hop",
                tier=2,
                intensity=min(1.0, 0.4 + reversals * 0.15),
                requires_corroboration=True,
                timing_basis="sqb_submission_window",
                section_id=section_id,
                time_range_ms=(ordered[0].start_offset_ms, ordered[-1].end_offset_ms),
                evidence_refs=[f"sqb:s={section_id}:hop", *[f"fact:{f.id}" for f in sub_facts[:3]]],
                rationale=(
                    f"Coding section submission order is erratic ({seq}) with {reversals} reversals"
                ),
            )
        )
    return out


def detect_large_unattributed_gap(ctx: DetectCtx) -> list[RiskContribution]:
    boundaries, timeline, config = ctx.boundaries, ctx.timeline, ctx.config
    by_section: dict[str, list] = {}
    for boundary in boundaries:
        by_section.setdefault(boundary.section_id, []).append(boundary)
    leave_events = [
        e
        for e in timeline
        if e.kind in BLUR_KINDS
        or (e.source == "perception_episode" and e.kind in CAMERA_ABSENT_TYPES)
    ]
    out: list[RiskContribution] = []
    for section_id, items in by_section.items():
        verified = [b for b in items if not b.timing_unavailable and not b.end_is_fallback]
        if not verified:
            continue
        attributed_sec = sum(
            b.authoritative_time_spent_seconds
            if b.authoritative_time_spent_seconds
            else max(0, (b.end_offset_ms - b.start_offset_ms) / 1000)
            for b in verified
        )
        section_start = min(b.start_offset_ms for b in items)
        section_end = max(b.end_offset_ms for b in items)
        wall_sec = max(0, (section_end - section_start) / 1000)
        unattributed = max(0, wall_sec - attributed_sec)
        if unattributed < config.unattributed_gap_min_seconds:
            continue
        nearby = next(
            (
                e
                for e in leave_events
                if e.t_ms >= section_start - 30_000 and (e.end_ms or e.t_ms) <= section_end + 30_000
            ),
            None,
        )
        if nearby is None:
            continue
        out.append(
            make_contribution(
                "large_unattributed_gap",
                tier=2,
                intensity=min(1.0, unattributed / (config.unattributed_gap_min_seconds * 3)),
                requires_corroboration=True,
                timing_basis="section_clock",
                section_id=section_id,
                time_range_ms=(section_start, section_end),
                evidence_refs=[nearby.evidence_ref, f"sqb:s={section_id}:gap"],
                rationale=(
                    f"~{round(unattributed)}s unattributed time in section with nearby {nearby.kind}"
                ),
            )
        )
    return out


def detect_face_absent_then_paste(ctx: DetectCtx) -> list[RiskContribution]:
    timeline, config = ctx.timeline, ctx.config
    absences = [
        e for e in timeline if e.source == "perception_episode" and e.kind in CAMERA_ABSENT_TYPES
    ]
    pastes = cluster_large_pastes([e for e in timeline if e.kind == MachineFactKind.LARGE_PASTE.value])
    out: list[RiskContribution] = []
    for absence in absences:
        absence_end = absence.end_ms or absence.t_ms
        for paste in pastes:
            delta = paste.t_ms - absence_end
            if delta < 0 or delta > config.paste_after_blur_ms:
                continue
            out.append(
                make_contribution(
                    "face_absent_then_paste",
                    tier=1,
                    intensity=min(1.0, 0.45 + (1 - delta / config.paste_after_blur_ms) * 0.45),
                    requires_corroboration=True,
                    timing_basis="section_clock",
                    section_id=paste.section_id or absence.section_id,
                    question_number=paste.question_number,
                    question_id=paste.question_id,
                    time_range_ms=(absence.t_ms, paste.end_ms),
                    evidence_refs=[absence.evidence_ref, *paste.evidence_refs][:4],
                    rationale=(
                        f"{absence.kind} then external LARGE_PASTE {delta}ms later "
                        f"(~{paste.total_chars_added} chars)"
                    ),
                )
            )
    return out


def detect_external_resource_then_paste(ctx: DetectCtx) -> list[RiskContribution]:
    timeline, config = ctx.timeline, ctx.config
    resources = [
        e
        for e in timeline
        if (e.source == "screen_episode" and e.kind == "external_resource_open")
        or e.kind
        in {MachineFactKind.SCREEN_EXTERNAL_RESOURCE.value, MachineFactKind.SCREEN_AI_ASSISTANT_UI.value}
    ]
    paste_clusters = cluster_large_pastes(
        [e for e in timeline if e.kind == MachineFactKind.LARGE_PASTE.value]
    )
    screen_pastes = [
        {
            "t_ms": e.t_ms,
            "end_ms": e.end_ms or e.t_ms,
            "total_chars_added": (
                e.detail.get("charsAdded")
                if isinstance(e.detail.get("charsAdded"), int)
                else len(str(e.detail.get("pastedExcerpt") or "")) or 40
            ),
            "section_id": e.section_id,
            "question_number": e.question_number or e.detail.get("questionNumber"),
            "question_id": e.question_id,
            "evidence_refs": [e.evidence_ref],
            "end_ms_val": e.end_ms or e.t_ms,
        }
        for e in timeline
        if e.kind == MachineFactKind.SCREEN_EXTERNAL_PASTE.value
        or (e.source == "screen_episode" and e.kind == "screen_external_paste")
    ]
    out: list[RiskContribution] = []
    for resource in resources:
        resource_end = resource.end_ms or resource.t_ms
        for paste in [*paste_clusters, *screen_pastes]:
            t_ms = paste.t_ms if hasattr(paste, "t_ms") else paste["t_ms"]
            end_ms = paste.end_ms if hasattr(paste, "end_ms") else paste["end_ms_val"]
            delta = t_ms - resource.t_ms
            if t_ms < resource.t_ms - 5_000:
                continue
            if delta > config.paste_after_blur_ms and t_ms > resource_end + config.paste_after_blur_ms:
                continue
            chars = paste.total_chars_added if hasattr(paste, "total_chars_added") else paste["total_chars_added"]
            refs = paste.evidence_refs if hasattr(paste, "evidence_refs") else paste["evidence_refs"]
            out.append(
                make_contribution(
                    "external_resource_then_paste",
                    tier=1,
                    intensity=min(1.0, 0.5 + min(0.4, chars / 200)),
                    timing_basis="section_clock",
                    section_id=getattr(paste, "section_id", None) or paste.get("section_id") or resource.section_id,
                    question_number=getattr(paste, "question_number", None) or paste.get("question_number"),
                    question_id=getattr(paste, "question_id", None) or paste.get("question_id"),
                    time_range_ms=(min(resource.t_ms, t_ms), max(resource_end, end_ms)),
                    evidence_refs=[resource.evidence_ref, *refs][:4],
                    rationale=f"External resource/app on screen near paste (~{chars} chars)",
                )
            )
    return out


def detect_face_absent_during_input(ctx: DetectCtx) -> list[RiskContribution]:
    timeline, config = ctx.timeline, ctx.config
    absences = [
        e for e in timeline if e.source == "perception_episode" and e.kind in CAMERA_ABSENT_TYPES
    ]
    inputs = [
        e
        for e in timeline
        if e.kind
        in {
            MachineFactKind.SCREEN_EXTERNAL_PASTE.value,
            MachineFactKind.LARGE_PASTE.value,
        }
        or (
            e.source == "screen_episode"
            and e.kind in {"screen_external_paste", "external_resource_open"}
        )
    ]
    out: list[RiskContribution] = []
    for absence in absences:
        absence_end = absence.end_ms or absence.t_ms
        for inp in inputs:
            input_end = inp.end_ms or inp.t_ms
            if not (
                inp.t_ms <= absence_end + config.video_overlap_pad_ms
                and input_end >= absence.t_ms - config.video_overlap_pad_ms
            ):
                continue
            qn = inp.question_number or inp.detail.get("questionNumber")
            out.append(
                make_contribution(
                    "face_absent_during_input",
                    tier=1,
                    intensity=0.7,
                    requires_corroboration=True,
                    timing_basis="section_clock",
                    section_id=inp.section_id or absence.section_id,
                    question_number=qn if isinstance(qn, int) else None,
                    question_id=inp.question_id,
                    time_range_ms=(min(absence.t_ms, inp.t_ms), max(absence_end, input_end)),
                    evidence_refs=[absence.evidence_ref, inp.evidence_ref],
                    rationale=f"{absence.kind} overlapping screen/input activity ({inp.kind})",
                )
            )
    return out


def detect_gaze_then_correct_mcq_burst(ctx: DetectCtx) -> list[RiskContribution]:
    timeline, config = ctx.timeline, ctx.config
    gazes = perception_episodes_by_kind(timeline, "suspicious_eye_movement")
    bursts = build_fast_mcq_answer_bursts(ctx)
    out: list[RiskContribution] = []
    for gaze in gazes:
        gaze_end = gaze.end_ms or gaze.t_ms
        for burst in bursts:
            if not intervals_overlap(
                gaze.t_ms, gaze_end, burst.start_ms, burst.end_ms, config.video_overlap_pad_ms
            ):
                continue
            out.append(
                make_contribution(
                    "gaze_then_correct_mcq_burst",
                    tier=2,
                    intensity=min(
                        1.0,
                        0.4
                        + min(0.3, burst.answer_count * 0.05)
                        + min(0.2, (config.mcq_fast_gap_ms - burst.median_gap_ms) / config.mcq_fast_gap_ms)
                        * 0.2,
                    ),
                    requires_corroboration=True,
                    timing_basis="section_clock",
                    section_id=burst.section_id,
                    section_title=burst.section_title,
                    time_range_ms=(min(gaze.t_ms, burst.start_ms), max(gaze_end, burst.end_ms)),
                    evidence_refs=[gaze.evidence_ref, *burst.evidence_refs][:4],
                    rationale=(
                        f"suspicious_eye_movement overlapping rapid MCQ burst "
                        f"({burst.answer_count} selects, median gap {burst.median_gap_ms}ms)"
                    ),
                )
            )
    return out


def detect_speech_coaching_then_fast_correct(ctx: DetectCtx) -> list[RiskContribution]:
    timeline, config = ctx.timeline, ctx.config
    speeches = speech_episodes(timeline, SPEECH_INTEGRITY_EVENT_TYPES)
    fast_correct = [q for q in build_fast_correct_questions(ctx) if q.fast and q.correct]
    out: list[RiskContribution] = []
    for speech in speeches:
        speech_end = speech.end_ms or speech.t_ms
        for q in fast_correct:
            delta = q.window_ms[0] - speech_end
            if delta < -config.video_overlap_pad_ms or delta > config.speech_then_correct_ms:
                continue
            if not intervals_overlap(
                speech.t_ms,
                speech_end,
                q.window_ms[0] - config.speech_then_correct_ms,
                q.window_ms[1],
                config.video_overlap_pad_ms,
            ):
                continue
            spent = q.time_spent_seconds or round((q.window_ms[1] - q.window_ms[0]) / 1000)
            out.append(
                make_contribution(
                    "speech_coaching_then_fast_correct",
                    tier=1,
                    intensity=min(1.0, 0.5 + (1 - spent / config.fast_correct_max_seconds) * 0.4),
                    requires_corroboration=True,
                    timing_basis="sqb_submission_window",
                    section_id=q.section_id,
                    question_number=q.question_number,
                    question_id=q.question_id,
                    time_range_ms=(speech.t_ms, q.window_ms[1]),
                    evidence_refs=[speech.evidence_ref, *q.evidence_refs][:4],
                    rationale=f"{speech.kind} then fast+correct Q{q.question_number} (~{spent}s)."
                    + speech_summary_from_event(speech),
                )
            )
    return out


def detect_second_person_then_paste_or_correct(ctx: DetectCtx) -> list[RiskContribution]:
    timeline, config = ctx.timeline, ctx.config
    helpers = [
        *perception_episodes_by_kind(timeline, "external_help"),
        *perception_episodes_by_kind(timeline, "multiple_faces"),
    ]
    pastes = cluster_large_pastes([e for e in timeline if e.kind == MachineFactKind.LARGE_PASTE.value])
    fast_correct = [q for q in build_fast_correct_questions(ctx) if q.fast and q.correct]
    out: list[RiskContribution] = []
    for help_ep in helpers:
        help_end = help_ep.end_ms or help_ep.t_ms
        for paste in pastes:
            delta = paste.t_ms - help_end
            if delta < 0 or delta > config.paste_after_blur_ms:
                continue
            out.append(
                make_contribution(
                    "second_person_then_paste_or_correct",
                    tier=1,
                    intensity=min(1.0, 0.5 + (1 - delta / config.paste_after_blur_ms) * 0.4),
                    requires_corroboration=True,
                    timing_basis="section_clock",
                    section_id=paste.section_id or help_ep.section_id,
                    question_number=paste.question_number,
                    question_id=paste.question_id,
                    time_range_ms=(help_ep.t_ms, paste.end_ms),
                    evidence_refs=[help_ep.evidence_ref, *paste.evidence_refs][:4],
                    rationale=f"{help_ep.kind} then external LARGE_PASTE {delta}ms later",
                )
            )
        for q in fast_correct:
            if not intervals_overlap(
                help_ep.t_ms, help_end, q.window_ms[0], q.window_ms[1], config.video_overlap_pad_ms
            ):
                continue
            spent = q.time_spent_seconds or round((q.window_ms[1] - q.window_ms[0]) / 1000)
            out.append(
                make_contribution(
                    "second_person_then_paste_or_correct",
                    tier=1,
                    intensity=min(1.0, 0.45 + (1 - spent / config.fast_correct_max_seconds) * 0.4),
                    requires_corroboration=True,
                    timing_basis="sqb_submission_window",
                    section_id=q.section_id,
                    question_number=q.question_number,
                    question_id=q.question_id,
                    time_range_ms=(help_ep.t_ms, q.window_ms[1]),
                    evidence_refs=[help_ep.evidence_ref, *q.evidence_refs][:4],
                    rationale=f"{help_ep.kind} overlapping fast+correct Q{q.question_number} (~{spent}s)",
                )
            )
    return out


def detect_idle_gap_then_fast_correct(ctx: DetectCtx) -> list[RiskContribution]:
    timeline, config = ctx.timeline, ctx.config
    gaps = [e for e in timeline if e.kind == MachineFactKind.ACTIVITY_GAP.value]
    fast_correct = [q for q in build_fast_correct_questions(ctx) if q.fast and q.correct]
    pastes = cluster_large_pastes([e for e in timeline if e.kind == MachineFactKind.LARGE_PASTE.value])
    out: list[RiskContribution] = []
    for gap in gaps:
        dur = gap_duration_ms(gap)
        if dur < config.idle_gap_min_ms:
            continue
        gap_end = gap.end_ms or gap.t_ms
        for q in fast_correct:
            delta = q.window_ms[0] - gap_end
            if delta < 0 or delta > config.idle_then_paste_ms:
                continue
            if any(
                p.t_ms >= q.window_ms[0]
                and p.t_ms <= q.window_ms[1]
                and (p.question_number is None or p.question_number == q.question_number)
                for p in pastes
            ):
                continue
            spent = q.time_spent_seconds or round((q.window_ms[1] - q.window_ms[0]) / 1000)
            out.append(
                make_contribution(
                    "idle_gap_then_fast_correct",
                    tier=1,
                    intensity=min(1.0, 0.45 + (1 - spent / config.fast_correct_max_seconds) * 0.4),
                    requires_corroboration=True,
                    timing_basis="sqb_submission_window",
                    section_id=q.section_id,
                    question_number=q.question_number,
                    question_id=q.question_id,
                    time_range_ms=(gap.t_ms, q.window_ms[1]),
                    evidence_refs=[gap.evidence_ref, *q.evidence_refs][:4],
                    rationale=(
                        f"ACTIVITY_GAP {dur}ms then fast+correct Q{q.question_number} (~{spent}s) "
                        "with no external paste"
                    ),
                )
            )
    return out


def detect_deterministic_paste_workflow(ctx: DetectCtx) -> list[RiskContribution]:
    timeline, config = ctx.timeline, ctx.config
    pastes = cluster_large_pastes([e for e in timeline if e.kind == MachineFactKind.LARGE_PASTE.value])
    corrections = [e for e in timeline if e.kind == MachineFactKind.TEXT_CORRECTION.value]
    by_q: dict[int, list] = {}
    for paste in pastes:
        if isinstance(paste.question_number, int):
            by_q.setdefault(paste.question_number, []).append(paste)
    out: list[RiskContribution] = []
    for qn, items in by_q.items():
        if len(items) < 2:
            continue
        window_start = min(p.t_ms for p in items)
        window_end = max(p.end_ms for p in items)
        nearby = [
            c
            for c in corrections
            if window_start - 5_000 <= c.t_ms <= window_end + 15_000
            and (c.question_number is None or c.question_number == qn)
        ]
        if not nearby:
            continue
        total_chars = sum(p.total_chars_added for p in items)
        out.append(
            make_contribution(
                "deterministic_paste_workflow",
                tier=1,
                intensity=min(1.0, 0.45 + len(items) * 0.12 + min(0.25, total_chars / 2000)),
                requires_corroboration=len(nearby) < 2,
                timing_basis="sqb_submission_window",
                section_id=items[0].section_id,
                question_number=qn,
                question_id=items[0].question_id,
                time_range_ms=(window_start, window_end),
                evidence_refs=[
                    *sum((p.evidence_refs[:1] for p in items[:3]), []),
                    *[c.evidence_ref for c in nearby[:2]],
                ][:4],
                rationale=(
                    f"{len(items)} external paste cluster(s) (~{total_chars} chars) on Q{qn} "
                    f"with {len(nearby)} bulk text correction(s)"
                ),
            )
        )
    if not out and len(pastes) >= max(2, config.paste_cluster_min_count):
        window_start = min(p.t_ms for p in pastes)
        window_end = max(p.end_ms for p in pastes)
        nearby = [c for c in corrections if window_start - 5_000 <= c.t_ms <= window_end + 15_000]
        if len(nearby) >= 2:
            total_chars = sum(p.total_chars_added for p in pastes)
            out.append(
                make_contribution(
                    "deterministic_paste_workflow",
                    tier=1,
                    intensity=min(1.0, 0.4 + len(pastes) * 0.08),
                    requires_corroboration=True,
                    timing_basis="section_clock",
                    time_range_ms=(window_start, window_end),
                    evidence_refs=[
                        *sum((p.evidence_refs[:1] for p in pastes[:3]), []),
                        *[c.evidence_ref for c in nearby[:2]],
                    ][:4],
                    rationale=(
                        f"{len(pastes)} external paste cluster(s) (~{total_chars} chars) "
                        f"with {len(nearby)} bulk corrections"
                    ),
                )
            )
    return out


def detect_interact_speech_input_correct(ctx: DetectCtx) -> list[RiskContribution]:
    timeline, config = ctx.timeline, ctx.config
    interact_kinds = ["external_help", "multiple_faces", *SPEECH_INTEGRITY_EVENT_TYPES]
    interacts: list = []
    for kind in interact_kinds:
        interacts.extend(perception_episodes_by_kind(timeline, kind))
    inputs = [
        e
        for e in timeline
        if e.kind
        in {
            MachineFactKind.LARGE_PASTE.value,
            MachineFactKind.PASTE.value,
            MachineFactKind.TYPING_STARTED.value,
            MachineFactKind.MCQ_ANSWER_SELECTED.value,
        }
    ]
    fast_correct = [q for q in build_fast_correct_questions(ctx) if q.fast and q.correct]
    out: list[RiskContribution] = []
    for q in fast_correct:
        q_inputs = [
            e
            for e in inputs
            if q.window_ms[0] - 5_000 <= e.t_ms <= q.window_ms[1] + 5_000
            and (e.question_number is None or e.question_number == q.question_number)
        ]
        if not q_inputs:
            continue
        for interact in interacts:
            interact_end = interact.end_ms or interact.t_ms
            if interact_end < q.window_ms[0] - config.speech_then_correct_ms:
                continue
            if interact.t_ms > q.window_ms[1] + 5_000:
                continue
            first_input = min(q_inputs, key=lambda e: e.t_ms)
            if first_input.t_ms + 2_000 < interact.t_ms:
                continue
            spent = q.time_spent_seconds or round((q.window_ms[1] - q.window_ms[0]) / 1000)
            speech_class = interact.detail.get("speechContentClass")
            out.append(
                make_contribution(
                    "interact_speech_input_correct",
                    tier=1,
                    intensity=min(1.0, 0.5 + (1 - spent / config.fast_correct_max_seconds) * 0.35),
                    requires_corroboration=True,
                    timing_basis="sqb_submission_window",
                    section_id=q.section_id,
                    question_number=q.question_number,
                    question_id=q.question_id,
                    time_range_ms=(interact.t_ms, q.window_ms[1]),
                    evidence_refs=[interact.evidence_ref, first_input.evidence_ref, *q.evidence_refs][:4],
                    rationale=(
                        f"{interact.kind}"
                        + (f" ({speech_class})" if speech_class else "")
                        + f" → input on Q{q.question_number} → correct (~{spent}s)"
                    ),
                )
            )
    return out


def _cooccurrence(
    left: list[TimelineEvent],
    right: list[TimelineEvent],
    pad_ms: int,
) -> list[tuple[TimelineEvent, TimelineEvent]]:
    """Every (left, right) pair whose episodes overlap within ``pad_ms``.

    Each pair is yielded once; a long episode on one side legitimately pairs
    with several on the other, which is what makes a sustained co-occurrence
    score higher than a single brush.
    """

    pairs: list[tuple[TimelineEvent, TimelineEvent]] = []
    seen: set[str] = set()
    for a in left:
        for b in right:
            key = f"{a.evidence_ref}|{b.evidence_ref}"
            if key in seen:
                continue
            if not intervals_overlap(
                a.t_ms, a.end_ms or a.t_ms, b.t_ms, b.end_ms or b.t_ms, pad_ms
            ):
                continue
            seen.add(key)
            pairs.append((a, b))
    return pairs


def detect_gaze_second_person_interacting(ctx: DetectCtx) -> list[RiskContribution]:
    """Off-screen gaze while a second person is interacting.

    Either alone is weak — candidates look away to think, and people walk
    behind them. Together the gaze has somewhere to have gone.
    """

    timeline, config = ctx.timeline, ctx.config
    gazes = perception_episodes_by_kinds(timeline, GAZE_EVENT_TYPES)
    others = perception_episodes_by_kinds(timeline, SECOND_PERSON_EVENT_TYPES)
    out: list[RiskContribution] = []
    for gaze, other in _cooccurrence(gazes, others, config.second_person_cooccur_ms):
        out.append(
            make_contribution(
                "gaze_second_person_interacting",
                tier=1,
                intensity=0.75,
                timing_basis="section_clock",
                section_id=gaze.section_id or other.section_id,
                time_range_ms=(
                    min(gaze.t_ms, other.t_ms),
                    max(gaze.end_ms or gaze.t_ms, other.end_ms or other.t_ms),
                ),
                evidence_refs=[gaze.evidence_ref, other.evidence_ref],
                rationale=(
                    f"{gaze.kind} co-occurs with {other.kind} "
                    f"(within {config.second_person_cooccur_ms}ms)"
                ),
            )
        )
    return out


def detect_second_person_discussing(ctx: DetectCtx) -> list[RiskContribution]:
    """A second person interacting while the speech is about the answers."""

    timeline, config = ctx.timeline, ctx.config
    others = perception_episodes_by_kinds(timeline, SECOND_PERSON_EVENT_TYPES)
    speech = [
        e
        for e in perception_episodes_by_kinds(timeline, ANSWER_SPEECH_EVENT_TYPES)
        if is_answer_speech(e)
    ]
    out: list[RiskContribution] = []
    for other, said in _cooccurrence(others, speech, config.second_person_cooccur_ms):
        speech_class = said.detail.get("speechContentClass")
        out.append(
            make_contribution(
                "second_person_discussing",
                tier=1,
                # Dictation and a direct request for the answer leave less room
                # for an innocent reading than a general discussion does.
                intensity=0.9
                if speech_class in {"receiving_dictation", "asking_for_answer"}
                else 0.75,
                timing_basis="section_clock",
                section_id=other.section_id or said.section_id,
                time_range_ms=(
                    min(other.t_ms, said.t_ms),
                    max(other.end_ms or other.t_ms, said.end_ms or said.t_ms),
                ),
                evidence_refs=[other.evidence_ref, said.evidence_ref],
                rationale=(
                    f"{other.kind} co-occurs with speech classified "
                    f"{speech_class} (within {config.second_person_cooccur_ms}ms)"
                ),
            )
        )
    return out


def detect_phone_with_answer_speech(ctx: DetectCtx) -> list[RiskContribution]:
    """A phone in view while the answers are being discussed or dictated."""

    timeline, config = ctx.timeline, ctx.config
    phones = perception_episodes_by_kind(timeline, "phone_usage")
    speech = [
        e
        for e in perception_episodes_by_kinds(timeline, ANSWER_SPEECH_EVENT_TYPES)
        if is_answer_speech(e)
    ]
    out: list[RiskContribution] = []
    for phone, said in _cooccurrence(phones, speech, config.phone_speech_cooccur_ms):
        speech_class = said.detail.get("speechContentClass")
        out.append(
            make_contribution(
                "phone_with_answer_speech",
                tier=1,
                intensity=0.9 if speech_class == "receiving_dictation" else 0.75,
                timing_basis="section_clock",
                section_id=phone.section_id or said.section_id,
                time_range_ms=(
                    min(phone.t_ms, said.t_ms),
                    max(phone.end_ms or phone.t_ms, said.end_ms or said.t_ms),
                ),
                evidence_refs=[phone.evidence_ref, said.evidence_ref],
                rationale=(
                    f"phone_usage co-occurs with speech classified {speech_class} "
                    f"(within {config.phone_speech_cooccur_ms}ms)"
                ),
            )
        )
    return out


def detect_second_monitor_with_gaze(ctx: DetectCtx) -> list[RiskContribution]:
    """A secondary workspace on screen while the candidate looks off-screen."""

    timeline, config = ctx.timeline, ctx.config
    gazes = perception_episodes_by_kinds(timeline, GAZE_EVENT_TYPES)
    workspaces = [
        e for e in timeline if e.kind == MachineFactKind.SCREEN_SECONDARY_WORKSPACE.value
    ]
    out: list[RiskContribution] = []
    for gaze, workspace in _cooccurrence(gazes, workspaces, config.second_monitor_cooccur_ms):
        out.append(
            make_contribution(
                "second_monitor_with_gaze",
                tier=1,
                intensity=0.8,
                timing_basis="section_clock",
                section_id=gaze.section_id or workspace.section_id,
                time_range_ms=(
                    min(gaze.t_ms, workspace.t_ms),
                    max(gaze.end_ms or gaze.t_ms, workspace.end_ms or workspace.t_ms),
                ),
                evidence_refs=[gaze.evidence_ref, workspace.evidence_ref],
                rationale=(
                    "secondary workspace visible while gaze was off-screen "
                    f"(within {config.second_monitor_cooccur_ms}ms)"
                ),
            )
        )
    return out


def detect_blur_burst(ctx: DetectCtx) -> list[RiskContribution]:
    """Repeated tab switches or focus losses in a short span.

    One switch is noise; a cluster of them is a pattern of leaving the exam,
    and it needs no paste to be worth a reviewer's attention.
    """

    timeline, config = ctx.timeline, ctx.config
    blurs = sorted((e for e in timeline if e.kind in BLUR_KINDS), key=lambda e: e.t_ms)
    out: list[RiskContribution] = []
    used: set[str] = set()
    for i, first in enumerate(blurs):
        window = [
            e
            for e in blurs[i:]
            if e.t_ms - first.t_ms <= config.blur_burst_window_ms
        ]
        if len(window) < config.blur_burst_min_count:
            continue
        key = window[0].evidence_ref
        if key in used:
            continue
        # Each blur belongs to the first burst that claims it, so a long run of
        # switches reports once rather than once per member.
        used.update(e.evidence_ref for e in window)
        span_ms = window[-1].t_ms - first.t_ms
        out.append(
            make_contribution(
                "blur_burst",
                tier=2,
                intensity=min(1.0, 0.4 + 0.1 * (len(window) - config.blur_burst_min_count + 1)),
                requires_corroboration=True,
                timing_basis="section_clock",
                section_id=first.section_id,
                time_range_ms=(first.t_ms, window[-1].end_ms or window[-1].t_ms),
                evidence_refs=[e.evidence_ref for e in window][:4],
                rationale=(
                    f"{len(window)} focus losses within {span_ms}ms "
                    f"(window {config.blur_burst_window_ms}ms)"
                ),
            )
        )
    return out


def detect_fullscreen_exit_then_input(ctx: DetectCtx) -> list[RiskContribution]:
    """Fullscreen lost, then typing or a paste before it is restored."""

    timeline, config = ctx.timeline, ctx.config
    exits = [e for e in timeline if e.kind == MachineFactKind.SCREEN_FULLSCREEN_LOST.value]
    inputs = sorted((e for e in timeline if e.kind in INPUT_AFTER_GAZE_KINDS), key=lambda e: e.t_ms)
    out: list[RiskContribution] = []
    for exited in exits:
        following = next(
            (
                e
                for e in inputs
                if 0 <= e.t_ms - (exited.end_ms or exited.t_ms)
                <= config.fullscreen_lost_to_input_ms
            ),
            None,
        )
        if following is None:
            continue
        delta = following.t_ms - (exited.end_ms or exited.t_ms)
        out.append(
            make_contribution(
                "fullscreen_exit_then_input",
                tier=2,
                intensity=max(0.4, 1 - delta / max(1, config.fullscreen_lost_to_input_ms)),
                requires_corroboration=True,
                timing_basis="section_clock",
                section_id=exited.section_id or following.section_id,
                time_range_ms=(exited.t_ms, following.end_ms or following.t_ms),
                evidence_refs=[exited.evidence_ref, following.evidence_ref],
                rationale=f"fullscreen lost → {following.kind} {delta}ms later",
            )
        )
    return out


def detect_paste_without_prior_copy(ctx: DetectCtx) -> list[RiskContribution]:
    """A paste whose clipboard was never filled by a copy inside the exam.

    Content pasted without a preceding in-exam COPY came from somewhere else by
    construction, which is what makes this stronger than paste volume alone.
    """

    timeline, config = ctx.timeline, ctx.config
    copies = [e for e in timeline if e.kind == MachineFactKind.COPY.value]
    pastes = [
        e
        for e in timeline
        if e.kind in {MachineFactKind.LARGE_PASTE.value, MachineFactKind.PASTE.value}
        and e.detail.get("pasteOrigin") != "internal"
    ]
    out: list[RiskContribution] = []
    for paste in pastes:
        preceded = any(
            0 <= paste.t_ms - c.t_ms <= config.copy_before_paste_ms for c in copies
        )
        if preceded:
            continue
        size = paste_size(paste)
        out.append(
            make_contribution(
                "paste_without_prior_copy",
                tier=1,
                intensity=0.9 if size >= config.min_paste_size_for_tier1 else 0.6,
                timing_basis="section_clock",
                section_id=paste.section_id,
                question_number=paste.question_number,
                question_id=paste.question_id,
                time_range_ms=(paste.t_ms, paste.end_ms or paste.t_ms),
                evidence_refs=[paste.evidence_ref],
                rationale=(
                    f"{paste.kind}"
                    + (f" of {size} chars" if size else "")
                    + " with no in-exam COPY in the preceding "
                    f"{config.copy_before_paste_ms}ms"
                ),
            )
        )
    return out


def detect_all_factors(ctx: DetectCtx) -> list[RiskContribution]:
    out: list[RiskContribution] = []
    out.extend(detect_paste_after_blur(ctx))
    out.extend(detect_idle_then_large_paste(ctx))
    out.extend(detect_video_overlap(ctx))
    out.extend(detect_gaze_off_then_input(ctx))
    out.extend(detect_improbable_hard_fast_paste(ctx))
    out.extend(detect_paste_on_coding_question(ctx))
    out.extend(detect_phone_fast_correct(ctx))
    out.extend(detect_phone_fast_mcq_answers(ctx))
    out.extend(detect_gaze_fast_correct(ctx))
    out.extend(detect_gaze_with_device(ctx))
    out.extend(detect_gaze_then_answer_commit(ctx))
    out.extend(detect_speech_then_correct_hard(ctx))
    out.extend(detect_whisper_then_paste(ctx))
    out.extend(detect_av_mismatch_then_input(ctx))
    out.extend(detect_gaze_off_then_speech_then_answer(ctx))
    out.extend(detect_blur_fast_correct(ctx))
    out.extend(detect_camera_absent_fast_correct(ctx))
    out.extend(detect_out_of_order_submission_hop(ctx))
    out.extend(detect_large_unattributed_gap(ctx))
    out.extend(detect_face_absent_then_paste(ctx))
    out.extend(detect_external_resource_then_paste(ctx))
    out.extend(detect_face_absent_during_input(ctx))
    out.extend(detect_gaze_then_correct_mcq_burst(ctx))
    out.extend(detect_speech_coaching_then_fast_correct(ctx))
    out.extend(detect_second_person_then_paste_or_correct(ctx))
    out.extend(detect_idle_gap_then_fast_correct(ctx))
    out.extend(detect_deterministic_paste_workflow(ctx))
    out.extend(detect_interact_speech_input_correct(ctx))
    out.extend(detect_gaze_second_person_interacting(ctx))
    out.extend(detect_second_person_discussing(ctx))
    out.extend(detect_phone_with_answer_speech(ctx))
    out.extend(detect_second_monitor_with_gaze(ctx))
    out.extend(detect_blur_burst(ctx))
    out.extend(detect_fullscreen_exit_then_input(ctx))
    out.extend(detect_paste_without_prior_copy(ctx))
    return dedupe_contributions(out)
