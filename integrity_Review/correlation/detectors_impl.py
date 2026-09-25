from __future__ import annotations

from ..machine_facts.kinds import MachineFactKind
from .contracts import DetectCtx, RiskContribution, TimelineEvent, make_contribution
from .helpers import (
    SPEECH_INTEGRITY_EVENT_TYPES,
    speech_episodes,
    speech_summary_from_event,
    ANSWER_SPEECH_EVENT_TYPES,
    BLUR_KINDS,
    GAZE_EVENT_TYPES,
    INPUT_AFTER_GAZE_KINDS,
    SECOND_PERSON_EVENT_TYPES,
    is_answer_speech,
    perception_episodes_by_kinds,
    cluster_large_pastes,
    intervals_overlap,
    paste_size,
    perception_episodes_by_kind,
    same_section,
    section_score_time_stats,
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
                    members=[gaze, phone],
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
                members=[gaze, ans],
                cohort_ref_ids=cohort_refs or None,
                rationale=(
                    f"suspicious_eye_movement ended then {ans.kind} {delta}ms later "
                    "(no intervening activity)"
                ),
            )
        )
    return out


def _cooccurrence(
    left: list[TimelineEvent],
    right: list[TimelineEvent],
    pad_ms: int,
) -> list[tuple[TimelineEvent, TimelineEvent]]:
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
                members=[gaze, other],
                rationale=(
                    f"{gaze.kind} co-occurs with {other.kind} "
                    f"(within {config.second_person_cooccur_ms}ms)"
                ),
            )
        )
    return out


def detect_second_person_discussing(ctx: DetectCtx) -> list[RiskContribution]:
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
                members=[other, said],
                rationale=(
                    f"{other.kind} co-occurs with speech classified "
                    f"{speech_class} (within {config.second_person_cooccur_ms}ms)"
                ),
            )
        )
    return out


def detect_phone_with_answer_speech(ctx: DetectCtx) -> list[RiskContribution]:
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
                members=[phone, said],
                rationale=(
                    f"phone_usage co-occurs with speech classified {speech_class} "
                    f"(within {config.phone_speech_cooccur_ms}ms)"
                ),
            )
        )
    return out


def detect_blur_burst(ctx: DetectCtx) -> list[RiskContribution]:
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


def detect_all_factors(ctx: DetectCtx) -> list[RiskContribution]:
    out: list[RiskContribution] = []
    out.extend(detect_paste_after_blur(ctx))
    out.extend(detect_gaze_with_device(ctx))
    out.extend(detect_gaze_then_answer_commit(ctx))
    out.extend(detect_gaze_off_then_speech_then_answer(ctx))
    out.extend(detect_gaze_second_person_interacting(ctx))
    out.extend(detect_second_person_discussing(ctx))
    out.extend(detect_phone_with_answer_speech(ctx))
    out.extend(detect_blur_burst(ctx))
    out.extend(detect_fullscreen_exit_then_input(ctx))
    out.extend(detect_paste_without_prior_copy(ctx))
    return dedupe_contributions(out)
