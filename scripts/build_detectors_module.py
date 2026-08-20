#!/usr/bin/env python3
"""Emit correlation/detectors.py from embedded source."""

from pathlib import Path

DETECTORS = Path(__file__).resolve().parents[1] / "integrity_review_pipeline/correlation/detectors.py"

BODY = r'''"""Behavioral correlation detectors — port of correlationEngine.ts (behavioral scope only)."""

from __future__ import annotations

from ..machine_facts.kinds import MachineFactKind
from .contracts import (
    DEFAULT_CORRELATION_CONFIG,
    CorrelationConfig,
    DetectCtx,
    RiskContribution,
    make_contribution,
)
from .helpers import (
    AV_MISMATCH_TYPES,
    BLUR_KINDS,
    CAMERA_ABSENT_TYPES,
    GAZE_EVENT_TYPES,
    INPUT_AFTER_GAZE_KINDS,
    SPEECH_INTEGRITY_EVENT_TYPES,
    VIDEO_OVERLAP_EVENT_TYPES,
    WHISPER_DICTATION_TYPES,
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

EXCLUDED_DETECTORS = [
    "detect_section_score_time_implausible",
    "detect_question_peer_outlier",
    "detect_performance_difficulty_mismatch",
    "detect_fast_correct_high_plagiarism",
    "apply_plagiarism_corroboration",
    "detect_second_person_then_score_jump",
    "detect_blur_then_score_jump",
    "detect_section_perf_video_cooccurrence",
]


def detect_paste_after_blur(ctx: DetectCtx) -> list[RiskContribution]:
    timeline, config = ctx.timeline, ctx.config
    blurs = [e for e in timeline if e.kind in BLUR_KINDS]
    phone_ends = perception_episodes_by_kind(timeline, "phone_usage")
    precursors = sorted(
        [*blurs, *[TimelineEventAdapter(p) for p in phone_ends]],
        key=lambda e: e.t_ms,
    )
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


class TimelineEventAdapter:
    """Treat phone episode end as precursor moment."""

    def __init__(self, phone):
        self.t_ms = phone.end_ms or phone.t_ms
        self.end_ms = phone.end_ms
        self.kind = "phone_usage"
        self.section_id = phone.section_id
        self.evidence_ref = phone.evidence_ref


# NOTE: Additional detectors are imported from detectors_impl module.
from .detectors_impl import (  # noqa: E402
    detect_all_factors,
    detect_av_mismatch_then_input,
    detect_blur_fast_correct,
    detect_camera_absent_fast_correct,
    detect_deterministic_paste_workflow,
    detect_external_resource_then_paste,
    detect_face_absent_during_input,
    detect_face_absent_then_paste,
    detect_gaze_fast_correct,
    detect_gaze_off_then_input,
    detect_gaze_off_then_speech_then_answer,
    detect_gaze_then_answer_commit,
    detect_gaze_then_correct_mcq_burst,
    detect_gaze_with_device,
    detect_idle_gap_then_fast_correct,
    detect_idle_then_large_paste,
    detect_improbable_hard_fast_paste,
    detect_interact_speech_input_correct,
    detect_large_unattributed_gap,
    detect_out_of_order_submission_hop,
    detect_paste_on_coding_question,
    detect_phone_fast_correct,
    detect_phone_fast_mcq_answers,
    detect_second_person_then_paste_or_correct,
    detect_speech_coaching_then_fast_correct,
    detect_speech_then_correct_hard,
    detect_video_overlap,
    detect_whisper_then_paste,
)

__all__ = [
    "EXCLUDED_DETECTORS",
    "detect_all_factors",
    "detect_paste_after_blur",
]
'''

DETECTORS.write_text(BODY, encoding="utf-8")
print(f"wrote {DETECTORS}")
