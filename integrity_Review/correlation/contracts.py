from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import Field

from ..contracts.base import ContractModel
from ..machine_facts.contracts import MachineFact
from ..baseline.contracts import StatisticalBaseline

SCOPE35_LOGIC_VERSION = "scope35-v11"
CORRELATED_SCORE_CAP = 100
CONTRIBUTION_DEDUP_BUCKET_MS = 30_000

WEIGHT_TABLE: dict[str, int] = {
    "paste_after_blur": 12,
    "idle_then_large_paste": 10,
    "video_overlap_submission": 18,
    "video_overlap_paste": 14,
    "gaze_off_then_input": 8,
    "improbable_hard_fast_paste": 20,
    "performance_difficulty_mismatch": 10,
    "section_score_time_implausible": 12,
    "question_peer_outlier": 10,
    "paste_on_coding_question": 8,
    "phone_fast_correct": 18,
    "phone_fast_mcq_answers": 16,
    "gaze_fast_correct": 14,
    "gaze_with_device": 16,
    "gaze_then_answer_commit": 10,
    "section_perf_video_cooccurrence": 14,
    "speech_then_correct_hard": 18,
    "whisper_then_paste": 16,
    "av_mismatch_then_input": 14,
    "second_person_then_score_jump": 16,
    "gaze_off_then_speech_then_answer": 18,
    "blur_fast_correct": 16,
    "camera_absent_fast_correct": 16,
    "fast_correct_high_plagiarism": 18,
    "out_of_order_submission_hop": 8,
    "large_unattributed_gap": 6,
    "face_absent_then_paste": 14,
    "gaze_then_correct_mcq_burst": 12,
    "blur_then_score_jump": 12,
    "speech_coaching_then_fast_correct": 18,
    "second_person_then_paste_or_correct": 16,
    "idle_gap_then_fast_correct": 14,
    "deterministic_paste_workflow": 16,
    "external_resource_then_paste": 14,
    "face_absent_during_input": 14,
    "interact_speech_input_correct": 18,
    "gaze_second_person_interacting": 14,
    "second_person_discussing": 16,
    "phone_with_answer_speech": 16,
    "second_monitor_with_gaze": 18,
    "blur_burst": 10,
    "fullscreen_exit_then_input": 14,
    "paste_without_prior_copy": 18,
}

FactorId = Literal[tuple(WEIGHT_TABLE.keys())]  # type: ignore[valid-type]

FACTOR_LABELS: dict[str, str] = {
    "paste_after_blur": "Paste shortly after window blur / tab switch",
    "idle_then_large_paste": "Large paste after sustained idle gap",
    "video_overlap_submission": "Phone / second person overlapping code submission",
    "video_overlap_paste": "Phone / second person overlapping large paste",
    "gaze_off_then_input": "Sustained off-screen gaze then input / paste",
    "improbable_hard_fast_paste": "HARD coding question finished fast with pre-submission paste",
    "performance_difficulty_mismatch": "Easy vs hard performance mismatch vs cohort",
    "section_score_time_implausible": "High section score with unusually fast section time vs peers",
    "question_peer_outlier": "Coding question score outlier vs peers",
    "paste_on_coding_question": "Multiple large pastes on one coding question",
    "phone_fast_correct": "Phone visible during a fast correct coding answer",
    "phone_fast_mcq_answers": "Phone visible during a burst of rapid MCQ answer commits",
    "gaze_fast_correct": "Off-screen gaze then a fast correct coding answer (no paste)",
    "gaze_with_device": "Off-screen gaze coinciding with a visible phone / device",
    "gaze_then_answer_commit": "Off-screen gaze immediately followed by an MCQ answer commit",
    "section_perf_video_cooccurrence": "High/fast section performance with phone or gaze in that section",
    "speech_then_correct_hard": "Answer-discussion speech then a correct HARD question",
    "whisper_then_paste": "Whisper / dictation speech then a paste",
    "av_mismatch_then_input": "Speech without lip movement then typing / paste",
    "second_person_then_score_jump": "Second-person interaction near a high section score",
    "gaze_off_then_speech_then_answer": "Off-screen gaze → coaching speech → answer commit",
    "blur_fast_correct": "Left exam tab/focus then answered a coding question unusually fast",
    "camera_absent_fast_correct": "Left camera frame then answered a coding question unusually fast",
    "fast_correct_high_plagiarism": "Fast correct coding answer with high Topin plagiarism match",
    "out_of_order_submission_hop": "Erratic out-of-order coding question submission sequence",
    "large_unattributed_gap": "Large unattributed section time near leave/blur",
    "face_absent_then_paste": "Left camera frame then a large external paste",
    "gaze_then_correct_mcq_burst": "Off-screen gaze then a rapid MCQ answer burst",
    "blur_then_score_jump": "Left exam tab/focus then a high section score jump",
    "speech_coaching_then_fast_correct": "Coaching speech then a fast correct coding answer",
    "second_person_then_paste_or_correct": "Second-person interaction then paste or fast correct answer",
    "idle_gap_then_fast_correct": "Long inactivity gap then a fast correct coding answer",
    "deterministic_paste_workflow": "Repeated external pastes with correction burst on one question",
    "external_resource_then_paste": "External site/app on screen then paste",
    "face_absent_during_input": "Face absent during screen activity or paste",
    "interact_speech_input_correct": (
        "Interaction/speech about a question then input on that question then correct"
    ),
    "gaze_second_person_interacting": (
        "Off-screen gaze while a second person is interacting with the candidate"
    ),
    "second_person_discussing": (
        "Second person interacting while the speech is about answers or dictation"
    ),
    "phone_with_answer_speech": (
        "Phone visible while answers are being discussed or dictated"
    ),
    "second_monitor_with_gaze": (
        "Secondary workspace on screen while the candidate gazes off-screen"
    ),
    "blur_burst": "Repeated tab switches or focus losses in a short span",
    "fullscreen_exit_then_input": "Fullscreen lost, then typing or a paste",
    "paste_without_prior_copy": (
        "Paste with no matching copy inside the exam — the content came from outside"
    ),
}

TimingBasis = Literal["sqb_submission_window", "section_clock", "cohort_only"]
ProvisionalLane = Literal["red", "amber", "green"]


class CorrelationConfig(ContractModel):
    paste_after_blur_ms: int = 15_000
    idle_then_paste_ms: int = 10_000
    idle_gap_min_ms: int = 8_000
    gaze_off_then_input_ms: int = 20_000
    video_overlap_pad_ms: int = 5_000
    hard_fast_max_seconds: int = 120
    paste_near_submission_ms: int = 30_000
    min_paste_size_for_tier1: int = 50
    peer_z_threshold: float = 1.5
    paste_cluster_min_count: int = 2
    mismatch_easy_max_percentile: int = 40
    mismatch_hard_min_percentile: int = 70
    fast_correct_max_seconds: int = 90
    gaze_device_cooccur_ms: int = 8_000
    gaze_to_commit_ms: int = 6_000
    mcq_fast_gap_ms: int = 3_000
    mcq_fast_min_answers: int = 3
    speech_then_correct_ms: int = 45_000
    whisper_then_paste_ms: int = 20_000
    av_mismatch_then_input_ms: int = 15_000
    blur_to_answer_ms: int = 60_000
    plagiarism_high_threshold: int = 70
    unattributed_gap_min_seconds: int = 180
    hop_min_reversals: int = 2
    second_person_cooccur_ms: int = 8_000
    phone_speech_cooccur_ms: int = 8_000
    second_monitor_cooccur_ms: int = 8_000
    blur_burst_window_ms: int = 120_000
    blur_burst_min_count: int = 3
    fullscreen_lost_to_input_ms: int = 30_000
    copy_before_paste_ms: int = 120_000


DEFAULT_CORRELATION_CONFIG = CorrelationConfig()

TimelineEventSource = Literal["fact", "perception_episode", "screen_episode", "submission", "sqb"]


class TimelineEvent(ContractModel):
    t_ms: int
    end_ms: int | None = None
    source: TimelineEventSource
    kind: str
    section_id: str | None = None
    section_title: str | None = None
    question_number: int | None = None
    question_id: str | None = None
    detail: dict = Field(default_factory=dict)
    evidence_ref: str


class ContributionMember(ContractModel):
    kind: str
    t_ms: int
    end_ms: int | None = None
    evidence_ref: str = ""


class RiskContribution(ContractModel):
    factor_id: str
    label: str
    tier: Literal[1, 2]
    weight: int
    score_contribution: float
    intensity: float = Field(ge=0.0, le=1.0)
    requires_corroboration: bool = False
    timing_basis: TimingBasis | None = None
    section_id: str | None = None
    section_title: str | None = None
    question_number: int | None = None
    question_id: str | None = None
    time_range_ms: tuple[int, int]
    evidence_refs: list[str] = Field(default_factory=list)
    member_events: list[ContributionMember] = Field(default_factory=list)
    cohort_ref_ids: list[str] | None = None
    rationale: str = ""


class CandidateChain(ContractModel):
    chain_id: str
    factor_id: str
    time_range_ms: tuple[int, int]
    duration_ms: int
    question_number: int | None = None
    question_id: str | None = None
    section_id: str | None = None
    members: list[str]
    modality_count: int
    outcome: Literal["correct", "incorrect", "unknown"] | None = None
    audio_summary_en: str | None = None
    speech_content_class: str | None = None
    role_cue: str | None = None
    provisional_lane: ProvisionalLane
    evidence_refs: list[str] = Field(default_factory=list)
    rationale: str = ""


class CorrelatedSignalsBundle(ContractModel):
    model_config = ContractModel.model_config | {"frozen": True}

    produced_at: str
    candidate_id: str
    assessment_id: str
    logic_version: str = SCOPE35_LOGIC_VERSION
    config: CorrelationConfig = DEFAULT_CORRELATION_CONFIG
    timeline_event_count: int
    contributions: list[RiskContribution]
    chains: list[CandidateChain] = Field(default_factory=list)
    total_weighted_score: float
    factor_counts: dict[str, int] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class SyntheticQuestionBoundary(ContractModel):
    section_id: str
    question_number: int
    question_id: str | None = None
    start_offset_ms: int
    end_offset_ms: int
    end_is_fallback: bool = False
    start_from_time_spent: bool | None = None
    timing_unavailable: bool | None = None
    authoritative_time_spent_seconds: float | None = None


def make_contribution(
    factor_id: str,
    *,
    tier: Literal[1, 2],
    intensity: float,
    time_range_ms: tuple[int, int],
    evidence_refs: list[str],
    rationale: str,
    requires_corroboration: bool = False,
    timing_basis: TimingBasis | None = None,
    section_id: str | None = None,
    section_title: str | None = None,
    question_number: int | None = None,
    question_id: str | None = None,
    cohort_ref_ids: list[str] | None = None,
    members: list[Any] | None = None,
) -> RiskContribution:
    weight = WEIGHT_TABLE[factor_id]
    clamped = max(0.0, min(1.0, intensity))
    return RiskContribution(
        factor_id=factor_id,
        label=FACTOR_LABELS[factor_id],
        tier=tier,
        weight=weight,
        intensity=clamped,
        score_contribution=round(weight * clamped, 2),
        requires_corroboration=requires_corroboration,
        timing_basis=timing_basis,
        section_id=section_id,
        section_title=section_title,
        question_number=question_number,
        question_id=question_id,
        time_range_ms=time_range_ms,
        evidence_refs=evidence_refs,
        member_events=[
            ContributionMember(
                kind=str(getattr(m, "kind", "")),
                t_ms=int(getattr(m, "t_ms", 0) or 0),
                end_ms=getattr(m, "end_ms", None),
                evidence_ref=str(getattr(m, "evidence_ref", "") or ""),
            )
            for m in (members or [])
        ],
        cohort_ref_ids=cohort_ref_ids,
        rationale=rationale,
    )


@dataclass
class DetectCtx:
    timeline: list[TimelineEvent]
    boundaries: list[SyntheticQuestionBoundary]
    facts: list[MachineFact]
    baseline: StatisticalBaseline
    config: CorrelationConfig
    notes: list[str] = field(default_factory=list)
