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
    "gaze_with_device": 16,
    "gaze_then_answer_commit": 10,
    "gaze_off_then_speech_then_answer": 18,
    "gaze_second_person_interacting": 14,
    "second_person_discussing": 16,
    "phone_with_answer_speech": 16,
    "blur_burst": 10,
    "fullscreen_exit_then_input": 14,
    "paste_without_prior_copy": 18,
}

FactorId = Literal[tuple(WEIGHT_TABLE.keys())]  # type: ignore[valid-type]

FACTOR_LABELS: dict[str, str] = {
    "paste_after_blur": "Paste shortly after window blur / tab switch",
    "gaze_with_device": "Off-screen gaze coinciding with a visible phone / device",
    "gaze_then_answer_commit": "Off-screen gaze immediately followed by an MCQ answer commit",
    "gaze_off_then_speech_then_answer": "Off-screen gaze → coaching speech → answer commit",
    "gaze_second_person_interacting": (
        "Off-screen gaze while a second person is interacting with the candidate"
    ),
    "second_person_discussing": (
        "Second person interacting while the speech is about answers or dictation"
    ),
    "phone_with_answer_speech": (
        "Phone visible while answers are being discussed or dictated"
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
    video_overlap_pad_ms: int = 5_000
    min_paste_size_for_tier1: int = 50
    gaze_device_cooccur_ms: int = 8_000
    gaze_off_then_input_ms: int = 20_000
    gaze_to_commit_ms: int = 6_000
    speech_then_correct_ms: int = 45_000
    second_person_cooccur_ms: int = 8_000
    phone_speech_cooccur_ms: int = 8_000
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
