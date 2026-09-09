"""Scope 4 deliberation contracts (video / screen / rrweb pipeline)."""

from typing import Literal

from pydantic import Field

from .base import ContractModel

RecommendationCategory = Literal["CLEAR", "REVIEW_REQUIRED", "STRONG_EVIDENCE"]
EvidenceSourceType = Literal[
    "visual_observation",
    "audio_observation",
    "machine_fact",
    "statistical_baseline",
    "screen_observation",
]
SignalType = Literal[
    "possible_external_consultation",
    "possible_second_person_involvement",
    "unauthorized_reference_usage",
    "abnormal_paste_workflow",
    "suspicious_focus_pattern",
    "abnormal_correction_pattern",
    "typing_cadence_mismatch",
    "inconsistent_interaction_sequence",
    "possible_audio_coaching",
    "possible_remote_dictation",
]
IntegrityStorySeverity = Literal["probable", "suspicious", "weak"]
SignalResolution = Literal["honest", "assisted", "ambiguous"]
EpisodeScope = Literal["timed", "session"]
RejectedByRule = Literal[
    "SignalTypeRule",
    "EpisodeRefRule",
    "SmartStudentGuard",
    "UnknownRule",
    "CitationRule",
    "ValueResolutionRule",
    "ConjunctionRule",
    "NoFactInventionRule",
]


class HypothesisArm(ContractModel):
    supporting: list[str] = Field(default_factory=list)
    contradicting: list[str] = Field(default_factory=list)


class IntegrityStoryProofAnchors(ContractModel):
    window_ids: list[str] = Field(default_factory=list)
    machine_fact_kinds: list[str] = Field(default_factory=list)
    audio_quotes_en: list[str] = Field(default_factory=list)
    seek_ms_hints: list[int] = Field(default_factory=list)
    factor_ids: list[str] = Field(default_factory=list)


class IntegrityStory(ContractModel):
    headline: str
    what_happened: str
    why_it_matters: str
    honest_alternative: str
    severity: IntegrityStorySeverity = "suspicious"
    involved_questions: list[int] = Field(default_factory=list)
    proof_anchors: IntegrityStoryProofAnchors = Field(
        default_factory=IntegrityStoryProofAnchors
    )


class ValidatedSignal(ContractModel):
    signal_id: str
    signal_type: SignalType
    hypothesis_honest: HypothesisArm
    hypothesis_assisted: HypothesisArm
    resolution: SignalResolution
    confidence: float = Field(ge=0, le=1)
    innocent_explanation_considered: bool = False
    why_rejected: str = ""
    machine_facts_cited: list[str] = Field(default_factory=list)
    observations_cited: list[str] = Field(default_factory=list)
    baseline_metrics_cited: list[str] = Field(default_factory=list)
    source_types: list[EvidenceSourceType] = Field(default_factory=list)
    integrity_story: IntegrityStory | None = None


class RejectedSignal(ContractModel):
    signal_type: str
    rejected_by: RejectedByRule
    reason: str


class DeliberationRecommendation(ContractModel):
    behavior_summary: str
    recommendation: str
    category: RecommendationCategory
    confidence: float = Field(ge=0, le=1)
    reasoning: str
    key_reasons: list[str] = Field(default_factory=list)
    independent_sources_count: int = 0
    independent_source_types: list[EvidenceSourceType] = Field(default_factory=list)
    supporting_signals: list[str] = Field(default_factory=list)


class DeliberationProvenance(ContractModel):
    deliberation_prompt_version: str
    perception_version_hash: str
    baseline_version_hash: str
    composite_version_hash: str


class EpisodeAnalysisEntry(ContractModel):
    episode_id: str
    scope: EpisodeScope
    time_range_ms: tuple[int, int]
    origins: list[str] = Field(default_factory=list)
    factor_ids: list[str] = Field(default_factory=list)
    window_ids: list[str] = Field(default_factory=list)
    machine_fact_kinds: list[str] = Field(default_factory=list)
    episode_summary: str = ""
    suspicious_behavior_type: str = "none"
    will_emit_signal: bool = False
    reason_not_signalled: str | None = None
    emitted_signal_ids: list[str] = Field(default_factory=list)


class DeliberationBundle(ContractModel):
    candidate_id: str
    assessment_id: str
    produced_at: str
    model_version: str
    prompt_version: str
    provenance: DeliberationProvenance
    validated_signals: list[ValidatedSignal] = Field(default_factory=list)
    rejected_signal_count: int = 0
    rejected_signals: list[RejectedSignal] = Field(default_factory=list)
    recommendation: DeliberationRecommendation
    episode_analysis: list[EpisodeAnalysisEntry] = Field(default_factory=list)
    episode_inventory_count: int = 0
    corroboration_downgrade_applied: bool = False
    capture_quality_cap_applied: bool = False
    informative_content_ratio: float = 1.0
    informative_content_cap_applied: bool = False
