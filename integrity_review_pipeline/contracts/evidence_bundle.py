"""Scope 5 evidence bundle contracts (video / screen / rrweb; no cohort extras)."""

from typing import Literal

from pydantic import Field

from .base import ContractModel
from .deliberation import EpisodeAnalysisEntry, RecommendationCategory, SignalResolution

TimelineEntryCategory = Literal["corroborated_evidence", "needs_review"]
ReviewerDecisionLabel = Literal["confirmed", "innocent", "unclear"]
SmartStudentNoteSource = Literal[
    "smart_student_guard_rejection",
    "baseline_within_normal",
    "policy_note",
]
EvidenceTypeLiteral = Literal["video", "screen"]


class ClipSegment(ContractModel):
    chunk_id: str
    evidence_type: EvidenceTypeLiteral
    seek_to_ms: int = Field(ge=0)
    end_ms_local: int | None = Field(default=None, ge=0)


class ClipRef(ContractModel):
    """Offset-only playback reference — no archived clip bytes."""

    segments: list[ClipSegment] = Field(default_factory=list)
    clip_start_ms: int = Field(ge=0)
    clip_end_ms: int = Field(ge=0)
    cache_key: str
    midpoint_fallback: bool = False


class MediaIndexEntry(ContractModel):
    chunk_id: str
    evidence_type: EvidenceTypeLiteral
    source_ref: str | None = None
    section_id: str | None = None
    session_start_ms: int = Field(ge=0)
    session_end_ms: int = Field(ge=0)
    duration_ms: int = Field(ge=0)
    sequence: int = Field(ge=0, default=0)


class BehaviorSummarySection(ContractModel):
    text: str
    source_scope: Literal["scope4_deliberation"] = "scope4_deliberation"


class RecommendationSection(ContractModel):
    category: RecommendationCategory
    confidence: float = Field(ge=0, le=1)
    reasoning: str
    recommendation: str
    independent_source_count: int = 0
    independent_source_types: list[str] = Field(default_factory=list)
    corroboration_downgrade_applied: bool = False
    supporting_signal_ids: list[str] = Field(default_factory=list)


class EvidenceSourceEntry(ContractModel):
    source_type: str
    signal_count: int = 0
    signal_types: list[str] = Field(default_factory=list)


class ConfidenceSection(ContractModel):
    capture_quality_cap_applied: bool = False
    usable_window_ratio: float = Field(ge=0, le=1)
    covered_windows: int = Field(ge=0)
    total_windows: int = Field(ge=0)
    unknown_windows: int = Field(ge=0)
    informative_content_ratio: float = Field(ge=0, le=1)
    informative_content_cap_applied: bool = False
    evidence_source_summary: list[EvidenceSourceEntry] = Field(default_factory=list)
    video_coverage_label: str | None = None
    keystroke_coverage_label: str | None = None
    screen_coverage_label: str | None = None
    video_modality_absent: bool | None = None


class KeyReasonsSection(ContractModel):
    reasons: list[str] = Field(default_factory=list)


class HypothesisArmEntry(ContractModel):
    supporting: list[str] = Field(default_factory=list)
    contradicting: list[str] = Field(default_factory=list)


class DetectedSignalEntry(ContractModel):
    signal_id: str
    signal_type: str
    timestamp_ms: int
    confidence: float = Field(ge=0, le=1)
    resolution: SignalResolution
    source_types: list[str] = Field(default_factory=list)
    supporting_observations: list[str] = Field(default_factory=list)
    supporting_machine_facts: list[str] = Field(default_factory=list)
    supporting_baseline_metrics: list[str] = Field(default_factory=list)
    honest_hypothesis: HypothesisArmEntry
    assisted_hypothesis: HypothesisArmEntry
    innocent_explanation_considered: bool = False
    why_rejected: str = ""
    citations: list[str] = Field(default_factory=list)


class RejectedSignalEntry(ContractModel):
    signal_type: str
    rejected_by: str
    reason: str


class DetectedSignalsSection(ContractModel):
    signals: list[DetectedSignalEntry] = Field(default_factory=list)
    rejected_signals: list[RejectedSignalEntry] = Field(default_factory=list)


class TrackBObservationCard(ContractModel):
    id: str
    event_type: str
    title: str
    timestamp_window_ms: tuple[int, int]
    duration_ms: int = Field(ge=0)
    nested_under_signal_id: str | None = None
    clip_ref: ClipRef | None = None
    detail: str | None = None


class CorrelatedPatternEvent(ContractModel):
    ref: str
    kind: str
    t_ms: int | None = None
    label: str
    detail_snippet: str | None = None
    chars_added: int | None = None
    element_id: str | None = None


class CorrelatedPatternProof(ContractModel):
    time_range_ms: tuple[int, int]
    video_seek_ms: int | None = None
    clip_ref: ClipRef | None = None
    machine_facts: list[dict[str, object]] = Field(default_factory=list)
    perception_window_ids: list[str] = Field(default_factory=list)
    telemetry_only: bool | None = None


class CorrelatedPatternEntry(ContractModel):
    factor_id: str
    label: str
    tier: Literal[1, 2]
    score_contribution: float
    requires_corroboration: bool = False
    rationale: str
    time_range_ms: tuple[int, int]
    evidence_refs: list[str] = Field(default_factory=list)
    events: list[CorrelatedPatternEvent] = Field(default_factory=list)
    proof: CorrelatedPatternProof | None = None
    question_number: int | None = None
    section_title: str | None = None


class CorrelatedPatternsSection(ContractModel):
    patterns: list[CorrelatedPatternEntry] = Field(default_factory=list)
    total_weighted_score: float = 0.0
    logic_version: str = ""
    notes: list[str] = Field(default_factory=list)


class EpisodeAnalysisSection(ContractModel):
    episodes: list[EpisodeAnalysisEntry] = Field(default_factory=list)
    emitted_count: int = 0


class TimelineEntry(ContractModel):
    id: str
    category: TimelineEntryCategory
    timestamp_ms: int
    end_ms: int | None = None
    duration_ms: int | None = None
    event: str
    explanation: str
    confidence: float = Field(ge=0, le=1)
    signal_id: str | None = None
    evidence_source_types: list[str] = Field(default_factory=list)
    references: list[str] = Field(default_factory=list)
    clip_ref: ClipRef | None = None
    gap_ms_to_next: int | None = None
    section_title: str | None = None
    interaction_summary: str | None = None
    speech_summary: str | None = None


class EvidenceTimelineSection(ContractModel):
    entries: list[TimelineEntry] = Field(default_factory=list)


class UnknownInterval(ContractModel):
    start_ms: int
    end_ms: int
    duration_ms: int
    reason: str
    window_id: str | None = None


class UnknownPanelSection(ContractModel):
    intervals: list[UnknownInterval] = Field(default_factory=list)
    total_intervals: int = 0
    total_duration_ms: int = 0
    disclaimer: str
    modality_absent_summary: str | None = None


class SmartStudentNote(ContractModel):
    behavior: str
    explanation: str
    source: SmartStudentNoteSource


class SmartStudentNotesSection(ContractModel):
    notes: list[SmartStudentNote] = Field(default_factory=list)


class ContextualSpeechEvent(ContractModel):
    event_type: str
    timestamp_window_ms: tuple[int, int]
    clip_start_ms: int
    clip_end_ms: int
    speech_language: str | None = None
    code_mixing: bool | None = None
    conversation_summary_en: str | None = None
    notable_phrases_original: list[str] | None = None


class ContextualEventsSection(ContractModel):
    events: list[ContextualSpeechEvent] = Field(default_factory=list)


class IntegrityStoryProof(ContractModel):
    time_range_ms: tuple[int, int]
    video_seek_ms: int | None = None
    clip_ref: ClipRef | None = None
    screen_seek_ms: int | None = None
    screen_clip_ref: ClipRef | None = None
    machine_facts: list[dict[str, object]] = Field(default_factory=list)
    audio_quotes: list[str] = Field(default_factory=list)
    perception_window_ids: list[str] = Field(default_factory=list)
    supporting_factor_ids: list[str] = Field(default_factory=list)
    supporting_factor_labels: list[str] | None = None
    telemetry_only: bool | None = None


class IntegrityStoryEntry(ContractModel):
    story_id: str
    signal_id: str
    signal_type: str
    headline: str
    what_happened: str
    why_it_matters: str
    honest_alternative: str
    severity: Literal["probable", "suspicious", "weak"]
    resolution: SignalResolution
    confidence: float = Field(ge=0, le=1)
    time_range_ms: tuple[int, int]
    question_numbers: list[int] = Field(default_factory=list)
    proof: IntegrityStoryProof


class IntegrityStoriesSection(ContractModel):
    stories: list[IntegrityStoryEntry] = Field(default_factory=list)


class ReviewerActionSection(ContractModel):
    decision: ReviewerDecisionLabel | None = None
    notes: str | None = None
    decided_by: str | None = None
    decided_at: str | None = None


class ProvenanceSection(ContractModel):
    candidate_id: str
    assessment_id: str
    generated_at: str
    timeline_version: str
    machine_facts_version: str
    perception_version: str
    baseline_version: str
    correlated_signals_version: str
    deliberation_version: str
    prompt_version: str
    evidence_bundle_version: str
    clip_cache_version: str = "offset-only-v1"
    model_versions: dict[str, str] = Field(default_factory=dict)
    composite_hash: str


class EvidenceBundle(ContractModel):
    candidate_id: str
    assessment_id: str
    produced_at: str
    behavior_summary: BehaviorSummarySection
    recommendation: RecommendationSection
    confidence: ConfidenceSection
    key_reasons: KeyReasonsSection
    integrity_stories: IntegrityStoriesSection
    track_b_observations: list[TrackBObservationCard] = Field(default_factory=list)
    detected_signals: DetectedSignalsSection
    correlated_patterns: CorrelatedPatternsSection
    episode_analysis: EpisodeAnalysisSection
    evidence_timeline: EvidenceTimelineSection
    unknown_panel: UnknownPanelSection
    smart_student_notes: SmartStudentNotesSection
    contextual_events: ContextualEventsSection
    reviewer_action: ReviewerActionSection
    provenance: ProvenanceSection
    media_index: list[MediaIndexEntry] = Field(default_factory=list)
