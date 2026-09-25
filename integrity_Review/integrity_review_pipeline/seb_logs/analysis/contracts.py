"""Pydantic contracts for SEB Log AI Analysis response."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import Field

from ....contracts.base import ContractModel


class Enforcement(str, Enum):
    """Enforcement status for a finding."""

    BLOCKED_BY_TSB = "blocked_by_tsb"
    RESTRICTION_BREACHED = "restriction_breached"
    NOT_ENFORCED = "not_enforced"
    NOT_APPLICABLE = "not_applicable"


class ConclusionQualifier(str, Enum):
    """Qualifier for the coverage assessment conclusion."""

    NO_SUSPICIOUS_ACTIVITY_OBSERVED = "no_suspicious_activity_observed"
    COULD_NOT_BE_ESTABLISHED = "could_not_be_established"


class PhaseStatus(str, Enum):
    """Status of a journey phase observation."""

    OBSERVED = "observed"
    INFERRED = "inferred"
    NOT_OBSERVED = "not_observed"


class IncidentClassification(str, Enum):
    """Classification of a correlated incident."""

    SUSPICIOUS = "suspicious"
    TECHNICAL = "technical"
    MIXED = "mixed"


class ConfidenceLevel(str, Enum):
    """Confidence level for assessments."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class AudienceGuidance(str, Enum):
    """Audience guidance for non-technical reviewers."""

    NO_ACTION_NEEDED = "no_action_needed"
    INFORMATIONAL = "informational"
    REVIEW_RECOMMENDED = "review_recommended"
    INVESTIGATE_BEFORE_RELEASE = "investigate_before_release"


class SimplifiedPhaseStatus(str, Enum):
    """Status of a simplified journey phase."""

    OBSERVED = "observed"
    INFERRED = "inferred"
    NOT_OBSERVED = "not_observed"
    MIXED = "mixed"


class ConcernLevel(str, Enum):
    """Concern level for a simplified journey phase."""

    NONE = "none"
    MINOR = "minor"
    NEEDS_REVIEW = "needs_review"


class SimplifiedJourneyPhase(ContractModel):
    """Simplified journey phase for non-technical readers."""

    phase_id: str
    phase_name: str
    technical_phase_ids: list[str] = Field(default_factory=list)
    status: SimplifiedPhaseStatus
    concern_level: ConcernLevel
    plain_summary: str
    key_events_plain: list[str] = Field(default_factory=list)


class JourneyPhase(ContractModel):
    """Analysis of a session journey phase."""

    phase_id: str
    phase_name: str
    started_at: datetime | None = None
    ended_at: datetime | None = None
    status: PhaseStatus
    basis: str
    summary: str
    key_events: list[str] = Field(default_factory=list)
    evidence_cited: list[str] = Field(default_factory=list)


class ProcessContext(ContractModel):
    """Process context for a finding."""

    name: str
    pid: int | None = None
    path: str | None = None
    original_name: str | None = None
    signed: bool | None = None


class WindowContext(ContractModel):
    """Window context for a finding."""

    title: str
    handle: int | None = None
    allowed: bool | None = None
    action_taken: str | None = None


class FindingAnalysis(ContractModel):
    """AI analysis of a security finding."""

    finding_id: str
    category: str
    title: str
    what_happened: str
    when_text: str
    why_it_matters: str
    severity: str
    confidence: str
    occurrence_count: int = 1
    first_seen: datetime | None = None
    last_seen: datetime | None = None
    enforcement: Enforcement
    potential_impact: str
    benign_explanation: str
    requires_manual_investigation: bool = False
    signals_cited: list[str] = Field(default_factory=list)
    evidence_cited: list[str] = Field(default_factory=list)
    config_cited: list[str] = Field(default_factory=list)
    process_context: ProcessContext | None = None
    window_context: WindowContext | None = None
    plain_language_title: str | None = None
    plain_language_explanation: str | None = None
    audience_guidance: AudienceGuidance | None = None


class CorrelatedIncidentAnalysis(ContractModel):
    """AI analysis of a correlated incident."""

    incident_id: str
    time_window_start: datetime
    time_window_end: datetime
    classification: IncidentClassification
    event: str
    context: str
    correlation: str
    likely_interpretation: str
    findings_cited: list[str] = Field(default_factory=list)
    evidence_cited: list[str] = Field(default_factory=list)
    severity: str
    confidence: str


class TechnicalProblem(ContractModel):
    """Technical problem identified in the session."""

    problem_type: str
    what_happened: str
    impact_on_session: str
    related_suspicious_findings: list[str] = Field(default_factory=list)
    relationship_explanation: str | None = None


class ConfigurationAssessment(ContractModel):
    """Assessment of the effective configuration."""

    applied_config_summary: str
    deviations: list[str] = Field(default_factory=list)
    protections_weakened: list[str] = Field(default_factory=list)
    config_explained_signals: list[str] = Field(default_factory=list)
    unknown_or_defaulted: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class CoverageAssessment(ContractModel):
    """Assessment of analysis coverage and confidence."""

    analysis_confidence: ConfidenceLevel
    missing_log_files: list[str] = Field(default_factory=list)
    log_gaps: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    conclusion_qualifier: ConclusionQualifier


class ReviewerSummary(ContractModel):
    """Summary for human reviewers."""

    overall_outcome: str
    significant_security_signals: list[str] = Field(default_factory=list)
    significant_technical_problems: list[str] = Field(default_factory=list)
    configuration_deviations: list[str] = Field(default_factory=list)
    coverage_limitations: list[str] = Field(default_factory=list)
    events_requiring_investigation: list[str] = Field(default_factory=list)
    summary_text: str
    plain_verdict_headline: str | None = None
    plain_summary_text: str | None = None


class SebLogAiAnalysisResult(ContractModel):
    """Complete AI analysis result for a SEB log session."""

    session_journey: list[JourneyPhase] = Field(default_factory=list)
    simplified_journey: list[SimplifiedJourneyPhase] = Field(default_factory=list)
    findings: list[FindingAnalysis] = Field(default_factory=list)
    correlated_incidents: list[CorrelatedIncidentAnalysis] = Field(default_factory=list)
    technical_problems: list[TechnicalProblem] = Field(default_factory=list)
    configuration_assessment: ConfigurationAssessment | None = None
    coverage_assessment: CoverageAssessment | None = None
    reviewer_summary: ReviewerSummary | None = None
    raw_llm_response: str | None = None
    validation_errors: list[str] = Field(default_factory=list)
    analysis_failed: bool = False
    failure_reason: str | None = None


class SebLogSessionAnalysis(ContractModel):
    """Combined reduction and AI analysis for a single session."""

    session_folder: str
    org_assessment_id: str
    user_id: str
    reduction: dict[str, Any]
    ai_analysis: SebLogAiAnalysisResult | None = None


class SebLogReviewResult(ContractModel):
    """Complete SEB log review result for a review."""

    review_id: str
    sessions: list[SebLogSessionAnalysis] = Field(default_factory=list)
    analysis_version: str
    total_sessions: int = 0
    sessions_with_ai: int = 0
    sessions_failed: int = 0
