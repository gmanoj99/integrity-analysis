"""Evidence finding contracts (Track B rename)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from ..contracts.base import ContractModel

FINDINGS_LOGIC_VERSION = "trackb-v9"

FindingSource = Literal["video", "keystroke", "screen"]
FindingVerdict = Literal["flagged", "cleared"]
FindingSeverity = Literal["high", "medium", "low"]
EvidenceStrength = Literal["strong", "moderate", "thin"]


class KeystrokeProof(ContractModel):
    keystrokes_in_window: int | None = None
    inserted_char_count: int | None = None
    preceding_gap_ms: int | None = None
    field_id: str | None = None
    final_value_length: int | None = None
    pasted_excerpt: str | None = None


class VideoProof(ContractModel):
    chunk_id: str | None = None
    chunk_sequence: int | None = None
    clip_label: str | None = None
    approx_frame_timestamp_ms: int | None = None


class ScreenProof(ContractModel):
    chunk_id: str
    approx_frame_timestamp_ms: int
    local_offset_ms: int
    clip_label: str | None = None
    pasted_excerpt: str | None = None
    question_number: int | None = None
    visible_question_ref: str | None = None


class SpeechProof(ContractModel):
    conversation_summary_en: str
    speech_language: str
    code_mixing: bool
    speech_content_class: str
    notable_phrases_original: list[str] | None = None
    clip_start_ms: int
    clip_end_ms: int


class VideoProofAnchor(ContractModel):
    chunk_id: str
    chunk_sequence: int | None = None
    clip_label: str | None = None
    approx_frame_timestamp_ms: int


class EvidenceFinding(ContractModel):
    id: str
    source: FindingSource
    event_type: str
    timestamp_window_ms: tuple[int, int]
    attribution: Literal["candidate", "environment", "other_person", "unclear"]
    phase: Literal["setup_verification", "live_exam", "unknown"] = "live_exam"
    severity: FindingSeverity = "medium"
    evidence_ref: str
    verdict: FindingVerdict = "flagged"
    reasoning: str = ""
    evidence_strength: EvidenceStrength | None = None
    data_gaps: str | None = None
    candidate_gaze_diverted_toward_person: bool | None = None
    keystroke_proof: KeystrokeProof | None = None
    video_proof: VideoProof | None = None
    video_proof_anchors: list[VideoProofAnchor] | None = None
    screen_proof: ScreenProof | None = None
    speech_proof: SpeechProof | None = None
    occurrence_count: int | None = None
    gap_ms_to_next: int | None = None


class ClearedItem(ContractModel):
    event_type: str
    reason: str
    note: str
    window_ids: list[str]
    timestamp_window_ms: tuple[int, int]
    duration_sec: int
    occurrence_count: int | None = None
    speech_proof: SpeechProof | None = None


class UnknownItem(ContractModel):
    event_type: str
    reason: str
    note: str
    window_ids: list[str]
    timestamp_window_ms: tuple[int, int]
    speech_proof: SpeechProof | None = None


class CriticalMoment(ContractModel):
    rank: int
    event_type: str
    timestamp_window_ms: tuple[int, int]
    clip_start_ms: int
    clip_end_ms: int
    duration_sec: int
    section_label: str | None = None
    score: float
    one_line_why: str


class ContextualEvent(ContractModel):
    event_type: str
    timestamp_window_ms: tuple[int, int]
    one_line_why: str
    clip_start_ms: int
    clip_end_ms: int
    speech_proof: SpeechProof | None = None


class VideoDerivationResult(ContractModel):
    model_config = ContractModel.model_config | {"frozen": True}

    findings: list[EvidenceFinding]
    critical_moments: list[CriticalMoment] = Field(default_factory=list)
    cleared_ledger: list[ClearedItem] = Field(default_factory=list)
    unknown_ledger: list[UnknownItem] = Field(default_factory=list)
    contextual_events: list[ContextualEvent] = Field(default_factory=list)
    video_observation: str = ""
    video_chunks_sampled: int = 0
    total_video_chunks: int = 0


class EvidenceFindingsResult(ContractModel):
    model_config = ContractModel.model_config | {"frozen": True}

    findings: list[EvidenceFinding]
    video_observation: str = ""
    keystroke_observation: str = ""
    screen_observation: str = ""
    logic_version: str = FINDINGS_LOGIC_VERSION
    video_available: bool = False
    keystroke_available: bool = False
    screen_available: bool = False


class VideoObservationWindow(ContractModel):
    """Minimal Scope 2 window input for deterministic video finding derivation."""

    window_id: str
    start_ms: int
    end_ms: int
    video_available: bool = True
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    face_present: Literal["yes", "no", "UNKNOWN"] = "UNKNOWN"
    second_person_visible: Literal["yes", "no", "UNKNOWN"] = "UNKNOWN"
    phone_visible: Literal["yes", "no", "UNKNOWN"] = "UNKNOWN"
    gaze_direction: Literal["screen", "off_screen", "UNKNOWN"] = "UNKNOWN"
    candidate_speaking: Literal["yes", "no", "UNKNOWN"] = "UNKNOWN"
    speech_content_class: str | None = None
    capture_quality_tier: Literal["HIGH", "MEDIUM", "LOW", "NONE"] = "HIGH"


class ScreenObservation(ContractModel):
    chunk_id: str
    sequence: int
    section_id: str | None = None
    start_ms: int
    end_ms: int
    exam_ui_visible: Literal["yes", "no", "UNKNOWN"] = "UNKNOWN"
    foreground_app_class: str = "UNKNOWN"
    external_resource_labels: list[str] = Field(default_factory=list)
    ai_assistant_ui_visible: Literal["yes", "no", "UNKNOWN"] = "UNKNOWN"
    secondary_workspace_visible: Literal["yes", "no", "UNKNOWN"] = "UNKNOWN"
    fullscreen_exam_likely: Literal["yes", "no", "UNKNOWN"] = "UNKNOWN"
    paste_cue_visible: Literal["yes", "no", "UNKNOWN"] = "UNKNOWN"
    pasted_text_excerpt: str | None = None
    visible_question_ref: str | None = None
    confidence: float = 1.0
    quality_caveat: str | None = None
    video_available: bool = True


class ScreenPerceptionBundle(ContractModel):
    observations: list[ScreenObservation] = Field(default_factory=list)
    session_start_ms: int = 0
    duration_ms: int = 0
    total_chunks: int = 0
