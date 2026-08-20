"""Scope 2 perception contracts (camera / webcam)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from .base import ContractModel

PerceptionTernary = Literal["yes", "no", "UNKNOWN"]
CaptureQualityTier = Literal["HIGH", "MEDIUM", "LOW", "NONE"]
PerceptionPhase = Literal["setup_verification", "live_exam", "unknown"]
PerceptionEventKind = Literal[
    "sustained_gaze",
    "phone_visible",
    "phone_in_hand",
    "second_person_present",
    "second_person_interaction",
    "face_absent",
    "leave_seat",
    "speech",
    "reaching_outside_frame",
    "notes_visible",
    "headphones_visible",
    "earphone_visible",
    "second_screen_visible",
    "other",
]


class PerceptionWindowChunk(ContractModel):
    chunk_id: str
    sequence: int
    chunk_start_ms: int
    chunk_end_ms: int


class PerceptionWindow(ContractModel):
    window_id: str
    start_ms: int
    end_ms: int
    section_id: str | None = None
    section_type: str | None = None
    phase: PerceptionPhase
    overlapping_chunks: list[PerceptionWindowChunk]
    machine_facts: list[dict[str, Any]] = Field(default_factory=list)
    video_available: bool
    capture_quality_tier: CaptureQualityTier


class PerceptionIdentity(ContractModel):
    face_present: PerceptionTernary = "UNKNOWN"
    face_count: int | Literal["UNKNOWN"] = "UNKNOWN"
    identity_consistent: PerceptionTernary = "UNKNOWN"
    face_occluded: PerceptionTernary = "UNKNOWN"
    face_orientation: Literal[
        "toward_camera", "turned_left", "turned_right", "down", "up", "UNKNOWN"
    ] = "UNKNOWN"


class PerceptionAttention(ContractModel):
    gaze_direction: Literal[
        "screen", "left", "right", "down", "up", "away", "UNKNOWN"
    ] = "UNKNOWN"
    gaze_stable: PerceptionTernary = "UNKNOWN"
    repeated_gaze_pattern: Literal[
        "none", "left_recurring", "right_recurring", "down_recurring", "UNKNOWN"
    ] = "UNKNOWN"
    attention_state: Literal["focused", "distracted", "thinking", "UNKNOWN"] = "UNKNOWN"
    gaze_target: Literal[
        "primary_screen",
        "secondary_monitor",
        "phone",
        "other_person",
        "off_screen_general",
        "UNKNOWN",
    ] = "UNKNOWN"


class PerceptionHands(ContractModel):
    hands_visible: PerceptionTernary = "UNKNOWN"
    hand_count: int | Literal["UNKNOWN"] = "UNKNOWN"
    hand_location: Literal[
        "keyboard", "desk", "phone", "below_frame", "face", "other", "UNKNOWN"
    ] = "UNKNOWN"
    hand_activity: Literal[
        "typing", "writing", "holding_object", "idle", "UNKNOWN"
    ] = "UNKNOWN"
    object_in_hand: Literal[
        "phone", "pen", "paper", "book", "none", "UNKNOWN"
    ] = "UNKNOWN"


class PerceptionObjects(ContractModel):
    phone_visible: PerceptionTernary = "UNKNOWN"
    notebook_visible: PerceptionTernary = "UNKNOWN"
    paper_visible: PerceptionTernary = "UNKNOWN"
    calculator_visible: PerceptionTernary = "UNKNOWN"
    headphones_visible: PerceptionTernary = "UNKNOWN"
    headphones_link: Literal["wired", "wireless", "UNKNOWN"] = "UNKNOWN"
    earphone_visible: PerceptionTernary = "UNKNOWN"
    earphone_link: Literal["wired", "wireless", "UNKNOWN"] = "UNKNOWN"
    second_screen_visible: PerceptionTernary = "UNKNOWN"
    unknown_object_visible: PerceptionTernary = "UNKNOWN"


class PerceptionPeople(ContractModel):
    people_in_frame: int | Literal["UNKNOWN"] = "UNKNOWN"
    second_person_visible: PerceptionTernary = "UNKNOWN"
    second_person_interacting: PerceptionTernary = "UNKNOWN"
    second_person_object_in_hand: Literal["phone", "paper", "none", "UNKNOWN"] = (
        "UNKNOWN"
    )
    second_person_activity: Literal[
        "using_phone",
        "reading",
        "writing",
        "speaking_to_candidate",
        "passing_by",
        "idle",
        "UNKNOWN",
    ] = "UNKNOWN"
    candidate_responding_to_second_person: PerceptionTernary = "UNKNOWN"
    second_person_position: Literal[
        "background", "adjacent", "behind", "leaning_in", "UNKNOWN"
    ] = "UNKNOWN"
    second_person_looks_like: Literal[
        "live_person", "reflection", "poster_or_photo", "on_screen_video", "UNKNOWN"
    ] = "UNKNOWN"
    second_person_role_cue: Literal[
        "unknown",
        "peer_helper",
        "household",
        "invigilator_or_staff",
        "passerby",
        "UNKNOWN",
    ] = "UNKNOWN"


class PerceptionEnvironment(ContractModel):
    lighting_condition: Literal[
        "adequate", "dim", "bright", "backlit", "UNKNOWN"
    ] = "UNKNOWN"
    framing: Literal[
        "full_face", "partial", "obscured", "off_center", "UNKNOWN"
    ] = "UNKNOWN"
    camera_movement: Literal[
        "stable", "minor_movement", "significant_movement", "UNKNOWN"
    ] = "UNKNOWN"
    background_activity: Literal["none", "minor", "significant", "UNKNOWN"] = "UNKNOWN"
    setting_type: Literal[
        "private_room", "shared_space", "exam_hall", "UNKNOWN"
    ] = "UNKNOWN"


class PerceptionBody(ContractModel):
    posture: Literal[
        "upright", "leaning_forward", "leaning_back", "slouched", "UNKNOWN"
    ] = "UNKNOWN"
    leaning_direction: Literal["none", "left", "right", "forward", "UNKNOWN"] = "UNKNOWN"
    left_seat: PerceptionTernary = "UNKNOWN"


class PerceptionInteraction(ContractModel):
    candidate_speaking: PerceptionTernary = "UNKNOWN"
    lip_movement: Literal["active", "minimal", "none", "UNKNOWN"] = "UNKNOWN"
    gestures: PerceptionTernary = "UNKNOWN"
    reaching_outside_frame: PerceptionTernary = "UNKNOWN"
    reaching_direction: Literal[
        "down", "left", "right", "forward", "none", "UNKNOWN"
    ] = "UNKNOWN"


class PerceptionAudio(ContractModel):
    speech_present: PerceptionTernary = "UNKNOWN"
    speech_source: Literal[
        "candidate",
        "other_person_in_room",
        "off_camera",
        "device_playback",
        "mixed",
        "UNKNOWN",
    ] = "UNKNOWN"
    speech_overlap_with_lips: PerceptionTernary = "UNKNOWN"
    speech_style: Literal["normal", "whisper", "raised", "reading_aloud", "UNKNOWN"] = (
        "UNKNOWN"
    )
    speech_language: Literal[
        "en", "te", "hi", "ml", "ta", "mr", "bn", "mixed", "other", "UNKNOWN"
    ] = "UNKNOWN"
    code_mixing: PerceptionTernary = "UNKNOWN"
    background_voices: Literal["none", "one", "multiple", "UNKNOWN"] = "UNKNOWN"
    device_sounds: Literal[
        "none", "notification", "call_ringtone", "keyboard_other", "UNKNOWN"
    ] = "UNKNOWN"
    speech_content_class: Literal[
        "silence",
        "self_talk_or_thinking",
        "reading_question",
        "asking_for_answer",
        "receiving_dictation",
        "discussing_solution",
        "reciting_answer_choices",
        "invigilator_or_admin",
        "technical_exam_help",
        "casual_non_exam",
        "unclear",
        "UNKNOWN",
    ] = "UNKNOWN"
    conversation_summary_en: str | None = None
    notable_phrases_original: list[str] | None = None


class PerceptionObservation(ContractModel):
    window_id: str
    section_id: str | None = None
    start_ms: int
    end_ms: int
    identity: PerceptionIdentity
    attention: PerceptionAttention
    hands: PerceptionHands
    objects: PerceptionObjects
    people: PerceptionPeople
    environment: PerceptionEnvironment
    body: PerceptionBody
    interaction: PerceptionInteraction
    audio: PerceptionAudio
    confidence: float
    quality_caveat: str | None = None
    video_available: bool
    capture_quality_tier: CaptureQualityTier
    missing_video_reason: str | None = None


class PerceptionEvent(ContractModel):
    event_id: str
    kind: PerceptionEventKind
    start_ms_local: int
    end_ms_local: int
    duration_ms: int
    start_ms_session: int
    end_ms_session: int
    chunk_id: str
    attrs: dict[str, Any] = Field(default_factory=dict)
    summary: str | None = None
    linked_prior_event_ids: list[str] = Field(default_factory=list)
    confidence: float
    quality_caveat: str | None = None


class PerceptionChunkBaseline(ContractModel):
    face_present_dominant: PerceptionTernary | None = None
    setting_type: Literal[
        "private_room", "shared_space", "exam_hall", "UNKNOWN"
    ] | None = None
    audio_notes: str | None = None


class PerceptionChunkResult(ContractModel):
    chunk_id: str
    chunk_duration_ms: int = 0
    chunk_fully_reviewed: bool
    capture_quality_tier: CaptureQualityTier
    chunk_baseline: PerceptionChunkBaseline | None = None
    events: list[PerceptionEvent] = Field(default_factory=list)


class PerceptionChunkReview(ContractModel):
    chunk_id: str
    chunk_fully_reviewed: bool
    capture_quality_tier: CaptureQualityTier


class PerceptionBundle(ContractModel):
    candidate_id: str
    assessment_id: str
    produced_at: str
    perception_version: str
    windows: list[PerceptionWindow]
    observations: list[PerceptionObservation]
    events: list[PerceptionEvent] = Field(default_factory=list)
    chunk_results: list[PerceptionChunkReview] = Field(default_factory=list)
    total_windows: int
    covered_windows: int
    unknown_windows: int
    session_start_ms: int
    duration_ms: int
    coverage_ratio: float
    total_video_chunks: int


class CachedPerceptionChunk(ContractModel):
    result: PerceptionChunkResult
    observations: list[PerceptionObservation]


class VideoChunkSpan(ContractModel):
    chunk_id: str
    sequence: int
    start_offset_ms: int
    end_offset_ms: int
    duration_ms: int
    section_id: str | None = None


class PerceptionChunkJobPayload(ContractModel):
    organization_id: str
    candidate_id: str
    assessment_id: str | None = None
    reviewer_id: str
    evidence_type: Literal["video", "screenRecording"]
    chunk_id: str
    sequence: int
    signed_url: str
    duration_ms: int
    start_offset_ms: int
    end_offset_ms: int
    section_id: str | None = None
    retry_not_before: int | None = None
