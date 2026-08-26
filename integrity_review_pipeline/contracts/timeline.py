"""Timeline contracts shared by analysis stages."""

from typing import Any, Literal

from pydantic import Field

from .base import ContractModel


class ActivityAnchor(ContractModel):
    order: int
    type: str
    epoch_ms: int
    session_offset_ms: int
    section_id: str | None = None
    end_reason: str | None = None


class SessionSection(ContractModel):
    section_id: str
    label: str
    section_type: str | None = None
    exam_attempt_id: str | None = None
    exam_id: str | None = None
    start_ms: int
    end_ms: int


class CanonicalTimeline(ContractModel):
    t0: int = Field(alias="T0")
    total_duration_ms: int
    anchors: list[ActivityAnchor]
    sections: list[SessionSection]
    source: Literal["activity_logs", "fallback"]


class VideoChunkSpan(ContractModel):
    sequence: int
    chunk_id: str
    start_offset_ms: int
    end_offset_ms: int
    duration_ms: int
    section_id: str | None = None


class RrwebChunkSpan(ContractModel):
    sequence: int
    chunk_id: str
    start_offset_ms: int
    end_offset_ms: int
    section_id: str | None = None


class UnknownSegment(ContractModel):
    start_offset_ms: int
    end_offset_ms: int
    reason: Literal["gap", "missing_sequence", "fetch_error"]
    sequences: list[int] | None = None


class ArtifactRecord(ContractModel):
    artifact_id: str
    artifact_type: Literal["video", "rrweb", "screenRecording"]
    section_id: str | None = None
    sequence: int
    session_start_ms: int
    session_end_ms: int
    local_start_ms: int = 0
    local_end_ms: int
    epoch_start_ms: int
    epoch_end_ms: int
    quality: Literal["ok", "estimated"]


class MergedSegment(ContractModel):
    start_ms: int
    end_ms: int
    duration_ms: int
    chunk_ids: list[str]
    sequences: list[int]
    sections: list[str]
    chunk_count: int


class GapSegment(ContractModel):
    start_ms: int
    end_ms: int
    duration_ms: int
    reason: Literal[
        "before_recording", "section_transition", "recording_gap", "unknown"
    ]
    adjacent_artifact_ids: tuple[str | None, str | None]


class TimelineLayer(ContractModel):
    type: Literal["video", "rrweb", "screenRecording"]
    artifacts: list[ArtifactRecord]


class TimelineEvent(ContractModel):
    id: str
    source: Literal[
        "video",
        "keystrokeData",
        "screenRecording",
        "client_reported",
        "activityLog",
        "topinBundle",
    ]
    kind: str
    raw_timestamp_ms: int
    session_offset_ms: int
    local_offset_ms: int | None = None
    chunk_artifact_id: str | None = None
    signal_source: str | None = None
    detail: dict[str, Any] = Field(default_factory=dict)


class TimelineEventRecord(ContractModel):
    """Optional behavior-timeline rows used to anchor chunk spans."""

    timestamp_ms: int
    evidence_type: Literal["video", "keystrokeData", "screenRecording", "client_reported"]
    sequence: int | None = None
    kind: str = "unknown"
    event_id: str | None = None
    signal_source: str | None = None
    detail: dict[str, Any] = Field(default_factory=dict)


class SyncReport(ContractModel):
    t0_epoch_ms: int
    t0_source: Literal[
        "activity_logs", "fallback_events", "fallback_chunk_epoch", "none"
    ]
    canonical_duration_ms: int
    duration_source: Literal[
        "activity_logs", "video_chunks", "screen_chunks", "rrweb_chunks", "none"
    ]
    video_alignment: dict[str, int | None]
    rrweb_alignment: dict[str, int | None]
    screen_alignment: dict[str, int | None]
    machine_fact_alignment: dict[str, int | None]
    activity_log_anchor_count: int
    canonical_section_count: int
    unresolved_issues: list[str]


class MasterTimeline(ContractModel):
    candidate_id: str
    assessment_id: str
    session_start_ms: int
    duration_ms: int
    canonical_timeline: CanonicalTimeline
    sync_report: SyncReport | None = None
    artifact_registry: list[ArtifactRecord]
    layers: list[TimelineLayer] = Field(default_factory=list)
    gaps: list[GapSegment]
    sections: list[SessionSection]
    merged_video_segments: list[MergedSegment]
    merged_keystroke_segments: list[MergedSegment]
    merged_screen_segments: list[MergedSegment]
    video_chunk_spans: list[VideoChunkSpan] = Field(default_factory=list)
    rrweb_chunk_spans: list[RrwebChunkSpan] = Field(default_factory=list)
    screen_chunk_spans: list[VideoChunkSpan] = Field(default_factory=list)
    unknown_segments: list[UnknownSegment] = Field(default_factory=list)
    events: list[TimelineEvent] = Field(default_factory=list)
