"""Offset clip resolution and media index (no physical clip bytes)."""

from __future__ import annotations

import hashlib
from typing import Any, Literal

from ..contracts.evidence_bundle import ClipRef, ClipSegment, MediaIndexEntry
from ..duck_helpers import attr

EVIDENCE_BUNDLE_LOGIC_VERSION = "scope5-v27"


def _artifact_type_to_evidence(artifact_type: str) -> tuple[Literal["video", "screen"], ...]:
    if artifact_type == "video":
        return ("video",)
    if artifact_type in {"screenRecording", "screen"}:
        return ("screen",)
    if artifact_type == "screenCamera":
        return ("video", "screen")
    return ()


def build_media_index(master_timeline: Any) -> list[MediaIndexEntry]:
    registry = attr(master_timeline, "artifact_registry", "artifactRegistry", default=[]) or []
    entries: list[MediaIndexEntry] = []
    for record in registry:
        artifact_type = str(attr(record, "artifact_type", "artifactType", default=""))
        evidence_types = _artifact_type_to_evidence(artifact_type)
        if not evidence_types:
            continue
       
        session_start_ms = max(
            0, int(attr(record, "session_start_ms", "sessionStartMs", default=0) or 0)
        )
        session_end_ms = max(
            session_start_ms,
            int(attr(record, "session_end_ms", "sessionEndMs", default=0) or 0),
        )
        entries.extend(
            MediaIndexEntry(
                chunk_id=str(attr(record, "artifact_id", "artifactId")),
                evidence_type=evidence_type,
                section_id=attr(record, "section_id", "sectionId", default=None),
                session_start_ms=session_start_ms,
                session_end_ms=session_end_ms,
                duration_ms=max(
                    0, int(attr(record, "local_end_ms", "localEndMs", default=0) or 0)
                ),
                sequence=max(0, int(attr(record, "sequence", default=0) or 0)),
            )
            for evidence_type in evidence_types
        )
    return entries


def resolve_offset_clip_ref(
    *,
    event_ms: int,
    duration_ms: int,
    media_index: list[MediaIndexEntry],
) -> ClipRef | None:
    if not media_index:
        return None
    end_ms = event_ms + max(duration_ms, 1)
    segments: list[ClipSegment] = []
    for entry in media_index:
        if entry.session_end_ms < event_ms or entry.session_start_ms > end_ms:
            continue
        local_start = max(0, event_ms - entry.session_start_ms)
        local_end = min(entry.duration_ms, end_ms - entry.session_start_ms)
        segments.append(
            ClipSegment(
                chunk_id=entry.chunk_id,
                evidence_type=entry.evidence_type,
                seek_to_ms=local_start,
                end_ms_local=local_end,
            )
        )
    if not segments:
        return None
    cache_key = hashlib.sha256(
        f"{event_ms}|{duration_ms}|{','.join(s.chunk_id for s in segments)}".encode()
    ).hexdigest()[:16]
    # An event derived from a pre-T0 chunk carries a negative session offset;
    # the playback window is clamped to the session start for the same reason as
    # in build_media_index.
    clip_start_ms = max(0, event_ms)
    return ClipRef(
        segments=segments,
        clip_start_ms=clip_start_ms,
        clip_end_ms=max(clip_start_ms, end_ms),
        cache_key=cache_key,
    )


def compute_evidence_bundle_version_hash(deliberation_composite_hash: str) -> str:
    bundle_version = hashlib.sha256(b"deterministic|scope5-v27").hexdigest()[:16]
    correlated_version = hashlib.sha256(b"deterministic|scope35-v10").hexdigest()[:16]
    payload = (
        f"{EVIDENCE_BUNDLE_LOGIC_VERSION}|{deliberation_composite_hash}|"
        f"{bundle_version}|{correlated_version}"
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
