from __future__ import annotations

import hashlib
from typing import Any, Literal

from ..contracts.evidence_bundle import ClipRef, ClipSegment, MediaIndexEntry
from ..duck_helpers import attr

EVIDENCE_BUNDLE_LOGIC_VERSION = "scope5-v29"


EVENT_TYPE_EVIDENCE_STREAM: dict[str, Literal["video", "screen"]] = {
    "possible_audio_coaching": "video",
    "possible_second_person_involvement": "video",
    "possible_external_consultation": "video",
    "possible_remote_dictation": "video",
    "audio_wearable_assisted_comms": "video",
    "phone_usage": "video",
    "reading_notes": "video",
    "external_help": "video",
    "discussing_solution_audio": "video",
    "whisper_or_dictation": "video",
    "off_camera_voice_coaching": "video",
    "device_call_during_attempt": "video",
    "av_speech_without_lips": "video",
    "call_ringtone": "video",
    "suspicious_eye_movement": "video",
    "suspicious_focus_pattern": "video",
    "brief_look_away": "video",
    "left_examination_area": "video",
    "left_seat_body_present": "video",
    "no_candidate": "video",
    "prolonged_absence": "video",
    "multiple_faces": "video",
    "background_person_passing_by": "video",
    "brief_seat_movement": "video",
    "fullscreen_exam_lost": "screen",
    "external_resource_open": "screen",
    "ai_assistant_ui_visible": "screen",
    "unauthorized_reference_usage": "screen",
    "secondary_workspace_visible": "screen",
    "external_paste": "screen",
    "mass_paste": "screen",
    "screen_external_paste": "screen",
    "answer_appeared": "screen",
    "abnormal_paste_workflow": "screen",
    "rapid_typing": "screen",
    "typing_restart": "screen",
    "bulk_delete": "screen",
    "abnormal_correction_pattern": "screen",
    "typing_cadence_mismatch": "screen",
}


def evidence_stream_for_event(event_type: str) -> Literal["video", "screen"] | None:
    return EVENT_TYPE_EVIDENCE_STREAM.get(event_type)


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
    single_segment: bool = False,
    prefer_evidence_type: Literal["video", "screen"] | None = None,
) -> ClipRef | None:
    if not media_index:
        return None
    end_ms = event_ms + max(duration_ms, 1)
    segments: list[ClipSegment] = []
    covered: list[MediaIndexEntry] = []
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
        covered.append(entry)
    if not segments:
        return None
    if single_segment:
        candidates = [
            e for e in covered if e.evidence_type == prefer_evidence_type
        ] or covered
        best = next(
            (
                e
                for e in candidates
                if e.session_start_ms <= event_ms < e.session_start_ms + e.duration_ms
            ),
            candidates[0],
        )
        want = min(max(duration_ms, 1), best.duration_ms)
        local_start = min(
            max(0, event_ms - best.session_start_ms), max(0, best.duration_ms - want)
        )
        segments = [
            ClipSegment(
                chunk_id=best.chunk_id,
                evidence_type=best.evidence_type,
                seek_to_ms=local_start,
                end_ms_local=local_start + want,
            )
        ]
        event_ms = best.session_start_ms + local_start
        end_ms = event_ms + want
    clip_start_ms = max(0, event_ms)
    return ClipRef(
        segments=segments,
        clip_start_ms=clip_start_ms,
        clip_end_ms=max(clip_start_ms, end_ms),
    )


def compute_evidence_bundle_version_hash(deliberation_composite_hash: str) -> str:
    bundle_version = hashlib.sha256(b"deterministic|scope5-v28").hexdigest()[:16]
    correlated_version = hashlib.sha256(b"deterministic|scope35-v10").hexdigest()[:16]
    payload = (
        f"{EVIDENCE_BUNDLE_LOGIC_VERSION}|{deliberation_composite_hash}|"
        f"{bundle_version}|{correlated_version}"
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
