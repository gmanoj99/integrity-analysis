from __future__ import annotations

import hashlib
from datetime import UTC, datetime

from ..contracts.screen_perception import (
    SCREEN_PERCEPTION_VERSION,
    ScreenObservation,
    ScreenPerceptionBundle,
)
from ..contracts.timeline import MasterTimeline
from ..contracts.perception import VideoChunkSpan
from ..deps import PipelineDeps
from .chunk_job import parse_screen_response, screen_chunk_cache_key


SCREEN_ARTIFACT_TYPES = frozenset({"screenRecording", "screenCamera"})


def screen_chunk_spans(timeline: MasterTimeline) -> list[VideoChunkSpan]:
    records = sorted(
        [
            record
            for record in timeline.artifact_registry
            if record.artifact_type in SCREEN_ARTIFACT_TYPES
        ],
        key=lambda item: (item.session_start_ms, item.sequence),
    )
    return [
        VideoChunkSpan(
            chunk_id=record.artifact_id,
            sequence=record.sequence,
            start_offset_ms=record.session_start_ms,
            end_offset_ms=record.session_end_ms,
            duration_ms=record.session_end_ms - record.session_start_ms,
            section_id=record.section_id,
        )
        for record in records
    ]


def compute_screen_perception_version_hash(timeline: MasterTimeline) -> str:
    chunk_ids = ",".join(sorted(span.chunk_id for span in screen_chunk_spans(timeline)))
    return hashlib.sha256(f"{SCREEN_PERCEPTION_VERSION}|chunks:{chunk_ids}".encode()).hexdigest()[:16]


def unknown_observation(span: VideoChunkSpan, caveat: str) -> ScreenObservation:
    return ScreenObservation(
        chunk_id=span.chunk_id,
        sequence=span.sequence,
        section_id=span.section_id,
        start_ms=span.start_offset_ms,
        end_ms=span.end_offset_ms,
        exam_ui_visible="UNKNOWN",
        foreground_app_class="UNKNOWN",
        external_resource_labels=[],
        ai_assistant_ui_visible="UNKNOWN",
        secondary_workspace_visible="UNKNOWN",
        fullscreen_exam_likely="UNKNOWN",
        paste_cue_visible="UNKNOWN",
        pasted_text_excerpt=None,
        visible_question_ref=None,
        confidence=0.0,
        quality_caveat=caveat,
        video_available=False,
    )


async def _load_screen_observation(
    deps: PipelineDeps,
    span: VideoChunkSpan,
    payload_kwargs: dict[str, object],
) -> ScreenObservation:
    cache_key = screen_chunk_cache_key(span.chunk_id)
    cached = await deps.cache.get(cache_key)
    if isinstance(cached, dict):
        if cached.get("chunkId") or cached.get("chunk_id"):
            return ScreenObservation.model_validate(cached)
        if isinstance(cached.get("text"), str):
            from ..contracts.perception import PerceptionChunkJobPayload

            payload = PerceptionChunkJobPayload.model_validate(payload_kwargs)
            return parse_screen_response(cached["text"], payload)
    return unknown_observation(span, "Perception job not yet complete.")


async def build_screen_perception_bundle(
    deps: PipelineDeps,
    *,
    organization_id: str,
    candidate_id: str,
    assessment_id: str,
    master_timeline: MasterTimeline,
) -> ScreenPerceptionBundle:
    spans = screen_chunk_spans(master_timeline)
    if not spans:
        return ScreenPerceptionBundle(
            candidate_id=candidate_id,
            assessment_id=assessment_id,
            produced_at=datetime.now(tz=UTC).isoformat(),
            screen_perception_version=SCREEN_PERCEPTION_VERSION,
            observations=[],
            total_chunks=0,
            covered_chunks=0,
            session_start_ms=master_timeline.session_start_ms,
            duration_ms=master_timeline.duration_ms,
        )

    deps.logger.info(
        "screen-perception: aggregating cached chunk observations",
        candidate_id=candidate_id,
        total_chunks=len(spans),
    )

    observations: list[ScreenObservation] = []

    async def load_one(span: VideoChunkSpan) -> ScreenObservation:
        async with deps.limiter:
            payload_kwargs = {
                "organizationId": organization_id,
                "candidateId": candidate_id,
                "assessmentId": assessment_id,
                "reviewerId": "system",
                "evidenceType": "screenRecording",
                "chunkId": span.chunk_id,
                "sequence": span.sequence,
                "signedUrl": "unused://cache",
                "durationMs": span.duration_ms,
                "startOffsetMs": span.start_offset_ms,
                "endOffsetMs": span.end_offset_ms,
                "sectionId": span.section_id,
            }
            return await _load_screen_observation(deps, span, payload_kwargs)

    for span in spans:
        try:
            observations.append(await load_one(span))
        except Exception as exc:  # noqa: BLE001 - mirror TS Promise.allSettled placeholder path
            deps.logger.warning(
                "screen-perception: chunk rejected — UNKNOWN placeholder",
                candidate_id=candidate_id,
                chunk_id=span.chunk_id,
                err=str(exc),
            )
            observations.append(unknown_observation(span, "Transient analysis failure — not cached."))

    observations.sort(key=lambda item: item.start_ms)
    covered_chunks = sum(1 for observation in observations if observation.video_available)
    return ScreenPerceptionBundle(
        candidate_id=candidate_id,
        assessment_id=assessment_id,
        produced_at=datetime.now(tz=UTC).isoformat(),
        screen_perception_version=SCREEN_PERCEPTION_VERSION,
        observations=observations,
        total_chunks=len(spans),
        covered_chunks=covered_chunks,
        session_start_ms=master_timeline.session_start_ms,
        duration_ms=master_timeline.duration_ms,
    )
