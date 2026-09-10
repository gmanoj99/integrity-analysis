"""Scope 2 camera perception bundle assembly."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any

from ..contracts.perception import (
    CachedPerceptionChunk,
    PerceptionAttention,
    PerceptionAudio,
    PerceptionBody,
    PerceptionBundle,
    PerceptionChunkReview,
    PerceptionChunkResult,
    PerceptionEnvironment,
    PerceptionEvent,
    PerceptionHands,
    PerceptionIdentity,
    PerceptionInteraction,
    PerceptionObservation,
    PerceptionObjects,
    PerceptionPeople,
    PerceptionTernary,
    PerceptionWindow,
    PerceptionWindowChunk,
    VideoChunkSpan,
)
from ..contracts.timeline import ArtifactRecord, MasterTimeline
from ..deps import PipelineDeps
from ..prompts import PERCEPTION_ASSEMBLY_LOGIC_VERSION, PERCEPTION_MODEL_VERSION, PERCEPTION_PROMPT_VERSION
from .chunk_job import perception_chunk_cache_key

PERCEPTION_VERSION = PERCEPTION_PROMPT_VERSION

# A combined chunk is one file holding both signals, so it is camera evidence
# as much as it is screen evidence: both selectors claim it, and the single
# perception job for it fills both caches under the same chunk id.
CAMERA_ARTIFACT_TYPES = frozenset({"video", "screenCamera"})


def video_chunk_spans(timeline: MasterTimeline) -> list[VideoChunkSpan]:
    records = sorted(
        [
            record
            for record in timeline.artifact_registry
            if record.artifact_type in CAMERA_ARTIFACT_TYPES
        ],
        key=lambda item: item.sequence,
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


def compute_perception_version_hash(timeline: MasterTimeline) -> str:
    sorted_chunk_ids = ",".join(
        sorted(span.chunk_id for span in video_chunk_spans(timeline))
    )
    digest = hashlib.sha256(
        f"{PERCEPTION_MODEL_VERSION}|{PERCEPTION_VERSION}|{PERCEPTION_ASSEMBLY_LOGIC_VERSION}|chunks:{sorted_chunk_ids}".encode()
    ).hexdigest()
    return digest[:16]


def _unknown_groups() -> dict[str, Any]:
    return {
        "identity": PerceptionIdentity(),
        "attention": PerceptionAttention(),
        "hands": PerceptionHands(),
        "objects": PerceptionObjects(),
        "people": PerceptionPeople(),
        "environment": PerceptionEnvironment(),
        "body": PerceptionBody(),
        "interaction": PerceptionInteraction(),
        "audio": PerceptionAudio(),
    }


def build_coverage_hole_observation(
    window: PerceptionWindow,
    reason: str,
) -> PerceptionObservation:
    groups = _unknown_groups()
    return PerceptionObservation(
        window_id=window.window_id,
        section_id=window.section_id,
        start_ms=window.start_ms,
        end_ms=window.end_ms,
        identity=groups["identity"],
        attention=groups["attention"],
        hands=groups["hands"],
        objects=groups["objects"],
        people=groups["people"],
        environment=groups["environment"],
        body=groups["body"],
        interaction=groups["interaction"],
        audio=groups["audio"],
        confidence=0.0,
        quality_caveat=reason,
        video_available=False,
        capture_quality_tier="NONE",
        missing_video_reason=reason,
    )


def chunk_to_window(span: VideoChunkSpan, live_start_ms: int) -> PerceptionWindow:
    phase = "setup_verification" if span.start_offset_ms < live_start_ms else "live_exam"
    return PerceptionWindow(
        window_id=f"w_{span.start_offset_ms}_{span.end_offset_ms}",
        start_ms=span.start_offset_ms,
        end_ms=span.end_offset_ms,
        section_id=span.section_id,
        section_type=span.section_id,
        phase=phase,  # type: ignore[arg-type]
        overlapping_chunks=[
            PerceptionWindowChunk(
                chunk_id=span.chunk_id,
                sequence=span.sequence,
                chunk_start_ms=span.start_offset_ms,
                chunk_end_ms=span.end_offset_ms,
            )
        ],
        machine_facts=[],
        video_available=True,
        capture_quality_tier="MEDIUM",
    )


def ensure_session_times(
    observations: list[PerceptionObservation],
    span: VideoChunkSpan,
) -> list[PerceptionObservation]:
    patched: list[PerceptionObservation] = []
    for observation in observations:
        if (
            observation.start_ms == 0
            and observation.end_ms == span.duration_ms
            and span.start_offset_ms > 0
        ):
            start_ms = span.start_offset_ms
            end_ms = span.end_offset_ms
            patched.append(
                observation.model_copy(
                    update={
                        "start_ms": start_ms,
                        "end_ms": end_ms,
                        "window_id": f"w_{start_ms}_{end_ms}",
                        "section_id": observation.section_id or span.section_id,
                    }
                )
            )
            continue
        if not observation.section_id and span.section_id:
            patched.append(observation.model_copy(update={"section_id": span.section_id}))
            continue
        patched.append(observation)
    return patched


def _load_cached_chunk(raw: dict[str, Any] | None) -> CachedPerceptionChunk | None:
    if not raw:
        return None
    if "result" in raw and "observations" in raw:
        return CachedPerceptionChunk.model_validate(raw)
    return None


async def assemble_perception_from_streamed_chunks(
    deps: PipelineDeps,
    *,
    organization_id: str,
    candidate_id: str,
    assessment_id: str,
    master_timeline: MasterTimeline,
) -> PerceptionBundle:
    spans = video_chunk_spans(master_timeline)
    live_start_ms = (
        min(section.start_ms for section in master_timeline.sections)
        if master_timeline.sections
        else 0
    )

    windows: list[PerceptionWindow] = []
    all_observations: list[PerceptionObservation] = []
    events: list[PerceptionEvent] = []
    chunk_results: list[PerceptionChunkReview] = []

    for span in spans:
        window = chunk_to_window(span, live_start_ms)
        windows.append(window)
        cache_key = perception_chunk_cache_key(span.chunk_id)
        raw = await deps.cache.get(cache_key)
        streamed = _load_cached_chunk(raw if isinstance(raw, dict) else None)

        if streamed and streamed.result.chunk_fully_reviewed:
            chunk_results.append(
                PerceptionChunkReview(
                    chunk_id=span.chunk_id,
                    chunk_fully_reviewed=True,
                    capture_quality_tier=streamed.result.capture_quality_tier,
                )
            )
            window.capture_quality_tier = streamed.result.capture_quality_tier
            events.extend(streamed.result.events)
            raw_obs = (
                ensure_session_times(streamed.observations, span)
                if streamed.observations
                else [
                    build_coverage_hole_observation(
                        window,
                        "Quiet chunk (fully reviewed, no events).",
                    ).model_copy(
                        update={
                            "video_available": True,
                            "capture_quality_tier": streamed.result.capture_quality_tier,
                            "confidence": 0.7,
                            "quality_caveat": None,
                            "missing_video_reason": None,
                        }
                    )
                ]
            )
            # One observation per perception event, each keeping the event's own
            # start/end. Merging them into a single per-window row kept only the
            # first event's span and OR-ed every other event's attributes onto
            # it, so a 3s glance and a 40s phone-in-hand in the same chunk came
            # out as one 3s row claiming both — and every duration downstream was
            # computed from that. The window id still groups them for citations.
            all_observations.extend(
                obs.model_copy(update={"window_id": window.window_id}) for obs in raw_obs
            )
            continue

        chunk_results.append(
            PerceptionChunkReview(
                chunk_id=span.chunk_id,
                chunk_fully_reviewed=False,
                capture_quality_tier="NONE",
            )
        )
        window.video_available = False
        window.capture_quality_tier = "NONE"
        reason = (
            "Streaming perception incomplete for this chunk (chunkFullyReviewed missing)."
            if streamed
            else "Streaming perception not yet available for this chunk — coverage hole (no GCS re-analysis)."
        )
        all_observations.append(build_coverage_hole_observation(window, reason))

    all_observations.sort(key=lambda item: item.start_ms)
    # Coverage counts windows, not observations: a chunk yielding five events is
    # still one window of video, and counting rows would push the ratio over 1.
    covered_windows = sum(1 for window in windows if window.video_available)
    unknown_windows = len(windows) - covered_windows
    coverage_ratio = covered_windows / len(windows) if windows else 0.0

    deps.logger.info(
        "perception: assembled from streamed PERCEPTION_CHUNK artifacts",
        candidate_id=candidate_id,
        total_chunks=len(spans),
        events=len(events),
        covered_windows=covered_windows,
        unknown_windows=unknown_windows,
        coverage_ratio=coverage_ratio,
    )

    return PerceptionBundle(
        candidate_id=candidate_id,
        assessment_id=assessment_id,
        produced_at=datetime.now(tz=UTC).isoformat(),
        perception_version=PERCEPTION_VERSION,
        windows=windows,
        observations=all_observations,
        events=events,
        chunk_results=chunk_results,
        total_windows=len(windows),
        covered_windows=covered_windows,
        unknown_windows=unknown_windows,
        session_start_ms=master_timeline.session_start_ms,
        duration_ms=master_timeline.duration_ms,
        coverage_ratio=coverage_ratio,
        total_video_chunks=len(spans),
    )


async def build_perception_bundle(
    deps: PipelineDeps,
    *,
    organization_id: str,
    candidate_id: str,
    assessment_id: str,
    master_timeline: MasterTimeline,
) -> PerceptionBundle:
    version_hash = compute_perception_version_hash(master_timeline)
    bundle_cache_key = f"perception_observations:{version_hash}"
    cached = await deps.cache.get(bundle_cache_key)
    if isinstance(cached, dict) and cached.get("bundle"):
        return PerceptionBundle.model_validate(cached["bundle"])

    bundle = await assemble_perception_from_streamed_chunks(
        deps,
        organization_id=organization_id,
        candidate_id=candidate_id,
        assessment_id=assessment_id,
        master_timeline=master_timeline,
    )

    total_video_chunks = len(video_chunk_spans(master_timeline))
    should_persist = bundle.coverage_ratio > 0 or total_video_chunks == 0
    if should_persist:
        await deps.cache.set(bundle_cache_key, {"bundle": bundle.model_dump(by_alias=True)})
    else:
        deps.logger.warning(
            "perception: skipping cache write for 0% coverage bundle",
            candidate_id=candidate_id,
            coverage_ratio=bundle.coverage_ratio,
            total_video_chunks=total_video_chunks,
        )
    return bundle


def compute_analysed_video_duration_ms(
    bundle: PerceptionBundle,
    chunk_spans: list[VideoChunkSpan],
) -> int:
    # Analysed footage is measured over windows, not observations: an
    # observation now spans one event, so unioning observation spans would
    # report "time something happened" rather than time reviewed.
    covered = [
        (window.start_ms, window.end_ms)
        for window in bundle.windows
        if window.video_available
    ]
    if not covered:
        return 0

    def union_length(intervals: list[tuple[int, int]]) -> int:
        ordered = sorted(intervals, key=lambda item: item[0])
        merged: list[tuple[int, int]] = []
        for start, end in ordered:
            if not merged or start > merged[-1][1]:
                merged.append((start, end))
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        return sum(end - start for start, end in merged)

    if not chunk_spans:
        return union_length(covered)

    footage = [(span.start_offset_ms, span.end_offset_ms) for span in chunk_spans]
    clipped: list[tuple[int, int]] = []
    for start, end in covered:
        for foot_start, foot_end in footage:
            clip_start = max(start, foot_start)
            clip_end = min(end, foot_end)
            if clip_end > clip_start:
                clipped.append((clip_start, clip_end))
    return union_length(clipped)
