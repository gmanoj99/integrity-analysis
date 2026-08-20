"""Perception test fixtures ported from perceptionFixtures.ts."""

from __future__ import annotations

from integrity_review_pipeline.contracts.perception import (
    PerceptionAttention,
    PerceptionAudio,
    PerceptionBody,
    PerceptionBundle,
    PerceptionEnvironment,
    PerceptionHands,
    PerceptionIdentity,
    PerceptionInteraction,
    PerceptionObjects,
    PerceptionObservation,
    PerceptionPeople,
    PerceptionWindow,
    PerceptionWindowChunk,
)


def make_observation(
    window_id: str,
    *,
    start_ms: int = 0,
    end_ms: int | None = None,
    **overrides,
) -> PerceptionObservation:
    end = end_ms if end_ms is not None else start_ms + 20_000
    base = PerceptionObservation(
        window_id=window_id,
        start_ms=start_ms,
        end_ms=end,
        identity=PerceptionIdentity(),
        attention=PerceptionAttention(),
        hands=PerceptionHands(),
        objects=PerceptionObjects(),
        people=PerceptionPeople(),
        environment=PerceptionEnvironment(),
        body=PerceptionBody(),
        interaction=PerceptionInteraction(),
        audio=PerceptionAudio(),
        confidence=0.8,
        video_available=True,
        capture_quality_tier="HIGH",
    )
    if overrides:
        data = base.model_dump()
        for key, value in overrides.items():
            if isinstance(value, dict) and key in data and isinstance(data[key], dict):
                data[key] = {**data[key], **value}
            else:
                data[key] = value
        return PerceptionObservation.model_validate(data)
    return base


def make_perception_bundle(observations: list[PerceptionObservation]) -> PerceptionBundle:
    windows = []
    for obs in observations:
        windows.append(
            PerceptionWindow(
                window_id=obs.window_id,
                start_ms=obs.start_ms,
                end_ms=obs.end_ms,
                phase="live_exam",
                overlapping_chunks=[
                    PerceptionWindowChunk(
                        chunk_id=f"chunk_{obs.window_id}",
                        sequence=0,
                        chunk_start_ms=obs.start_ms,
                        chunk_end_ms=obs.end_ms,
                    )
                ],
                video_available=True,
                capture_quality_tier="HIGH",
            )
        )
    return PerceptionBundle(
        candidate_id="c1",
        assessment_id="a1",
        produced_at="2026-01-01T00:00:00Z",
        perception_version="test",
        windows=windows,
        observations=observations,
        total_windows=len(windows),
        covered_windows=len(windows),
        unknown_windows=0,
        session_start_ms=0,
        duration_ms=max((o.end_ms for o in observations), default=60_000),
        coverage_ratio=1.0,
        total_video_chunks=len(windows),
    )
