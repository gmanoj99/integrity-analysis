"""Camera and screen perception layer."""

from .chunk_job import (
    ScreenCameraChunkAnalysis,
    analyze_camera_chunk,
    analyze_screen_camera_chunk,
    analyze_screen_chunk,
    ensure_speech_event_completeness,
    is_sidecar_chunk,
    parse_events_response,
    parse_screen_response,
    perception_chunk_cache_key,
    process_perception_chunk_job,
    project_events_to_observations,
    screen_chunk_cache_key,
    split_screen_camera_response,
)
from .perception_engine import (
    PERCEPTION_VERSION,
    assemble_perception_from_streamed_chunks,
    build_perception_bundle,
    compute_analysed_video_duration_ms,
    compute_perception_version_hash,
    merge_observations_sharing_window_id,
    video_chunk_spans,
)
from .screen_perception_engine import (
    build_screen_perception_bundle,
    compute_screen_perception_version_hash,
    screen_chunk_spans,
)

__all__ = [
    "PERCEPTION_VERSION",
    "ScreenCameraChunkAnalysis",
    "analyze_camera_chunk",
    "analyze_screen_camera_chunk",
    "analyze_screen_chunk",
    "assemble_perception_from_streamed_chunks",
    "build_perception_bundle",
    "build_screen_perception_bundle",
    "compute_analysed_video_duration_ms",
    "compute_perception_version_hash",
    "compute_screen_perception_version_hash",
    "ensure_speech_event_completeness",
    "is_sidecar_chunk",
    "merge_observations_sharing_window_id",
    "parse_events_response",
    "parse_screen_response",
    "perception_chunk_cache_key",
    "process_perception_chunk_job",
    "project_events_to_observations",
    "screen_chunk_cache_key",
    "screen_chunk_spans",
    "split_screen_camera_response",
    "video_chunk_spans",
]
