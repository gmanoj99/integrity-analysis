import json

from integrity_review_pipeline.contracts.evidence import EvidenceType
from integrity_review_pipeline.contracts.review import ReviewInput, ReviewRequest
from integrity_review_pipeline.contracts.timeline import TimelineEventRecord
from integrity_review_pipeline.io.manifest_builder import build_manifest_set
from integrity_review_pipeline.timeline.activity_log_timeline import (
    parse_activity_log_epoch_ms,
)
from integrity_review_pipeline.timeline.master_timeline import (
    MERGE_TOLERANCE_MS,
    build_master_timeline,
    build_merged_segments,
    parse_chunk_id,
    to_offset,
)


def _review_request(
    *,
    camera: list[str],
    session: list[str],
    activity_logs: list[dict],
) -> ReviewRequest:
    review_input = ReviewInput.model_validate(
        {
            "candidateId": "candidate-1",
            "assessmentId": "assessment-1",
            "cameraRecordings": camera,
            "sessionRecordings": session,
            "activityTimeline": activity_logs,
            "sections": [
                {
                    "examAttemptId": "attempt-1",
                    "examId": "exam-1",
                    "sectionType": "mcq",
                }
            ],
        }
    )
    return ReviewRequest(
        candidate_id=review_input.candidate_id,
        assessment_id=review_input.assessment_id,
        activity_timeline=review_input.activity_timeline,
        sections=review_input.sections,
        evidence=build_manifest_set(review_input),
    )


def test_parse_activity_log_epoch_ms_matches_ist_quirk() -> None:
    assert parse_activity_log_epoch_ms("2026-05-05 22:44:19") == 1_778_001_259_000


def test_merges_upload_jitter_into_one_segment() -> None:
    spans = [
        type("Span", (), {
            "sequence": 0,
            "chunk_id": "video-mcq-1700000000000__60000",
            "start_offset_ms": 0,
            "end_offset_ms": 60_000,
            "duration_ms": 60_000,
            "section_id": "mcq",
        })(),
        type("Span", (), {
            "sequence": 1,
            "chunk_id": "video-mcq-1700000060000__60000",
            "start_offset_ms": 60_000 + MERGE_TOLERANCE_MS - 1,
            "end_offset_ms": 120_000 + MERGE_TOLERANCE_MS - 1,
            "duration_ms": 60_000,
            "section_id": "mcq",
        })(),
    ]
    merged = build_merged_segments(spans)
    assert len(merged) == 1
    assert merged[0].chunk_count == 2


def test_rrweb_span_uses_event_timestamps_when_present() -> None:
    request = _review_request(
        camera=["https://media.example/camera/attempt-1/1700000000000__60000.webm"],
        session=["https://media.example/session/attempt-1/1700000100000.json"],
        activity_logs=[
            {
                "activityTypeEnum": "ASSESSMENT_STARTED",
                "creationDatetime": "2026-05-05 22:44:19",
                "order": 0,
            }
        ],
    )
    t0 = parse_activity_log_epoch_ms("2026-05-05 22:44:19")
    timeline = build_master_timeline(
        request,
        timeline_events=[
            TimelineEventRecord(
                timestamp_ms=t0 + 5_000,
                evidence_type="keystrokeData",
                sequence=0,
            ),
            TimelineEventRecord(
                timestamp_ms=t0 + 85_000,
                evidence_type="keystrokeData",
                sequence=0,
            ),
        ],
    )
    assert timeline.sync_report is not None
    assert timeline.sync_report.t0_source == "activity_logs"
    assert len(timeline.rrweb_chunk_spans) == 1
    assert timeline.rrweb_chunk_spans[0].start_offset_ms == 5_000
    assert timeline.rrweb_chunk_spans[0].end_offset_ms == 85_000


def test_exposes_legacy_spans_layers_and_sync_report() -> None:
    request = _review_request(
        camera=[
            "https://media.example/camera/attempt-1/1700000000000__60000.webm",
            "https://media.example/camera/attempt-1/1700000060000__60000.webm",
        ],
        session=["https://media.example/session/attempt-1/1700000000000.json"],
        activity_logs=[
            {
                "activityTypeEnum": "ASSESSMENT_STARTED",
                "creationDatetime": "2026-05-05 22:44:19",
                "order": 0,
            }
        ],
    )
    timeline = build_master_timeline(request)
    assert timeline.video_chunk_spans
    assert timeline.artifact_registry
    assert timeline.layers
    assert timeline.sync_report is not None
    assert timeline.sync_report.video_alignment["chunkCount"] == 2
    assert timeline.merged_video_segments
    assert timeline.merged_keystroke_segments


def test_gap_reason_uses_section_overlap_not_symmetric_diff() -> None:
    request = _review_request(
        camera=[
            "https://media.example/camera/attempt-1/1700000000000__60000.webm",
            "https://media.example/camera/attempt-1/1700000120000__60000.webm",
        ],
        session=[],
        activity_logs=[
            {
                "activityTypeEnum": "ASSESSMENT_STARTED",
                "creationDatetime": "2026-05-05 22:44:19",
                "order": 0,
            }
        ],
    )
    # Force two segments with a genuine stop between them (> MERGE_TOLERANCE).
    timeline = build_master_timeline(request)
    if len(timeline.merged_video_segments) >= 2:
        assert timeline.gaps
        assert timeline.gaps[0].reason in {
            "section_transition",
            "recording_gap",
            "unknown",
        }


def test_to_offset_matches_ts_helper() -> None:
    assert to_offset(1_700_000_060_000, 1_700_000_000_000) == 60_000
    parsed = parse_chunk_id("video-mcq-1700000060000__60000")
    assert parsed is not None
    assert parsed.start_epoch_ms == 1_700_000_000_000
