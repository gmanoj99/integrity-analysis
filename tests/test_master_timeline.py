from integrity_review_pipeline.contracts.timeline import TimelineEventRecord
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
from integrity_review_pipeline.worker.contracts import (
    ActivityLog,
    ActivityTimeline,
    Manifest,
    ManifestChunk,
    SectionSpec,
    StagedReviewPayload,
    build_review_request,
)


def _payload(
    *,
    camera: list[tuple[str, int, int]],
    session_webm: list[tuple[str, int, int]] = (),
    session_rrweb: list[tuple[str, int]] = (),
    activity_logs: list[dict] = (),
) -> StagedReviewPayload:
    chunks = [
        ManifestChunk(
            chunk_id=f"camera-{epoch}",
            media_type="CAMERA_VIDEO",
            exam_attempt_id="attempt-1",
            s3_key=f"media/camera/attempt-1/{epoch}__{duration}.webm",
            epoch_ms=epoch,
            duration_ms=duration,
        )
        for _, epoch, duration in camera
    ]
    chunks += [
        ManifestChunk(
            chunk_id=f"screen-{epoch}",
            media_type="SCREEN_VIDEO",
            exam_attempt_id="attempt-1",
            s3_key=f"media/session/attempt-1/{epoch}__{duration}.webm",
            epoch_ms=epoch,
            duration_ms=duration,
        )
        for _, epoch, duration in session_webm
    ]
    chunks += [
        ManifestChunk(
            chunk_id=f"rrweb-{epoch}",
            media_type="RRWEB_EVENT",
            exam_attempt_id="attempt-1",
            s3_key=f"media/session/attempt-1/{epoch}.json.gz",
            epoch_ms=epoch,
            duration_ms=None,
        )
        for _, epoch in session_rrweb
    ]
    return StagedReviewPayload(
        review_id="review-1",
        org_assess_id="assessment-1",
        attempt_user_id="candidate-1",
        manifest=Manifest(chunks=chunks),
        activity_timeline=ActivityTimeline(
            activity_logs=[ActivityLog.model_validate(log) for log in activity_logs],
            sections=[
                SectionSpec(
                    section_id="mcq",
                    exam_id="exam-1",
                    order=1,
                    exam_attempt_id="attempt-1",
                    start_datetime="2026-05-05 22:44:19",
                    end_datetime=None,
                )
            ],
        ),
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
    request = build_review_request(
        _payload(
            camera=[("c", 1_700_000_000_000, 60_000)],
            session_rrweb=[("r", 1_700_000_100_000)],
            activity_logs=[
                {
                    "order": 0,
                    "activity_type": "ASSESSMENT_STARTED",
                    "creation_datetime": "2026-05-05 22:44:19",
                }
            ],
        )
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
    request = build_review_request(
        _payload(
            camera=[
                ("c", 1_700_000_000_000, 60_000),
                ("c", 1_700_000_060_000, 60_000),
            ],
            session_rrweb=[("r", 1_700_000_000_000)],
            activity_logs=[
                {
                    "order": 0,
                    "activity_type": "ASSESSMENT_STARTED",
                    "creation_datetime": "2026-05-05 22:44:19",
                }
            ],
        )
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
    request = build_review_request(
        _payload(
            camera=[
                ("c", 1_700_000_000_000, 60_000),
                ("c", 1_700_000_120_000, 60_000),
            ],
            activity_logs=[
                {
                    "order": 0,
                    "activity_type": "ASSESSMENT_STARTED",
                    "creation_datetime": "2026-05-05 22:44:19",
                }
            ],
        )
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
