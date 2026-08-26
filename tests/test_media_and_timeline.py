import base64
import gzip
import json
import zlib

from integrity_review_pipeline.media.keystroke_reader import decode_rrweb_chunk
from integrity_review_pipeline.media.perception_media import (
    EBML_MAGIC,
    normalize_perception_media,
)
from integrity_review_pipeline.timeline.activity_log_timeline import (
    parse_activity_log_epoch_ms,
)
from integrity_review_pipeline.timeline.master_timeline import (
    build_master_timeline,
    parse_chunk_id,
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


def test_decodes_legacy_screen_json() -> None:
    webm = EBML_MAGIC + b"webm-data"
    wrapped = json.dumps(
        {"file": f"data:video/webm;base64,{base64.b64encode(webm).decode()}"}
    ).encode()
    payload = normalize_perception_media(gzip.compress(wrapped))
    assert payload.bytes == webm
    assert payload.mime_type == "video/webm"


def test_decodes_plain_and_per_event_zlib_rrweb() -> None:
    event = {"type": 5, "timestamp": 123, "data": {"tag": "WINDOW_BLUR"}}
    assert decode_rrweb_chunk(json.dumps([event]).encode()) == [event]
    compressed = zlib.compress(json.dumps(event).encode()).decode("latin-1")
    assert decode_rrweb_chunk(json.dumps([compressed]).encode()) == [event]


def test_chunk_epoch_is_upload_time_for_duration_chunks() -> None:
    parsed = parse_chunk_id("video-mcq-1700000060000__60000")
    assert parsed is not None
    assert parsed.start_epoch_ms == 1_700_000_000_000
    assert parsed.end_epoch_ms == 1_700_000_060_000


def test_builds_one_master_timeline() -> None:
    payload = StagedReviewPayload(
        review_id="review-1",
        org_assess_id="assessment-1",
        attempt_user_id="candidate-1",
        manifest=Manifest(
            chunks=[
                ManifestChunk(
                    chunk_id="camera-1",
                    media_type="CAMERA_VIDEO",
                    exam_attempt_id="attempt-1",
                    s3_key="media/camera/attempt-1/1778001319000__60000.webm",
                    epoch_ms=1_778_001_319_000,
                    duration_ms=60_000,
                ),
                ManifestChunk(
                    chunk_id="rrweb-1",
                    media_type="RRWEB_EVENT",
                    exam_attempt_id="attempt-1",
                    s3_key="media/session/attempt-1/1778001319000.json",
                    epoch_ms=1_778_001_319_000,
                    duration_ms=None,
                ),
            ]
        ),
        activity_timeline=ActivityTimeline(
            activity_logs=[
                ActivityLog(
                    order=0,
                    activity_type="ASSESSMENT_STARTED",
                    creation_datetime="2026-05-05 22:44:19",
                )
            ],
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
    timeline = build_master_timeline(build_review_request(payload))
    assert timeline.sync_report is not None
    assert timeline.sync_report.t0_source == "activity_logs"
    assert timeline.session_start_ms == parse_activity_log_epoch_ms("2026-05-05 22:44:19")
    assert len(timeline.artifact_registry) == 2
    assert timeline.merged_video_segments[0].start_ms == 0
    assert timeline.video_chunk_spans
    assert timeline.rrweb_chunk_spans
    assert timeline.layers
