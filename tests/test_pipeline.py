import json
from typing import Any

import pytest

from integrity_review_pipeline.adapters.memory_cache import InMemoryPerceptionCache
from integrity_review_pipeline.deps import PipelineDeps
from integrity_review_pipeline.pipeline import run_integrity_review
from integrity_review_pipeline.worker.contracts import (
    ActivityLog,
    ActivityTimeline,
    Manifest,
    ManifestChunk,
    SectionSpec,
    StagedReviewPayload,
    build_review_request,
)


class FakeObjectStore:
    async def get_bytes(self, ref: str) -> bytes:
        return b"[]"

    async def get_json(self, ref: str) -> Any:
        return []


class FakeGemini:
    async def generate(self, *, model: str, contents: list[Any], config: dict[str, Any]):
        if "flash" in model:
            return {
                "text": json.dumps(
                    {
                        "chunkId": "ignored",
                        "chunkDurationMs": 60_000,
                        "chunkFullyReviewed": True,
                        "captureQualityTier": "HIGH",
                        "events": [],
                    }
                )
            }
        return {
            "text": json.dumps(
                {
                    "episodeAnalysis": [],
                    "candidateSignals": [],
                    "behaviorSummary": "No reviewable integrity signal was detected.",
                    "recommendation": "No manual action is required.",
                    "reasoning": "Available evidence did not support a signal.",
                    "keyReasons": [],
                    "category": "CLEAR",
                }
            )
        }


class NoopLimiter:
    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, *_args: object) -> None:
        return None


class NoopLogger:
    def debug(self, message: str, **fields: Any) -> None:
        pass

    def info(self, message: str, **fields: Any) -> None:
        pass

    def warning(self, message: str, **fields: Any) -> None:
        pass

    def error(self, message: str, **fields: Any) -> None:
        pass


class FakeMediaUriProvider:
    async def media_uri_for(self, chunk: Any) -> str:
        return f"https://media.example/{chunk.source_ref}"


@pytest.mark.asyncio
async def test_input_to_evidence_bundle_pipeline(monkeypatch) -> None:
    async def fake_media_parts(*args: Any, **kwargs: Any):
        return ([{"fileData": {"fileUri": args[0], "mimeType": "video/webm"}}], "file_data")

    monkeypatch.setattr(
        "integrity_review_pipeline.perception.chunk_job.resolve_media_parts",
        fake_media_parts,
    )
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
    request = build_review_request(payload)
    deps = PipelineDeps(
        object_store=FakeObjectStore(),
        gemini=FakeGemini(),
        cache=InMemoryPerceptionCache(),
        limiter=NoopLimiter(),
        logger=NoopLogger(),
        media_uri_provider=FakeMediaUriProvider(),
        organization_id="test-org",
    )
    bundle = await run_integrity_review(request, deps)
    payload = bundle.model_dump(mode="json", by_alias=True, exclude_none=True)

    assert payload["candidateId"] == "candidate-1"
    assert payload["assessmentId"] == "assessment-1"
    assert payload["recommendation"]["category"] == "CLEAR"
    assert "questionInsights" not in payload
    assert "performanceEvidence" not in payload
    assert payload["mediaIndex"][0]["sourceRef"] == "media/camera/attempt-1/1778001319000__60000.webm"
