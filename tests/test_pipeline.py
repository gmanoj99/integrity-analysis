import json
from typing import Any

import pytest

from integrity_review_pipeline.adapters.defaults import (
    InMemoryPerceptionCache,
    SemaphoreLimiter,
)
from integrity_review_pipeline.deps import PipelineDeps
from integrity_review_pipeline.io.review_io import load_review_request
from integrity_review_pipeline.pipeline import run_integrity_review


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


class NoopLogger:
    def debug(self, message: str, **fields: Any) -> None:
        pass

    def info(self, message: str, **fields: Any) -> None:
        pass

    def warning(self, message: str, **fields: Any) -> None:
        pass

    def error(self, message: str, **fields: Any) -> None:
        pass


class MediaProvider:
    async def media_uri_for(self, chunk: Any) -> str:
        return str(chunk.signed_url)


@pytest.mark.asyncio
async def test_input_to_evidence_bundle_pipeline(tmp_path, monkeypatch) -> None:
    async def fake_media_parts(*args: Any, **kwargs: Any):
        return ([{"fileData": {"fileUri": args[0], "mimeType": "video/webm"}}], "file_data")

    monkeypatch.setattr(
        "integrity_review_pipeline.perception.chunk_job.resolve_media_parts",
        fake_media_parts,
    )
    input_path = tmp_path / "input.json"
    input_path.write_text(
        json.dumps(
            {
                "candidateId": "candidate-1",
                "assessmentId": "assessment-1",
                "cameraRecordings": [
                    "https://media.example/camera/attempt-1/"
                    "1700000060000__60000.webm"
                ],
                "sessionRecordings": [
                    "https://media.example/session/attempt-1/1700000060000.json"
                ],
                "activityTimeline": [
                    {
                        "eventType": "ASSESSMENT_STARTED",
                        "timestamp": 1_700_000_000_000,
                        "order": 0,
                    }
                ],
                "sections": [
                    {
                        "examAttemptId": "attempt-1",
                        "examId": "exam-1",
                        "sectionType": "mcq",
                    }
                ],
            }
        )
    )
    request = load_review_request(input_path)
    deps = PipelineDeps(
        object_store=FakeObjectStore(),
        gemini=FakeGemini(),
        cache=InMemoryPerceptionCache(),
        limiter=SemaphoreLimiter(12),
        logger=NoopLogger(),
        media_uri_provider=MediaProvider(),
    )
    bundle = await run_integrity_review(request, deps)
    payload = bundle.model_dump(mode="json", by_alias=True, exclude_none=True)

    assert payload["candidateId"] == "candidate-1"
    assert payload["assessmentId"] == "assessment-1"
    assert payload["recommendation"]["category"] == "CLEAR"
    assert "questionInsights" not in payload
    assert "performanceEvidence" not in payload
    assert payload["mediaIndex"][0]["sourceRef"].startswith("https://")
