"""Focused tests for the Python perception layer port."""

from __future__ import annotations

import base64
import json
from typing import Any

import httpx
import pytest

from integrity_review_pipeline.contracts.perception import (
    CachedPerceptionChunk,
    PerceptionChunkJobPayload,
    PerceptionChunkResult,
    PerceptionEvent,
)
from integrity_review_pipeline.contracts.timeline import (
    ArtifactRecord,
    CanonicalTimeline,
    MasterTimeline,
)
from integrity_review_pipeline.deps import PipelineDeps
from integrity_review_pipeline.media.perception_media import EBML_MAGIC, resolve_media_parts
from integrity_review_pipeline.perception.chunk_job import (
    is_sidecar_chunk,
    parse_events_response,
    parse_screen_response,
    process_perception_chunk_job,
    project_events_to_observations,
)
from integrity_review_pipeline.perception.perception_engine import (
    build_perception_bundle,
    merge_observations_sharing_window_id,
)
from integrity_review_pipeline.perception.screen_perception_engine import (
    build_screen_perception_bundle,
)


class MemoryCache:
    def __init__(self) -> None:
        self.store: dict[str, dict[str, Any]] = {}

    async def get(self, key: str) -> dict[str, Any] | None:
        return self.store.get(key)

    async def set(self, key: str, value: dict[str, Any]) -> None:
        self.store[key] = value


class NoopLimiter:
    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, *_args: object) -> None:
        return None


class RecordingGemini:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[dict[str, Any]] = []

    async def generate(self, *, model: str, contents: Any, config: dict[str, Any]) -> dict[str, Any]:
        self.calls.append({"model": model, "contents": contents, "config": config})
        return {"text": self.text}


class NoopLogger:
    def debug(self, message: str, **fields: Any) -> None: ...

    def info(self, message: str, **fields: Any) -> None: ...

    def warning(self, message: str, **fields: Any) -> None: ...

    def error(self, message: str, **fields: Any) -> None: ...


class DummyObjectStore:
    async def get_bytes(self, ref: str) -> bytes:
        raise NotImplementedError

    async def get_json(self, ref: str) -> Any:
        raise NotImplementedError


class DummyMediaUriProvider:
    async def media_uri_for(self, chunk: Any) -> str:
        return "unused://media"


def make_payload(**overrides: Any) -> PerceptionChunkJobPayload:
    base = {
        "organizationId": "org-1",
        "candidateId": "candidate-1",
        "assessmentId": "assessment-1",
        "reviewerId": "reviewer-1",
        "evidenceType": "video",
        "chunkId": "chunk_test",
        "sequence": 1,
        "signedUrl": "https://media.example/chunk.webm",
        "durationMs": 60_000,
        "startOffsetMs": 0,
        "endOffsetMs": 60_000,
        "sectionId": "section-1",
    }
    base.update(overrides)
    return PerceptionChunkJobPayload.model_validate(base)


def make_event(kind: str, attrs: dict[str, Any] | None = None) -> PerceptionEvent:
    return PerceptionEvent(
        event_id=f"e_{kind}",
        kind=kind,  # type: ignore[arg-type]
        start_ms_local=0,
        end_ms_local=5_000,
        duration_ms=5_000,
        start_ms_session=10_000,
        end_ms_session=15_000,
        chunk_id="chunk_test",
        attrs=attrs or {},
        summary=None,
        linked_prior_event_ids=[],
        confidence=0.9,
        quality_caveat=None,
    )


def make_result(events: list[PerceptionEvent]) -> PerceptionChunkResult:
    return PerceptionChunkResult(
        chunk_id="chunk_test",
        chunk_duration_ms=60_000,
        chunk_fully_reviewed=True,
        capture_quality_tier="HIGH",
        events=events,
    )


def make_master_timeline(*records: ArtifactRecord) -> MasterTimeline:
    return MasterTimeline(
        candidate_id="candidate-1",
        assessment_id="assessment-1",
        session_start_ms=1_700_000_000_000,
        duration_ms=120_000,
        canonical_timeline=CanonicalTimeline(
            T0=1_700_000_000_000,
            total_duration_ms=120_000,
            anchors=[],
            sections=[],
            source="activity_logs",
        ),
        artifact_registry=list(records),
        gaps=[],
        sections=[],
        merged_video_segments=[],
        merged_keystroke_segments=[],
        merged_screen_segments=[],
    )


def make_deps(gemini: RecordingGemini | None = None) -> PipelineDeps:
    return PipelineDeps(
        object_store=DummyObjectStore(),
        gemini=gemini or RecordingGemini("{}"),
        cache=MemoryCache(),
        limiter=NoopLimiter(),
        logger=NoopLogger(),
        media_uri_provider=DummyMediaUriProvider(),
        organization_id="test-org",
    )


def test_sidecar_detection() -> None:
    assert is_sidecar_chunk("screen-1700000060000__events")
    assert is_sidecar_chunk("screen-1700000060000__metadata")
    assert not is_sidecar_chunk("screen-1700000060000__60000")


def test_project_events_phone_in_hand() -> None:
    observations = project_events_to_observations(make_result([make_event("phone_in_hand")]), None)
    assert observations[0].objects.phone_visible == "yes"
    assert observations[0].hands.hands_visible == "yes"
    assert observations[0].hands.object_in_hand == "phone"
    assert observations[0].hands.hand_location == "phone"


def test_project_events_speech_backfill() -> None:
    observations = project_events_to_observations(
        make_result([make_event("speech", {"speechPresent": "yes"})]),
        None,
    )
    assert observations[0].audio.speech_present == "yes"
    assert observations[0].audio.speech_content_class == "unclear"
    assert observations[0].audio.conversation_summary_en == "Speech present without parseable content."
    assert observations[0].quality_caveat


def test_merge_observations_by_window_id() -> None:
    phone = project_events_to_observations(make_result([make_event("phone_visible")]), None)[0]
    speech = project_events_to_observations(
        make_result(
            [
                make_event(
                    "speech",
                    {
                        "speechPresent": "yes",
                        "speechContentClass": "reciting_answer_choices",
                        "conversationSummaryEn": "Candidate murmured option C.",
                    },
                )
            ]
        ),
        None,
    )[0]
    merged = merge_observations_sharing_window_id(
        [
            phone.model_copy(update={"window_id": "w_0_60000"}),
            speech.model_copy(update={"window_id": "w_0_60000"}),
        ]
    )
    assert len(merged) == 1
    assert merged[0].objects.phone_visible == "yes"
    assert merged[0].audio.speech_content_class == "reciting_answer_choices"


@pytest.mark.asyncio
async def test_resolve_media_parts_uses_file_data_for_raw_webm() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.headers.get("Range") == "bytes=0-3":
            return httpx.Response(206, content=EBML_MAGIC)
        raise AssertionError("full fetch should not run for raw webm")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        parts, delivery = await resolve_media_parts(
            "https://media.example/raw.webm",
            "video/webm",
            ["prompt"],
            client=client,
        )
    assert delivery == "file_data"
    assert parts[-1]["fileData"]["fileUri"].endswith("raw.webm")


@pytest.mark.asyncio
async def test_resolve_media_parts_uses_inline_data_for_wrapped_json() -> None:
    webm = EBML_MAGIC + b"screen-data"
    wrapped = json.dumps(
        {"file": f"data:video/webm;base64,{base64.b64encode(webm).decode()}"}
    ).encode()

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.headers.get("Range") == "bytes=0-3":
            return httpx.Response(206, content=b"{")
        return httpx.Response(200, content=wrapped)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        parts, delivery = await resolve_media_parts(
            "https://media.example/wrapped.json",
            "video/webm",
            ["prompt"],
            client=client,
        )
    assert delivery == "inline_data"
    decoded = base64.b64decode(parts[-1]["inlineData"]["data"])
    assert decoded == webm


@pytest.mark.asyncio
async def test_process_camera_chunk_uses_file_data_and_caches_result() -> None:
    gemini = RecordingGemini(
        json.dumps(
            {
                "chunkId": "camera-1",
                "chunkDurationMs": 60_000,
                "chunkFullyReviewed": True,
                "captureQualityTier": "HIGH",
                "events": [],
            }
        )
    )
    deps = make_deps(gemini)

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.headers.get("Range") == "bytes=0-3":
            return httpx.Response(206, content=EBML_MAGIC)
        raise AssertionError("unexpected full fetch")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        original_resolve = resolve_media_parts

        async def patched_resolve(*args: Any, **kwargs: Any):
            kwargs["client"] = client
            return await original_resolve(*args, **kwargs)

        import integrity_review_pipeline.perception.chunk_job as chunk_job

        chunk_job.resolve_media_parts = patched_resolve  # type: ignore[assignment]
        try:
            payload = make_payload(
                chunkId="camera-1",
                signedUrl="https://media.example/camera-1.webm",
            )
            result = await process_perception_chunk_job(deps, payload)
        finally:
            chunk_job.resolve_media_parts = original_resolve  # type: ignore[assignment]

    assert isinstance(result, CachedPerceptionChunk)
    assert result.result.chunk_fully_reviewed is True
    assert gemini.calls[0]["contents"][0]["parts"][-1]["fileData"]["fileUri"].endswith(
        "camera-1.webm"
    )


@pytest.mark.asyncio
async def test_process_screen_sidecar_is_skipped() -> None:
    deps = make_deps()
    payload = make_payload(
        evidenceType="screenRecording",
        chunkId="screen-1700000060000__events",
    )
    result = await process_perception_chunk_job(deps, payload)
    assert result is None


@pytest.mark.asyncio
async def test_build_perception_bundle_from_cached_chunks() -> None:
    deps = make_deps()
    cached = CachedPerceptionChunk(
        result=PerceptionChunkResult(
            chunk_id="camera-1",
            chunk_duration_ms=60_000,
            chunk_fully_reviewed=True,
            capture_quality_tier="HIGH",
            events=[make_event("phone_visible", {"phoneVisible": "yes"})],
        ),
        observations=project_events_to_observations(
            PerceptionChunkResult(
                chunk_id="camera-1",
                chunk_duration_ms=60_000,
                chunk_fully_reviewed=True,
                capture_quality_tier="HIGH",
                events=[make_event("phone_visible", {"phoneVisible": "yes"})],
            ),
            "section-1",
        ),
    )
    await deps.cache.set("scope2-v11|chunk:camera-1", cached.model_dump(by_alias=True))

    timeline = make_master_timeline(
        ArtifactRecord(
            artifact_id="camera-1",
            artifact_type="video",
            section_id="section-1",
            sequence=1,
            session_start_ms=0,
            session_end_ms=60_000,
            local_end_ms=60_000,
            epoch_start_ms=1_700_000_000_000,
            epoch_end_ms=1_700_000_060_000,
            quality="ok",
        )
    )
    bundle = await build_perception_bundle(
        deps,
        organization_id="org-1",
        candidate_id="candidate-1",
        assessment_id="assessment-1",
        master_timeline=timeline,
    )
    assert bundle.total_video_chunks == 1
    assert bundle.coverage_ratio == 1.0
    assert bundle.observations[0].objects.phone_visible == "yes"
    assert bundle.windows[0].capture_quality_tier == "HIGH"


@pytest.mark.asyncio
async def test_build_screen_bundle_from_cached_observation() -> None:
    deps = make_deps()
    await deps.cache.set(
        "screen-perc-v4|chunk:screen-1",
        {
            "chunkId": "screen-1",
            "sequence": 1,
            "sectionId": "section-1",
            "startMs": 0,
            "endMs": 60_000,
            "examUiVisible": "yes",
            "foregroundAppClass": "browser",
            "externalResourceLabels": ["chatgpt.com"],
            "aiAssistantUiVisible": "no",
            "secondaryWorkspaceVisible": "no",
            "fullscreenExamLikely": "yes",
            "pasteCueVisible": "no",
            "pastedTextExcerpt": None,
            "visibleQuestionRef": "Q1",
            "confidence": 0.8,
            "qualityCaveat": None,
            "videoAvailable": True,
        },
    )
    timeline = make_master_timeline(
        ArtifactRecord(
            artifact_id="screen-1",
            artifact_type="screenRecording",
            section_id="section-1",
            sequence=1,
            session_start_ms=0,
            session_end_ms=60_000,
            local_end_ms=60_000,
            epoch_start_ms=1_700_000_000_000,
            epoch_end_ms=1_700_000_060_000,
            quality="ok",
        )
    )
    bundle = await build_screen_perception_bundle(
        deps,
        organization_id="org-1",
        candidate_id="candidate-1",
        assessment_id="assessment-1",
        master_timeline=timeline,
    )
    assert bundle.total_chunks == 1
    assert bundle.covered_chunks == 1
    assert bundle.observations[0].foreground_app_class == "browser"


def test_parse_screen_response_truncates_fields() -> None:
    payload = make_payload(evidenceType="screenRecording", chunkId="screen-1")
    observation = parse_screen_response(
        json.dumps(
            {
                "examUiVisible": "yes",
                "foregroundAppClass": "browser",
                "externalResourceLabels": ["x" * 120],
                "aiAssistantUiVisible": "no",
                "secondaryWorkspaceVisible": "no",
                "fullscreenExamLikely": "yes",
                "pasteCueVisible": "yes",
                "pastedTextExcerpt": "x" * 400,
                "visibleQuestionRef": "y" * 200,
                "confidence": 0.7,
                "qualityCaveat": "z" * 300,
            }
        ),
        payload,
    )
    assert len(observation.external_resource_labels[0]) == 80
    assert len(observation.pasted_text_excerpt or "") == 300
    assert len(observation.visible_question_ref or "") == 120
    assert len(observation.quality_caveat or "") == 240


def test_parse_events_response_defaults_invalid_tier() -> None:
    payload = make_payload()
    parsed = parse_events_response(
        json.dumps(
            {
                "chunkId": "chunk_test",
                "chunkDurationMs": 60_000,
                "chunkFullyReviewed": True,
                "captureQualityTier": "GOOD",
                "events": [],
            }
        ),
        payload,
    )
    assert parsed is not None
    assert parsed.capture_quality_tier == "MEDIUM"
