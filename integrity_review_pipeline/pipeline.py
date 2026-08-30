"""End-to-end in-memory integrity review pipeline."""

from __future__ import annotations

import asyncio
from typing import Any

from .adapters.ai_usage_logger import STEP_DELIBERATION
from .behavioral_pipeline import BehavioralArtifacts, decode_rrweb_chunks, run_behavioral_analysis
from .contracts.evidence import EvidenceType, ExamMode
from .contracts.evidence_bundle import EvidenceBundle, MediaIndexEntry
from .contracts.perception import PerceptionChunkJobPayload, PerceptionObservation
from .contracts.review import ReviewRequest
from .deliberation import (
    DeliberationInput,
    build_deliberation_bundle,
    build_deliberation_prompt,
)
from .deps import PipelineDeps
from .evidence_bundle import EvidenceBundleInput, assemble_evidence_bundle
from .findings.contracts import EvidenceFinding, VideoObservationWindow
from .gemini_call import generate_and_log
from .perception import (
    build_perception_bundle,
    build_screen_perception_bundle,
    process_perception_chunk_job,
)
from .perception.perception_engine import compute_perception_version_hash
from .prompts.deliberation import DELIBERATION_PROMPT_VERSION
from .prompts.shared import DEFAULT_GEMINI_PRO_MODEL
from .timeline import build_master_timeline


def _chunk_map(request: ReviewRequest) -> dict[str, Any]:
    return {
        chunk.chunk_id: chunk
        for manifest in request.evidence.manifests
        for chunk in manifest.chunks
    }


async def _perception_jobs(
    request: ReviewRequest, timeline: Any, deps: PipelineDeps
) -> list[PerceptionChunkJobPayload]:
    chunks = _chunk_map(request)
    spans = [*timeline.video_chunk_spans, *timeline.screen_chunk_spans]
    relevant = [
        (span, chunks[span.chunk_id])
        for span in spans
        if span.chunk_id in chunks and chunks[span.chunk_id].sidecar_role is None
    ]
    media_uris = await asyncio.gather(
        *(deps.media_uri_provider.media_uri_for(chunk) for _, chunk in relevant)
    )
    return [
        PerceptionChunkJobPayload(
            organization_id=deps.organization_id,
            candidate_id=request.candidate_id,
            assessment_id=request.assessment_id,
            reviewer_id="system:local-runner",
            evidence_type=(
                "video" if chunk.evidence_type == EvidenceType.VIDEO else "screenRecording"
            ),
            chunk_id=chunk.chunk_id,
            sequence=chunk.sequence,
            signed_url=media_uri,
            duration_ms=span.duration_ms,
            start_offset_ms=span.start_offset_ms,
            end_offset_ms=span.end_offset_ms,
            section_id=chunk.section_id,
        )
        for (span, chunk), media_uri in zip(relevant, media_uris, strict=True)
    ]


def _video_windows(observations: list[PerceptionObservation]) -> list[VideoObservationWindow]:
    windows: list[VideoObservationWindow] = []
    for observation in observations:
        gaze = observation.attention.gaze_direction
        windows.append(
            VideoObservationWindow(
                window_id=observation.window_id,
                start_ms=observation.start_ms,
                end_ms=observation.end_ms,
                video_available=observation.video_available,
                confidence=observation.confidence,
                face_present=observation.identity.face_present,
                second_person_visible=observation.people.second_person_visible,
                phone_visible=observation.objects.phone_visible,
                gaze_direction=(
                    "screen"
                    if gaze == "screen"
                    else "UNKNOWN"
                    if gaze == "UNKNOWN"
                    else "off_screen"
                ),
                candidate_speaking=observation.interaction.candidate_speaking,
                speech_content_class=observation.audio.speech_content_class,
                capture_quality_tier=observation.capture_quality_tier,
            )
        )
    return windows


async def _load_rrweb(request: ReviewRequest, deps: PipelineDeps) -> list[Any]:
    manifest = request.evidence.by_type(EvidenceType.KEYSTROKE_DATA)

    async def load(chunk: Any) -> tuple[str, int, bytes]:
        return (
            chunk.chunk_id,
            chunk.sequence,
            await deps.object_store.get_bytes(chunk.source_ref),
        )

    loaded = await asyncio.gather(*(load(chunk) for chunk in manifest.chunks))
    return decode_rrweb_chunks(list(loaded))


def _all_findings(artifacts: BehavioralArtifacts) -> list[EvidenceFinding]:
    results = (
        artifacts.video_findings,
        artifacts.screen_findings,
        artifacts.keystroke_findings,
    )
    return [finding for result in results if result for finding in result.findings]


async def run_integrity_review(
    request: ReviewRequest,
    deps: PipelineDeps,
) -> EvidenceBundle:
    """Run one candidate review and return the final validated bundle."""

    timeline = build_master_timeline(request)
    deps.logger.info(
        "integrity-review: timeline built",
        candidate_id=request.candidate_id,
        exam_mode=request.evidence.exam_mode.value,
        artifacts=len(timeline.artifact_registry),
    )

    jobs = await _perception_jobs(request, timeline, deps)
    await asyncio.gather(
        *(process_perception_chunk_job(deps, payload) for payload in jobs)
    )

    camera_bundle, screen_bundle = await asyncio.gather(
        build_perception_bundle(
            deps,
            organization_id=deps.organization_id,
            candidate_id=request.candidate_id,
            assessment_id=request.assessment_id,
            master_timeline=timeline,
        ),
        build_screen_perception_bundle(
            deps,
            organization_id=deps.organization_id,
            candidate_id=request.candidate_id,
            assessment_id=request.assessment_id,
            master_timeline=timeline,
        ),
    )
    rrweb_chunks = (
        await _load_rrweb(request, deps)
        if request.evidence.exam_mode == ExamMode.RRWEB
        else None
    )
    behavioral = run_behavioral_analysis(
        timeline,
        exam_mode=request.evidence.exam_mode,
        rrweb_chunks=rrweb_chunks,
        screen_bundle=(
            screen_bundle
            if request.evidence.exam_mode == ExamMode.SCREEN
            else None
        ),
        perception_bundle=camera_bundle,
        video_windows=_video_windows(camera_bundle.observations),
    )
    findings = _all_findings(behavioral)

    deliberation_input = DeliberationInput(
        candidate_id=request.candidate_id,
        assessment_id=request.assessment_id,
        machine_facts_bundle=behavioral.machine_facts,
        perception_bundle=camera_bundle,
        statistical_baseline=behavioral.baseline,
        master_timeline=timeline,
        correlated_signals=behavioral.correlated_signals,
        video_findings=findings,
        perception_version_hash=compute_perception_version_hash(timeline),
        baseline_version_hash=behavioral.baseline.baseline_version,
    )
    prompt = build_deliberation_prompt(deliberation_input)
    raw_response = await generate_and_log(
        deps,
        step=STEP_DELIBERATION,
        model=DEFAULT_GEMINI_PRO_MODEL,
        model_version=DELIBERATION_PROMPT_VERSION,
        contents=[{"role": "user", "parts": [{"text": prompt}]}],
        config={
            "temperature": 0,
            "maxOutputTokens": 16_384,
            "thinkingConfig": {"thinkingBudget": 1_024},
            "responseMimeType": "application/json",
        },
        extra_meta={},
    )
    raw_text = raw_response.get("text")
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise RuntimeError("Gemini deliberation returned no JSON text")
    deliberation = build_deliberation_bundle(
        deliberation_input,
        raw_llm_text=raw_text,
    )
    bundle = assemble_evidence_bundle(
        EvidenceBundleInput(
            candidate_id=request.candidate_id,
            assessment_id=request.assessment_id,
            deliberation_bundle=deliberation,
            machine_facts_bundle=behavioral.machine_facts,
            perception_bundle=camera_bundle,
            statistical_baseline=behavioral.baseline,
            master_timeline=timeline,
            correlated_signals=behavioral.correlated_signals,
            track_b_findings=findings,
            contextual_events=(
                behavioral.video_findings.contextual_events
                if behavioral.video_findings is not None
                and hasattr(behavioral.video_findings, "contextual_events")
                else []
            ),
            perception_version_hash=compute_perception_version_hash(timeline),
            baseline_version_hash=behavioral.baseline.baseline_version,
        )
    )

    chunks = _chunk_map(request)
    media_index = [
        entry.model_copy(
            update={
                "source_ref": chunks[entry.chunk_id].source_ref,
                "section_id": chunks[entry.chunk_id].section_id,
            }
        )
        if entry.chunk_id in chunks
        else entry
        for entry in bundle.media_index
    ]
    return bundle.model_copy(update={"media_index": media_index})
