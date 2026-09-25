from __future__ import annotations

import asyncio
import time
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
from .prompts import DELIBERATION_MODEL_VERSION
from .prompts.deliberation import DELIBERATION_PROMPT_VERSION
from .timeline import build_master_timeline

DELIBERATION_ATTEMPTS = 2


def _chunk_map(request: ReviewRequest) -> dict[str, Any]:
    return {
        chunk.chunk_id: chunk
        for manifest in request.evidence.manifests
        for chunk in manifest.chunks
    }


def _job_evidence_type(evidence_type: EvidenceType) -> str:
    if evidence_type == EvidenceType.VIDEO:
        return "video"
    if evidence_type == EvidenceType.SCREEN_CAMERA:
        return "screenCamera"
    return "screenRecording"


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
            evidence_type=_job_evidence_type(chunk.evidence_type),
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
    best_by_window: dict[str, PerceptionObservation] = {}
    for observation in observations:
        current = best_by_window.get(observation.window_id)
        if current is None or observation.confidence > current.confidence:
            best_by_window[observation.window_id] = observation

    windows: list[VideoObservationWindow] = []
    for observation in best_by_window.values():
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

    results = await asyncio.gather(
        *(load(chunk) for chunk in manifest.chunks), return_exceptions=True
    )
    payloads: list[tuple[str, int, bytes]] = []
    failed = 0
    for chunk, result in zip(manifest.chunks, results, strict=True):
        if isinstance(result, BaseException):
            failed += 1
            deps.logger.warning(
                "integrity-review: rrweb chunk fetch failed, skipping",
                chunk_id=chunk.chunk_id,
                source_ref=chunk.source_ref,
                error=result,
            )
            continue
        payloads.append(result)

    decoded: list[Any] = []
    undecodable = 0
    for payload in payloads:
        try:
            decoded.extend(decode_rrweb_chunks([payload]))
        except Exception as error:  # noqa: BLE001 - one bad chunk, not a bad review
            undecodable += 1
            deps.logger.warning(
                "integrity-review: rrweb chunk undecodable, skipping",
                chunk_id=payload[0],
                error=error,
            )

    deps.logger.info(
        "integrity-review: rrweb chunks loaded",
        total=len(manifest.chunks),
        loaded=len(decoded),
        fetch_failed=failed,
        undecodable=undecodable,
        events=sum(len(chunk.events) for chunk in decoded),
    )
    return decoded


def _all_findings(artifacts: BehavioralArtifacts) -> list[EvidenceFinding]:
    results = (
        artifacts.video_findings,
        artifacts.screen_findings,
        artifacts.keystroke_findings,
    )
    return [finding for result in results if result for finding in result.findings]


async def _run_perception_jobs(
    jobs: list[PerceptionChunkJobPayload], deps: PipelineDeps
) -> None:
    if not jobs:
        deps.logger.info("integrity-review: no perception chunks to analyse")
        return

    started = time.perf_counter()
    results = await asyncio.gather(
        *(process_perception_chunk_job(deps, payload) for payload in jobs),
        return_exceptions=True,
    )
    failures = 0
    for payload, result in zip(jobs, results, strict=True):
        if not isinstance(result, BaseException):
            continue
        failures += 1
        deps.logger.warning(
            "integrity-review: perception chunk failed, continuing without it",
            chunk_id=payload.chunk_id,
            evidence_type=payload.evidence_type,
            section_id=payload.section_id,
            error=result,
        )

    log = deps.logger.warning if failures else deps.logger.info
    log(
        "integrity-review: perception stage complete",
        chunks=len(jobs),
        analysed=len(jobs) - failures,
        failed=failures,
        elapsed_seconds=round(time.perf_counter() - started, 1),
    )


async def _deliberate(
    deliberation_input: DeliberationInput, prompt: str, deps: PipelineDeps
) -> Any:
    started = time.perf_counter()
    deps.logger.info(
        "integrity-review: deliberation started",
        model=DELIBERATION_MODEL_VERSION,
        prompt_version=DELIBERATION_PROMPT_VERSION,
        prompt_chars=len(prompt),
    )
    for attempt in range(1, DELIBERATION_ATTEMPTS + 1):
        raw_response = await generate_and_log(
            deps,
            step=STEP_DELIBERATION,
            model=DELIBERATION_MODEL_VERSION,
            model_version=DELIBERATION_PROMPT_VERSION,
            contents=[{"role": "user", "parts": [{"text": prompt}]}],
            config={
                "temperature": 0,
                "maxOutputTokens": 16_384,
                "thinkingConfig": {"thinkingBudget": 2_048},
                "responseMimeType": "application/json",
            },
            extra_meta={"attempt": attempt},
        )
        raw_text = raw_response.get("text")
        if isinstance(raw_text, str) and raw_text.strip():
            deps.logger.info(
                "integrity-review: deliberation response received",
                attempt=attempt,
                elapsed_seconds=round(time.perf_counter() - started, 1),
                response_chars=len(raw_text),
                usage=raw_response.get("usage"),
            )
            try:
                return build_deliberation_bundle(deliberation_input, raw_llm_text=raw_text)
            except ValueError as error:
                reason = f"unparseable JSON ({error})"
        else:
            reason = "empty response"

        if attempt == DELIBERATION_ATTEMPTS:
            raise RuntimeError(f"Gemini deliberation failed after {attempt} attempts: {reason}")
        deps.logger.warning(
            "integrity-review: deliberation response unusable, retrying",
            attempt=attempt,
            attempts=DELIBERATION_ATTEMPTS,
            reason=reason,
        )
    raise RuntimeError("Gemini deliberation failed")  # pragma: no cover - loop always returns


async def run_integrity_review(
    request: ReviewRequest,
    deps: PipelineDeps,
) -> EvidenceBundle:
    review_started = time.perf_counter()
    timeline = build_master_timeline(request)
    deps.logger.info(
        "integrity-review: timeline built",
        candidate_id=request.candidate_id,
        exam_mode=request.evidence.exam_mode.value,
        artifacts=len(timeline.artifact_registry),
        sections=len(timeline.sections),
        duration_ms=timeline.duration_ms,
        t0_source=getattr(timeline.sync_report, "t0_source", None),
        duration_source=getattr(timeline.sync_report, "duration_source", None),
        activity_logs=len(request.activity_timeline),
    )

    jobs = await _perception_jobs(request, timeline, deps)
    deps.logger.info(
        "integrity-review: perception stage started",
        chunks=len(jobs),
        camera_chunks=sum(1 for job in jobs if job.evidence_type == "video"),
        screen_chunks=sum(1 for job in jobs if job.evidence_type == "screenRecording"),
        screen_camera_chunks=sum(1 for job in jobs if job.evidence_type == "screenCamera"),
    )
    await _run_perception_jobs(jobs, deps)

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
    deps.logger.info(
        "integrity-review: perception bundles built",
        camera_observations=len(camera_bundle.observations),
        screen_observations=len(screen_bundle.observations),
    )

    rrweb_chunks = (
        await _load_rrweb(request, deps)
        if request.evidence.exam_mode == ExamMode.RRWEB
        else None
    )
    behavioral_started = time.perf_counter()
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
    deps.logger.info(
        "integrity-review: behavioral analysis complete",
        elapsed_seconds=round(time.perf_counter() - behavioral_started, 1),
        findings=len(findings),
        risk_contributions=len(behavioral.correlated_signals.contributions),
        candidate_chains=len(behavioral.correlated_signals.chains),
        elapsed_since_start_seconds=round(time.perf_counter() - review_started, 1),
    )

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
    deliberation = await _deliberate(deliberation_input, prompt, deps)
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
    deps.logger.info(
        "integrity-review: bundle assembled",
        elapsed_seconds=round(time.perf_counter() - review_started, 1),
        recommendation_category=str(bundle.recommendation.category),
        detected_signals=len(bundle.detected_signals.signals),
        media_index_entries=len(media_index),
    )
    return bundle.model_copy(update={"media_index": media_index})
