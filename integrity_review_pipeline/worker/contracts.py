"""Backend staged-payload contract and mapping onto the pipeline's ReviewRequest.

The backend builds the authoritative manifest and stages it at
``{STAGE}/media/ai_integrity_review_requests/{review_id}.json``; the worker only
consumes it. See the plan for the frozen payload shape.
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Any

from pydantic import Field

from ..contracts.base import ContractModel
from ..contracts.evidence import (
    EvidenceChunkRef,
    EvidenceManifest,
    EvidenceType,
    ExamMode,
    ManifestSet,
)
from ..contracts.review import ReviewRequest, SectionInput

_MEDIA_TYPE_TO_EVIDENCE_TYPE: dict[str, EvidenceType] = {
    "CAMERA_VIDEO": EvidenceType.VIDEO,
    "SCREEN_VIDEO": EvidenceType.SCREEN_RECORDING,
    "RRWEB_EVENT": EvidenceType.KEYSTROKE_DATA,
}

_STEM_SUFFIX_RE = re.compile(r"\.json\.gz$", re.IGNORECASE)
_STEM_EXT_RE = re.compile(r"\.(?:webm|json)$", re.IGNORECASE)


class ManifestChunk(ContractModel):
    chunk_id: str = Field(min_length=1)
    media_type: str
    exam_attempt_id: str = Field(min_length=1)
    s3_key: str = Field(min_length=1)
    epoch_ms: int
    duration_ms: int | None = None


class Manifest(ContractModel):
    chunks: list[ManifestChunk] = Field(default_factory=list)


class ActivityLog(ContractModel):
    order: int
    activity_type: str
    creation_datetime: str
    offset_in_seconds: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SectionSpec(ContractModel):
    section_id: str = Field(min_length=1)
    exam_id: str = Field(min_length=1)
    order: int
    exam_attempt_id: str = Field(min_length=1)
    start_datetime: str
    end_datetime: str | None = None


class ActivityTimeline(ContractModel):
    activity_logs: list[ActivityLog] = Field(default_factory=list)
    sections: list[SectionSpec] = Field(default_factory=list)


class StagedReviewPayload(ContractModel):
    review_id: str = Field(min_length=1)
    org_assess_id: str = Field(min_length=1)
    attempt_user_id: str = Field(min_length=1)
    manifest: Manifest
    activity_timeline: ActivityTimeline


class SqsRequestEnvelope(ContractModel):
    message_type: str
    review_id: str = Field(min_length=1)
    payload_s3_key: str = Field(min_length=1)


def _chunk_id_stem(s3_key: str) -> str:
    """Derive the legacy filename-stem chunk id token from an S3 key's filename."""

    name = PurePosixPath(s3_key).name
    stem = _STEM_SUFFIX_RE.sub("", name)
    stem = _STEM_EXT_RE.sub("", stem)
    return stem or name


def _build_refs(
    evidence_type: EvidenceType,
    chunks: list[ManifestChunk],
    section_by_attempt: dict[str, SectionSpec],
) -> list[EvidenceChunkRef]:
    ordered = sorted(chunks, key=lambda item: item.epoch_ms)
    refs: list[EvidenceChunkRef] = []
    for index, chunk in enumerate(ordered):
        section = section_by_attempt.get(chunk.exam_attempt_id)
        section_id = section.section_id if section is not None else chunk.exam_attempt_id
        stem = _chunk_id_stem(chunk.s3_key)
        refs.append(
            EvidenceChunkRef(
                evidence_type=evidence_type,
                chunk_id=f"{evidence_type.value}-{section_id}-{stem}",
                sequence=index,
                source_ref=chunk.s3_key,
                duration_ms=chunk.duration_ms,
                section_id=section_id,
            )
        )
    return refs


def _manifest(evidence_type: EvidenceType, chunks: list[EvidenceChunkRef]) -> EvidenceManifest:
    durations = [chunk.duration_ms for chunk in chunks if chunk.duration_ms is not None]
    return EvidenceManifest(
        evidence_type=evidence_type,
        total_chunks=len(chunks),
        chunks=chunks,
        total_duration_ms=sum(durations) if durations else None,
    )


def build_review_request(payload: StagedReviewPayload) -> ReviewRequest:
    """Map a staged backend payload onto the pipeline's ``ReviewRequest`` contract."""

    section_by_attempt = {
        spec.exam_attempt_id: spec for spec in payload.activity_timeline.sections
    }

    grouped: dict[EvidenceType, list[ManifestChunk]] = {
        EvidenceType.VIDEO: [],
        EvidenceType.SCREEN_RECORDING: [],
        EvidenceType.KEYSTROKE_DATA: [],
    }
    for chunk in payload.manifest.chunks:
        evidence_type = _MEDIA_TYPE_TO_EVIDENCE_TYPE.get(chunk.media_type)
        if evidence_type is None:
            continue
        grouped[evidence_type].append(chunk)

    video_chunks = _build_refs(EvidenceType.VIDEO, grouped[EvidenceType.VIDEO], section_by_attempt)
    screen_chunks = _build_refs(
        EvidenceType.SCREEN_RECORDING, grouped[EvidenceType.SCREEN_RECORDING], section_by_attempt
    )
    rrweb_chunks = _build_refs(
        EvidenceType.KEYSTROKE_DATA, grouped[EvidenceType.KEYSTROKE_DATA], section_by_attempt
    )

    if screen_chunks and rrweb_chunks:
        raise ValueError(
            "manifest contains both SCREEN_VIDEO and RRWEB_EVENT chunks; "
            "a single review must use exactly one exam mode"
        )
    exam_mode = (
        ExamMode.SCREEN if screen_chunks else ExamMode.RRWEB if rrweb_chunks else ExamMode.NONE
    )

    evidence = ManifestSet(
        exam_mode=exam_mode,
        manifests=[
            _manifest(EvidenceType.VIDEO, video_chunks),
            _manifest(EvidenceType.SCREEN_RECORDING, screen_chunks),
            _manifest(EvidenceType.KEYSTROKE_DATA, rrweb_chunks),
        ],
    )

    sections = [
        SectionInput(
            section_id=spec.section_id,
            exam_attempt_id=spec.exam_attempt_id,
            exam_id=spec.exam_id,
            order=spec.order,
            start_datetime=spec.start_datetime,
            end_datetime=spec.end_datetime,
        )
        for spec in sorted(payload.activity_timeline.sections, key=lambda item: item.order)
    ]

    activity_timeline = [log.model_dump() for log in payload.activity_timeline.activity_logs]

    return ReviewRequest(
        candidate_id=payload.attempt_user_id,
        assessment_id=payload.org_assess_id,
        activity_timeline=activity_timeline,
        sections=sections,
        evidence=evidence,
    )
