"""Backend staged-payload contract and mapping onto the pipeline's ReviewRequest.

The backend builds the authoritative manifest and stages it at
``{s3_media_prefix}/media/ai_integrity_review_requests/{review_id}.json``
(``topin_beta`` or ``topin_prod``); the worker only consumes it.

Robustness policy for this boundary: the payload is produced by another service
from S3 listings and DB rows, so individual fields are routinely absent for
real attempts — a candidate who never opened a section has no
``exam_attempt_id`` and no ``start_datetime``, and a chunk can be missing its
duration. None of that is a reason to fail a whole review, so the models below
accept nulls and unknown keys, and :func:`build_review_request` drops only the
individual items it cannot use, reporting what it dropped so the worker can log
it. Only a payload with no usable identity at all is rejected.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any

from pydantic import ConfigDict, Field, field_validator

from ..contracts.base import ContractModel, to_camel
from ..contracts.evidence import (
    EvidenceChunkRef,
    EvidenceManifest,
    EvidenceType,
    ExamMode,
    ManifestSet,
)
from ..contracts.review import ReviewRequest, SectionInput
from ..timeline.activity_log_timeline import parse_activity_epoch_ms

_MEDIA_TYPE_TO_EVIDENCE_TYPE: dict[str, EvidenceType] = {
    "CAMERA_VIDEO": EvidenceType.VIDEO,
    "SCREEN_VIDEO": EvidenceType.SCREEN_RECORDING,
    "SCREEN_CAMERA_VIDEO": EvidenceType.SCREEN_CAMERA,
    "RRWEB_EVENT": EvidenceType.KEYSTROKE_DATA,
}

_MANIFEST_GROUP: dict[EvidenceType, EvidenceType] = {
    EvidenceType.VIDEO: EvidenceType.VIDEO,
    EvidenceType.SCREEN_RECORDING: EvidenceType.SCREEN_RECORDING,
    EvidenceType.SCREEN_CAMERA: EvidenceType.SCREEN_RECORDING,
    EvidenceType.KEYSTROKE_DATA: EvidenceType.KEYSTROKE_DATA,
}

_STEM_SUFFIX_RE = re.compile(r"\.json\.gz$", re.IGNORECASE)
_STEM_EXT_RE = re.compile(r"\.(?:webm|json)$", re.IGNORECASE)
_STEM_EPOCH_RE = re.compile(r"(\d{10,})")

MAX_REPORTED_WARNINGS = 20


class StagedModel(ContractModel):
    """Inbound half of the contract: ignore unknown keys instead of failing.

    ``extra="forbid"`` is right for the pipeline's own contracts, but here it
    would turn any additive change on the backend into a review-wide failure.
    """

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        extra="ignore",
        validate_assignment=True,
    )


def _blank_to_none(value: Any) -> Any:
    if isinstance(value, str) and not value.strip():
        return None
    return value


def _optional_int(value: Any) -> int | None:
    """Accept ints, numeric strings and floats; treat anything else as absent."""

    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return int(float(text))
        except ValueError:
            return None
    return None


class ManifestChunk(StagedModel):
    chunk_id: str | None = None
    media_type: str | None = None
    exam_attempt_id: str | None = None
    s3_key: str | None = None
    epoch_ms: int | None = None
    duration_ms: int | None = None

    @field_validator("chunk_id", "media_type", "exam_attempt_id", "s3_key", mode="before")
    @classmethod
    def _strip_blanks(cls, value: Any) -> Any:
        return _blank_to_none(value)

    @field_validator("epoch_ms", "duration_ms", mode="before")
    @classmethod
    def _coerce_ints(cls, value: Any) -> Any:
        return _optional_int(value)


class Manifest(StagedModel):
    chunks: list[ManifestChunk] = Field(default_factory=list)

    @field_validator("chunks", mode="before")
    @classmethod
    def _coerce_missing_chunks(cls, value: Any) -> Any:
        return [] if value is None else value


class ActivityLog(StagedModel):
    order: int | None = None
    activity_type: str | None = None
    creation_datetime: str | None = None
    offset_in_seconds: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("activity_type", "creation_datetime", mode="before")
    @classmethod
    def _strip_blanks(cls, value: Any) -> Any:
        return _blank_to_none(value)

    @field_validator("order", mode="before")
    @classmethod
    def _coerce_order(cls, value: Any) -> Any:
        return _optional_int(value)

    @field_validator("metadata", mode="before")
    @classmethod
    def _coerce_missing_metadata(cls, value: Any) -> Any:
        """The backend emits ``null`` for logs that carry no extra detail, and
        older rows store the JSON as a string."""

        if value is None:
            return {}
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                return {}
            return parsed if isinstance(parsed, dict) else {}
        return value if isinstance(value, dict) else {}


class SectionSpec(StagedModel):
    section_id: str | None = None
    exam_id: str | None = None
    order: int | None = None
    exam_attempt_id: str | None = None
    start_datetime: str | None = None
    end_datetime: str | None = None
    section_type: str | None = None

    @field_validator(
        "section_id",
        "exam_id",
        "exam_attempt_id",
        "start_datetime",
        "end_datetime",
        "section_type",
        mode="before",
    )
    @classmethod
    def _strip_blanks(cls, value: Any) -> Any:
        return _blank_to_none(value)

    @field_validator("order", mode="before")
    @classmethod
    def _coerce_order(cls, value: Any) -> Any:
        return _optional_int(value)

    @property
    def attempted(self) -> bool:
        """A section the candidate actually opened has an attempt and a start."""

        return self.exam_attempt_id is not None and self.start_datetime is not None


class ActivityTimeline(StagedModel):
    activity_logs: list[ActivityLog] = Field(default_factory=list)
    sections: list[SectionSpec] = Field(default_factory=list)

    @field_validator("activity_logs", "sections", mode="before")
    @classmethod
    def _coerce_missing_list(cls, value: Any) -> Any:
        return [] if value is None else value


class StagedReviewPayload(StagedModel):
    review_id: str = Field(min_length=1)
    org_assess_id: str = Field(min_length=1)
    attempt_user_id: str = Field(min_length=1)
    manifest: Manifest = Field(default_factory=Manifest)
    activity_timeline: ActivityTimeline = Field(default_factory=ActivityTimeline)
    should_analyse_seb_logs: bool = Field(default=False)
    should_analyse_video: bool = Field(default=False)

    @field_validator("manifest", "activity_timeline", mode="before")
    @classmethod
    def _coerce_missing_section(cls, value: Any) -> Any:
        return {} if value is None else value



class SqsRequestEnvelope(StagedModel):
    message_type: str
    review_id: str = Field(min_length=1)
    payload_s3_key: str = Field(min_length=1)


@dataclass(slots=True)
class PayloadSummary:
    """Enterprise log line for one staged payload: shape, not content."""

    review_id: str
    org_assess_id: str
    attempt_user_id: str
    exam_mode: str
    camera_chunks: int = 0
    screen_chunks: int = 0
    screen_camera_chunks: int = 0
    rrweb_chunks: int = 0
    sections_total: int = 0
    sections_attempted: int = 0
    activity_logs: int = 0
    dropped_chunks: int = 0
    dropped_sections: int = 0
    dropped_activity_logs: int = 0
    warnings: list[str] = field(default_factory=list)

    def add_warning(self, warning: str) -> None:
        if len(self.warnings) < MAX_REPORTED_WARNINGS:
            self.warnings.append(warning)

    def as_log_fields(self) -> dict[str, Any]:
        return {
            "org_assess_id": self.org_assess_id,
            "attempt_user_id": self.attempt_user_id,
            "exam_mode": self.exam_mode,
            "camera_chunks": self.camera_chunks,
            "screen_chunks": self.screen_chunks,
            "screen_camera_chunks": self.screen_camera_chunks,
            "rrweb_chunks": self.rrweb_chunks,
            "sections_total": self.sections_total,
            "sections_attempted": self.sections_attempted,
            "activity_logs": self.activity_logs,
            "dropped_chunks": self.dropped_chunks,
            "dropped_sections": self.dropped_sections,
            "dropped_activity_logs": self.dropped_activity_logs,
            "warnings": self.warnings,
        }

    @property
    def has_degradations(self) -> bool:
        return bool(
            self.warnings
            or self.dropped_chunks
            or self.dropped_sections
            or self.dropped_activity_logs
        )


@dataclass(frozen=True, slots=True)
class NormalizedReviewRequest:
    request: ReviewRequest
    summary: PayloadSummary


def _chunk_id_stem(s3_key: str) -> str:
    """Derive the legacy filename-stem chunk id token from an S3 key's filename."""

    name = PurePosixPath(s3_key).name
    stem = _STEM_SUFFIX_RE.sub("", name)
    stem = _STEM_EXT_RE.sub("", stem)
    return stem or name


def _epoch_from_key(s3_key: str) -> int | None:
    """Recover the upload epoch from the filename when the manifest omits it."""

    match = _STEM_EPOCH_RE.search(_chunk_id_stem(s3_key))
    return int(match.group(1)) if match else None


@dataclass(frozen=True, slots=True)
class _UsableChunk:
    evidence_type: EvidenceType
    exam_attempt_id: str | None
    s3_key: str
    epoch_ms: int
    duration_ms: int | None


def _usable_chunks(manifest: Manifest, summary: PayloadSummary) -> list[_UsableChunk]:
    """Keep every chunk we can actually fetch and place on the timeline."""

    usable: list[_UsableChunk] = []
    seen_keys: set[str] = set()
    unknown_media_types: set[str] = set()

    for chunk in manifest.chunks:
        if not chunk.s3_key:
            summary.dropped_chunks += 1
            summary.add_warning("manifest chunk without s3_key dropped")
            continue
        if chunk.s3_key in seen_keys:
            summary.dropped_chunks += 1
            summary.add_warning(f"duplicate manifest chunk dropped: {chunk.s3_key}")
            continue
        evidence_type = _MEDIA_TYPE_TO_EVIDENCE_TYPE.get(chunk.media_type or "")
        if evidence_type is None:
            summary.dropped_chunks += 1
            if chunk.media_type:
                unknown_media_types.add(chunk.media_type)
            continue
        epoch_ms = chunk.epoch_ms if chunk.epoch_ms and chunk.epoch_ms > 0 else None
        if epoch_ms is None:
            epoch_ms = _epoch_from_key(chunk.s3_key)
        if epoch_ms is None:
            summary.dropped_chunks += 1
            summary.add_warning(f"manifest chunk without a usable epoch dropped: {chunk.s3_key}")
            continue
        seen_keys.add(chunk.s3_key)
        usable.append(
            _UsableChunk(
                evidence_type=evidence_type,
                exam_attempt_id=chunk.exam_attempt_id,
                s3_key=chunk.s3_key,
                epoch_ms=epoch_ms,
                # EvidenceChunkRef rejects negatives; a bad duration is better
                # treated as unknown than as a review-wide validation error.
                duration_ms=(
                    chunk.duration_ms
                    if chunk.duration_ms is not None and chunk.duration_ms >= 0
                    else None
                ),
            )
        )

    if unknown_media_types:
        summary.add_warning(
            f"unsupported media types skipped: {','.join(sorted(unknown_media_types))}"
        )
    return usable


def _build_refs(
    chunks: list[_UsableChunk],
    section_by_attempt: dict[str, SectionSpec],
) -> list[EvidenceChunkRef]:
    # The worker regenerates chunk_id from the S3 key rather than trusting one
    # from the backend manifest; this is only safe because
    # timeline.master_timeline.parse_chunk_id re-derives epoch/duration from a
    # "{epoch}__{duration}" filename stem, so every filename in the manifest
    # must keep carrying that suffix.
    ordered = sorted(chunks, key=lambda item: item.epoch_ms)
    refs: list[EvidenceChunkRef] = []
    for index, chunk in enumerate(ordered):
        section = (
            section_by_attempt.get(chunk.exam_attempt_id)
            if chunk.exam_attempt_id is not None
            else None
        )
        section_id = (
            section.section_id
            if section is not None and section.section_id
            else chunk.exam_attempt_id
        )
        stem = _chunk_id_stem(chunk.s3_key)
        # The prefix follows the chunk's own evidence type, not the manifest
        # it was grouped into, so a combined chunk stays recognisable as
        # "screenCamera-…" everywhere downstream — including on the wire.
        refs.append(
            EvidenceChunkRef(
                evidence_type=chunk.evidence_type,
                chunk_id=f"{chunk.evidence_type.value}-{section_id}-{stem}",
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


def _resolve_exam_mode(
    screen_chunks: list[EvidenceChunkRef],
    rrweb_chunks: list[EvidenceChunkRef],
    summary: PayloadSummary,
) -> ExamMode:
    """A review runs in exactly one exam mode; pick one instead of failing.

    Both kinds of key can legitimately sit under one candidate's session prefix
    (a client upgrade mid-assessment, or a re-attempt on a newer build). The
    mode with more chunks is the one that describes the attempt; the other is
    ignored so the review still produces a report.
    """

    if not (screen_chunks and rrweb_chunks):
        return ExamMode.SCREEN if screen_chunks else ExamMode.RRWEB if rrweb_chunks else ExamMode.NONE

    if len(screen_chunks) > len(rrweb_chunks):
        summary.add_warning(
            f"mixed exam modes: kept {len(screen_chunks)} screen chunks, "
            f"ignored {len(rrweb_chunks)} rrweb chunks"
        )
        summary.dropped_chunks += len(rrweb_chunks)
        return ExamMode.SCREEN

    summary.add_warning(
        f"mixed exam modes: kept {len(rrweb_chunks)} rrweb chunks, "
        f"ignored {len(screen_chunks)} screen chunks"
    )
    summary.dropped_chunks += len(screen_chunks)
    return ExamMode.RRWEB


def _usable_sections(
    sections: list[SectionSpec], summary: PayloadSummary
) -> list[SectionSpec]:
    """Normalize the section specs, keeping unattempted sections as context.

    An unattempted section has no ``exam_attempt_id``/``start_datetime``; it is
    kept so evidence and labels can still be attributed to it, and the timeline
    builder simply produces no span for it.
    """

    usable: list[SectionSpec] = []
    for index, spec in enumerate(sections):
        section_id = spec.section_id or spec.exam_id
        if not section_id:
            summary.dropped_sections += 1
            summary.add_warning("section without section_id or exam_id dropped")
            continue
        start_datetime = spec.start_datetime
        if start_datetime is not None and not _parses_as_datetime(start_datetime):
            summary.add_warning(f"unparseable start_datetime ignored for section {section_id}")
            start_datetime = None
        end_datetime = spec.end_datetime
        if end_datetime is not None and not _parses_as_datetime(end_datetime):
            summary.add_warning(f"unparseable end_datetime ignored for section {section_id}")
            end_datetime = None
        usable.append(
            spec.model_copy(
                update={
                    "section_id": section_id,
                    "order": spec.order if spec.order is not None else index,
                    "start_datetime": start_datetime,
                    "end_datetime": end_datetime,
                }
            )
        )
    return usable


def _parses_as_datetime(value: str) -> bool:
    try:
        parse_activity_epoch_ms(value)
    except (ValueError, TypeError, OverflowError):
        return False
    return True


def _usable_activity_logs(
    logs: list[ActivityLog], summary: PayloadSummary
) -> list[dict[str, Any]]:
    """Drop only the individual log rows the timeline cannot place in time."""

    usable: list[dict[str, Any]] = []
    for index, log in enumerate(logs):
        if not log.creation_datetime or not _parses_as_datetime(log.creation_datetime):
            summary.dropped_activity_logs += 1
            continue
        if not log.activity_type:
            summary.dropped_activity_logs += 1
            continue
        payload = log.model_dump()
        if payload.get("order") is None:
            payload["order"] = index
        usable.append(payload)

    if summary.dropped_activity_logs:
        summary.add_warning(
            f"{summary.dropped_activity_logs} activity log(s) without a usable "
            "timestamp or type were skipped"
        )
    return usable


def _section_by_attempt(
    sections: list[SectionSpec], summary: PayloadSummary
) -> dict[str, SectionSpec]:
    mapping: dict[str, SectionSpec] = {}
    for spec in sections:
        attempt_id = spec.exam_attempt_id
        if attempt_id is None:
            continue
        if attempt_id in mapping:
            summary.add_warning(f"duplicate exam_attempt_id in sections: {attempt_id}")
            continue
        mapping[attempt_id] = spec
    return mapping


def build_review_request(payload: StagedReviewPayload) -> NormalizedReviewRequest:
    """Map a staged backend payload onto the pipeline's ``ReviewRequest``.

    Never raises for partial data: unusable individual chunks, sections and
    activity logs are dropped and counted in the returned summary so the caller
    can log exactly what was degraded.
    """

    summary = PayloadSummary(
        review_id=payload.review_id,
        org_assess_id=payload.org_assess_id,
        attempt_user_id=payload.attempt_user_id,
        exam_mode=ExamMode.NONE.value,
    )

    sections = _usable_sections(payload.activity_timeline.sections, summary)
    summary.sections_total = len(sections)
    summary.sections_attempted = sum(1 for spec in sections if spec.attempted)
    section_by_attempt = _section_by_attempt(sections, summary)

    grouped: dict[EvidenceType, list[_UsableChunk]] = {
        EvidenceType.VIDEO: [],
        EvidenceType.SCREEN_RECORDING: [],
        EvidenceType.KEYSTROKE_DATA: [],
    }
    for chunk in _usable_chunks(payload.manifest, summary):
        grouped[_MANIFEST_GROUP[chunk.evidence_type]].append(chunk)

    video_chunks = _build_refs(grouped[EvidenceType.VIDEO], section_by_attempt)
    screen_chunks = _build_refs(grouped[EvidenceType.SCREEN_RECORDING], section_by_attempt)
    rrweb_chunks = _build_refs(grouped[EvidenceType.KEYSTROKE_DATA], section_by_attempt)

    exam_mode = _resolve_exam_mode(screen_chunks, rrweb_chunks, summary)
    if exam_mode == ExamMode.SCREEN:
        rrweb_chunks = []
    elif exam_mode == ExamMode.RRWEB:
        screen_chunks = []

    summary.exam_mode = exam_mode.value
    summary.camera_chunks = len(video_chunks)
    # The screen manifest holds both plain and combined chunks; report them
    # separately so cost and coverage stay attributable per recording style.
    summary.screen_camera_chunks = sum(
        1 for chunk in screen_chunks if chunk.evidence_type == EvidenceType.SCREEN_CAMERA
    )
    summary.screen_chunks = len(screen_chunks) - summary.screen_camera_chunks
    summary.rrweb_chunks = len(rrweb_chunks)

    if not video_chunks and not screen_chunks and not rrweb_chunks:
        summary.add_warning("manifest carries no usable evidence chunks")

    evidence = ManifestSet(
        exam_mode=exam_mode,
        manifests=[
            _manifest(EvidenceType.VIDEO, video_chunks),
            _manifest(EvidenceType.SCREEN_RECORDING, screen_chunks),
            _manifest(EvidenceType.KEYSTROKE_DATA, rrweb_chunks),
        ],
    )

    section_inputs = [
        SectionInput(
            section_id=spec.section_id,
            exam_attempt_id=spec.exam_attempt_id,
            exam_id=spec.exam_id,
            order=spec.order,
            start_datetime=spec.start_datetime,
            end_datetime=spec.end_datetime,
            section_type=spec.section_type,
        )
        for spec in sorted(sections, key=lambda item: item.order or 0)
    ]

    activity_timeline = _usable_activity_logs(payload.activity_timeline.activity_logs, summary)
    summary.activity_logs = len(activity_timeline)
    if not activity_timeline:
        summary.add_warning(
            "no usable activity logs; session timeline falls back to chunk epochs"
        )

    request = ReviewRequest(
        candidate_id=payload.attempt_user_id,
        assessment_id=payload.org_assess_id,
        activity_timeline=activity_timeline,
        sections=section_inputs,
        evidence=evidence,
    )
    return NormalizedReviewRequest(request=request, summary=summary)
