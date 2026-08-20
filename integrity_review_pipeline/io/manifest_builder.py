"""Build evidence manifests from signed recording URLs.

This is the framework-free Python equivalent of ``topinEvidenceSourceClient.ts``.
"""

from __future__ import annotations

import hashlib
import re
from enum import StrEnum
from pathlib import PurePosixPath
from urllib.parse import unquote, urlparse

from ..contracts.evidence import (
    EvidenceChunkRef,
    EvidenceManifest,
    EvidenceType,
    ExamMode,
    ManifestSet,
)
from ..contracts.review import ReviewInput, SectionInput


class SessionUrlKind(StrEnum):
    RRWEB_JSON = "rrweb_json"
    SCREEN_WEBM = "screen_webm"
    SCREEN_JSON = "screen_json"
    SIDECAR_EVENTS = "sidecar_events"
    SIDECAR_METADATA = "sidecar_metadata"
    UNKNOWN = "unknown"


_DURATION_RE = re.compile(
    r"__(\d+)(?:__events|__metadata)?\.(?:webm|json(?:\.gz)?)$", re.IGNORECASE
)


def _url_path(url: str) -> str:
    return unquote(urlparse(url).path)


def classify_session_url(url: str) -> SessionUrlKind:
    path = _url_path(url)
    name = PurePosixPath(path).name
    if re.search(r"__events\.json\.gz$", name, re.IGNORECASE):
        return SessionUrlKind.SIDECAR_EVENTS
    if re.search(r"__metadata\.json\.gz$", name, re.IGNORECASE):
        return SessionUrlKind.SIDECAR_METADATA
    if "/v3/" in path and name.lower().endswith(".webm"):
        return SessionUrlKind.SCREEN_WEBM
    if re.search(r"__\d+\.json$", name, re.IGNORECASE):
        return SessionUrlKind.SCREEN_JSON
    if re.fullmatch(r"\d+\.json\.gz", name, re.IGNORECASE):
        return SessionUrlKind.RRWEB_JSON
    if name.lower().endswith(".json") and not name.lower().endswith(".json.gz"):
        return SessionUrlKind.RRWEB_JSON
    if name.lower().endswith(".webm"):
        return SessionUrlKind.SCREEN_WEBM
    return SessionUrlKind.UNKNOWN


def parse_duration_ms(url: str) -> int | None:
    match = _DURATION_RE.search(PurePosixPath(_url_path(url)).name)
    return int(match.group(1)) if match else None


def stable_chunk_id(prefix: str, url: str) -> str:
    name = PurePosixPath(_url_path(url)).name
    stem = re.sub(r"\.json\.gz$", "", name, flags=re.IGNORECASE)
    stem = re.sub(r"\.(?:webm|json)$", "", stem, flags=re.IGNORECASE)
    if stem:
        return f"{prefix}-{stem}"
    digest = hashlib.sha256(url.encode()).hexdigest()[:16]
    return f"{prefix}-{digest}"


def _section_for_url(url: str, sections: list[SectionInput]) -> SectionInput:
    path_parts = set(PurePosixPath(_url_path(url)).parts)
    for section in sections:
        if section.exam_attempt_id in path_parts or section.exam_id in path_parts:
            return section
    if len(sections) == 1:
        return sections[0]
    raise ValueError(
        "could not map recording URL to a section; URL path must contain "
        "examAttemptId or examId when multiple sections are supplied"
    )


def _chunk(
    evidence_type: EvidenceType,
    url: str,
    sequence: int,
    section: SectionInput,
    *,
    sidecar_role: str | None = None,
    parent_chunk_stem: str | None = None,
) -> EvidenceChunkRef:
    return EvidenceChunkRef(
        evidence_type=evidence_type,
        chunk_id=stable_chunk_id(f"{evidence_type.value}-{section.section_id}", url),
        sequence=sequence,
        signed_url=url,
        duration_ms=parse_duration_ms(url),
        section_id=section.section_id,
        sidecar_role=sidecar_role,
        parent_chunk_stem=parent_chunk_stem,
    )


def _manifest(
    evidence_type: EvidenceType, chunks: list[EvidenceChunkRef]
) -> EvidenceManifest:
    durations = [chunk.duration_ms for chunk in chunks if chunk.duration_ms is not None]
    return EvidenceManifest(
        evidence_type=evidence_type,
        total_chunks=len(chunks),
        chunks=chunks,
        total_duration_ms=sum(durations) if durations else None,
    )


def build_manifest_set(review_input: ReviewInput) -> ManifestSet:
    video_chunks = [
        _chunk(
            EvidenceType.VIDEO,
            url,
            sequence,
            _section_for_url(url, review_input.sections),
        )
        for sequence, url in enumerate(review_input.camera_recordings)
    ]

    rrweb_chunks: list[EvidenceChunkRef] = []
    screen_chunks: list[EvidenceChunkRef] = []
    unknown_urls: list[str] = []
    for url in review_input.session_recordings:
        kind = classify_session_url(url)
        section = _section_for_url(url, review_input.sections)
        if kind == SessionUrlKind.RRWEB_JSON:
            rrweb_chunks.append(
                _chunk(EvidenceType.KEYSTROKE_DATA, url, len(rrweb_chunks), section)
            )
        elif kind in (SessionUrlKind.SCREEN_WEBM, SessionUrlKind.SCREEN_JSON):
            screen_chunks.append(
                _chunk(EvidenceType.SCREEN_RECORDING, url, len(screen_chunks), section)
            )
        elif kind in (
            SessionUrlKind.SIDECAR_EVENTS,
            SessionUrlKind.SIDECAR_METADATA,
        ):
            name = PurePosixPath(_url_path(url)).name
            stem_match = re.match(r"^(\d{13}__\d+)", name)
            screen_chunks.append(
                _chunk(
                    EvidenceType.SCREEN_RECORDING,
                    url,
                    len(screen_chunks),
                    section,
                    sidecar_role=(
                        "events"
                        if kind == SessionUrlKind.SIDECAR_EVENTS
                        else "metadata"
                    ),
                    parent_chunk_stem=stem_match.group(1) if stem_match else None,
                )
            )
        else:
            unknown_urls.append(url)

    if unknown_urls:
        raise ValueError(f"unrecognized session recording URLs: {unknown_urls}")

    screen_media = [chunk for chunk in screen_chunks if chunk.sidecar_role is None]
    if screen_media and rrweb_chunks:
        raise ValueError(
            "session_recordings must contain either screen media or rrweb JSON, not both"
        )

    exam_mode = (
        ExamMode.SCREEN
        if screen_media
        else ExamMode.RRWEB
        if rrweb_chunks
        else ExamMode.NONE
    )
    return ManifestSet(
        exam_mode=exam_mode,
        manifests=[
            _manifest(EvidenceType.VIDEO, video_chunks),
            _manifest(EvidenceType.SCREEN_RECORDING, screen_chunks),
            _manifest(EvidenceType.KEYSTROKE_DATA, rrweb_chunks),
        ],
    )
