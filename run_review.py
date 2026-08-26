#!/usr/bin/env python3
"""One-shot local runner: input.json (signed URLs) -> output/evidence_bundle.json.

Runs the real pipeline with the live Gemini client and HTTPS media fetching.
No SQS/S3/ECS involved. Requires GEMINI_API_KEY (and network access to the
signed URLs referenced by input.json).
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import unquote, urlparse

import httpx
from pydantic import Field, field_validator

from integrity_review_pipeline.adapters import (
    GoogleGeminiClient,
    InMemoryPerceptionCache,
    StructuredLogger,
)
from integrity_review_pipeline.contracts.base import ContractModel
from integrity_review_pipeline.contracts.evidence import (
    EvidenceChunkRef,
    EvidenceManifest,
    EvidenceType,
    ExamMode,
    ManifestSet,
)
from integrity_review_pipeline.contracts.review import ReviewRequest, SectionInput
from integrity_review_pipeline.deps import PipelineDeps
from integrity_review_pipeline.pipeline import run_integrity_review

_DURATION_RE = re.compile(r"__(\d+)(?:__events|__metadata)?\.(?:webm|json(?:\.gz)?)$", re.IGNORECASE)


class ReviewInputSection(ContractModel):
    exam_attempt_id: str = Field(min_length=1)
    exam_id: str = Field(min_length=1)
    section_type: str | None = None
    title: str | None = None

    @property
    def section_id(self) -> str:
        if self.section_type:
            return self.section_type.lower()
        if self.title:
            return "-".join(self.title.lower().split())
        return self.exam_attempt_id[:8]


class ReviewInput(ContractModel):
    """Exact shape accepted from ``input/input.json``."""

    candidate_id: str = Field(min_length=1)
    assessment_id: str = Field(min_length=1)
    camera_recordings: list[str] = Field(min_length=1)
    session_recordings: list[str] = Field(default_factory=list)
    activity_timeline: list[dict[str, Any]] = Field(min_length=1)
    sections: list[ReviewInputSection] = Field(min_length=1)

    @field_validator("camera_recordings", "session_recordings")
    @classmethod
    def _require_https(cls, urls: list[str]) -> list[str]:
        bad = [url for url in urls if not url.startswith("https://")]
        if bad:
            raise ValueError(f"recording URLs must use https: {bad}")
        return urls


def _url_path(url: str) -> str:
    return unquote(urlparse(url).path)


def _duration_ms(url: str) -> int | None:
    match = _DURATION_RE.search(PurePosixPath(_url_path(url)).name)
    return int(match.group(1)) if match else None


def _chunk_id(prefix: str, url: str) -> str:
    name = PurePosixPath(_url_path(url)).name
    stem = re.sub(r"\.json\.gz$", "", name, flags=re.IGNORECASE)
    stem = re.sub(r"\.(?:webm|json)$", "", stem, flags=re.IGNORECASE)
    if stem:
        return f"{prefix}-{stem}"
    return f"{prefix}-{hashlib.sha256(url.encode()).hexdigest()[:16]}"


def _classify_session(url: str) -> str:
    path = _url_path(url)
    name = PurePosixPath(path).name.lower()
    if name.endswith("__events.json.gz"):
        return "sidecar_events"
    if name.endswith("__metadata.json.gz"):
        return "sidecar_metadata"
    if "/v3/" in path and name.endswith(".webm"):
        return "screen_webm"
    if re.search(r"__\d+\.json$", name):
        return "screen_json"
    if name.endswith(".json.gz") or name.endswith(".json"):
        return "rrweb_json"
    if name.endswith(".webm"):
        return "screen_webm"
    return "unknown"


def _section_for_url(url: str, sections: list[ReviewInputSection]) -> ReviewInputSection:
    parts = set(PurePosixPath(_url_path(url)).parts)
    for section in sections:
        if section.exam_attempt_id in parts or section.exam_id in parts:
            return section
    if len(sections) == 1:
        return sections[0]
    raise ValueError(f"could not map recording URL to a section: {url}")


def _chunk(
    evidence_type: EvidenceType,
    url: str,
    sequence: int,
    section: ReviewInputSection,
    *,
    sidecar_role: str | None = None,
    parent_chunk_stem: str | None = None,
) -> EvidenceChunkRef:
    return EvidenceChunkRef(
        evidence_type=evidence_type,
        chunk_id=_chunk_id(f"{evidence_type.value}-{section.section_id}", url),
        sequence=sequence,
        source_ref=url,
        signed_url=url,
        duration_ms=_duration_ms(url),
        section_id=section.section_id,
        sidecar_role=sidecar_role,
        parent_chunk_stem=parent_chunk_stem,
    )


def _manifest(evidence_type: EvidenceType, chunks: list[EvidenceChunkRef]) -> EvidenceManifest:
    durations = [chunk.duration_ms for chunk in chunks if chunk.duration_ms is not None]
    return EvidenceManifest(
        evidence_type=evidence_type,
        total_chunks=len(chunks),
        chunks=chunks,
        total_duration_ms=sum(durations) if durations else None,
    )


def build_manifest_set(review_input: ReviewInput) -> ManifestSet:
    video_chunks = [
        _chunk(EvidenceType.VIDEO, url, index, _section_for_url(url, review_input.sections))
        for index, url in enumerate(review_input.camera_recordings)
    ]

    rrweb_chunks: list[EvidenceChunkRef] = []
    screen_chunks: list[EvidenceChunkRef] = []
    unknown_urls: list[str] = []
    for url in review_input.session_recordings:
        kind = _classify_session(url)
        section = _section_for_url(url, review_input.sections)
        if kind == "rrweb_json":
            rrweb_chunks.append(_chunk(EvidenceType.KEYSTROKE_DATA, url, len(rrweb_chunks), section))
        elif kind in ("screen_webm", "screen_json"):
            screen_chunks.append(_chunk(EvidenceType.SCREEN_RECORDING, url, len(screen_chunks), section))
        elif kind in ("sidecar_events", "sidecar_metadata"):
            name = PurePosixPath(_url_path(url)).name
            stem_match = re.match(r"^(\d{13}__\d+)", name)
            screen_chunks.append(
                _chunk(
                    EvidenceType.SCREEN_RECORDING,
                    url,
                    len(screen_chunks),
                    section,
                    sidecar_role="events" if kind == "sidecar_events" else "metadata",
                    parent_chunk_stem=stem_match.group(1) if stem_match else None,
                )
            )
        else:
            unknown_urls.append(url)

    if unknown_urls:
        raise ValueError(f"unrecognized session recording URLs: {unknown_urls}")

    screen_media = [chunk for chunk in screen_chunks if chunk.sidecar_role is None]
    if screen_media and rrweb_chunks:
        raise ValueError("session_recordings must contain either screen media or rrweb JSON, not both")

    exam_mode = ExamMode.SCREEN if screen_media else ExamMode.RRWEB if rrweb_chunks else ExamMode.NONE
    return ManifestSet(
        exam_mode=exam_mode,
        manifests=[
            _manifest(EvidenceType.VIDEO, video_chunks),
            _manifest(EvidenceType.SCREEN_RECORDING, screen_chunks),
            _manifest(EvidenceType.KEYSTROKE_DATA, rrweb_chunks),
        ],
    )


def load_review_request(path: Path) -> ReviewRequest:
    review_input = ReviewInput.model_validate(json.loads(path.read_text(encoding="utf-8")))
    sections = [
        SectionInput(
            section_id=section.section_id,
            exam_attempt_id=section.exam_attempt_id,
            exam_id=section.exam_id,
            section_type=section.section_type,
            title=section.title,
        )
        for section in review_input.sections
    ]
    return ReviewRequest(
        candidate_id=review_input.candidate_id,
        assessment_id=review_input.assessment_id,
        activity_timeline=review_input.activity_timeline,
        sections=sections,
        evidence=build_manifest_set(review_input),
    )


class SignedUrlObjectStore:
    """Fetches rrweb/legacy-screen JSON directly over HTTPS from signed URLs."""

    def __init__(self, timeout_seconds: float = 120.0) -> None:
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(timeout_seconds), follow_redirects=True)

    async def get_bytes(self, ref: str) -> bytes:
        response = await self._client.get(ref)
        response.raise_for_status()
        return response.content

    async def get_json(self, ref: str) -> Any:
        return json.loads((await self.get_bytes(ref)).decode("utf-8"))


class SignedUrlMediaProvider:
    async def media_uri_for(self, chunk: EvidenceChunkRef) -> str:
        return chunk.source_ref


class SemaphoreLimiter:
    def __init__(self, slots: int) -> None:
        self._semaphore = asyncio.Semaphore(slots)

    async def __aenter__(self) -> None:
        await self._semaphore.acquire()

    async def __aexit__(self, *_exc: object) -> None:
        self._semaphore.release()


async def run(input_path: Path, output_path: Path) -> None:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is required")

    request = load_review_request(input_path)
    deps = PipelineDeps(
        object_store=SignedUrlObjectStore(),
        gemini=GoogleGeminiClient(api_key),
        cache=InMemoryPerceptionCache(),
        limiter=SemaphoreLimiter(int(os.environ.get("GEMINI_TASK_LIMIT", "12"))),
        logger=StructuredLogger(f"{request.assessment_id}:{request.candidate_id}"),
        media_uri_provider=SignedUrlMediaProvider(),
        organization_id=os.environ.get("ORGANIZATION_ID", "local"),
    )
    bundle = await run_integrity_review(request, deps)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = bundle.model_dump(mode="json", by_alias=True, exclude_none=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    temporary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary_path.replace(output_path)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "input" / "input.json",
        help="Path to signed-URL input JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "output" / "evidence_bundle.json",
        help="Path for the final EvidenceBundle JSON",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    asyncio.run(run(args.input, args.output))
    print(f"EvidenceBundle written to {args.output}")


if __name__ == "__main__":
    main()
