#!/usr/bin/env python3
"""Local one-shot runner: raw signed S3 URLs -> output/evidence_bundle.json.

Temporary local-testing harness for the v3 combined recording style, where one
``.webm`` chunk carries *both* the screen and the camera. It deliberately goes
through the production mapping in
``nw_assessments_ai_analysis_backend/worker/contracts.py``
(``StagedReviewPayload`` -> ``build_review_request``) so the chunk ids, manifest
grouping, exam-mode resolution and ``screenCamera`` perception path are exactly
the ones the ECS worker would take. Nothing here touches SQS, S3 or IAM: the
signed URLs given in ``input/input.json`` are used verbatim.

Usage
-----
    # 0. one-off: cache the media locally so later runs survive URL expiry
    .venv/bin/python run_review.py --download

    # 1. no Gemini calls at all: check ingest, timeline and section mapping
    .venv/bin/python run_review.py --dry-run

    # 2. the real AI flow
    GEMINI_API_KEY=... .venv/bin/python run_review.py
"""
 
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import unquote, urlparse

import httpx
from nw_assessments_ai_analysis_backend.adapters import (
    GoogleGeminiClient,
    OpenRouterClient,
    InMemoryPerceptionCache,
    NullAiUsageLogger,
    StructuredLogger,
    configure_logging,
)
from nw_assessments_ai_analysis_backend.contracts.evidence import EvidenceChunkRef
from nw_assessments_ai_analysis_backend.deps import PipelineDeps
from nw_assessments_ai_analysis_backend.media.perception_media import (
    build_parts_from_inline,
    build_parts_from_url,
    decode_screen_recording,
    is_ebml_magic,
    maybe_gunzip,
)
from nw_assessments_ai_analysis_backend.deliberation import rules as _rules
from nw_assessments_ai_analysis_backend.perception import chunk_job as _chunk_job
from nw_assessments_ai_analysis_backend import pipeline as _pipeline
from nw_assessments_ai_analysis_backend.pipeline import run_integrity_review
from nw_assessments_ai_analysis_backend.timeline import build_master_timeline
from nw_assessments_ai_analysis_backend.worker.contracts import (
    NormalizedReviewRequest,
    StagedReviewPayload,
    build_review_request,
)

ROOT = Path(__file__).resolve().parent

# "{upload_epoch_ms}[__{duration_ms}]" — the stem contract the whole pipeline
# relies on (see worker/contracts._build_refs and timeline.parse_chunk_id).
# rrweb batches carry only the epoch; media chunks carry both.
STEM_RE = re.compile(r"^(?P<epoch>\d{13})(?:__(?P<duration>\d+))?$")

MEDIA_TYPE_CHOICES = ("SCREEN_CAMERA_VIDEO", "SCREEN_VIDEO", "CAMERA_VIDEO")

# Captured before anything patches it, so --media-delivery auto really does
# restore the shipped presigned-fileUri path.
STOCK_RESOLVE_MEDIA_PARTS = _chunk_job.resolve_media_parts


# --------------------------------------------------------------------------
# 1. Signed URL -> manifest chunk
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ChunkUrl:
    """One entry of ``chunk_urls``, classified by its filename."""

    url: str
    s3_key: str
    filename: str
    stem: str
    role: str  # "media" | "rrweb" | "metadata" | "events" | "unknown"
    media_type: str | None  # the manifest mediaType, for role in {media, rrweb}
    epoch_ms: int | None
    duration_ms: int | None

    @property
    def start_ms(self) -> int | None:
        if self.epoch_ms is None or self.duration_ms is None:
            return None
        return self.epoch_ms - self.duration_ms


def classify_url(url: str) -> ChunkUrl:
    """Map one signed URL onto a manifest ``mediaType``.

    The S3 prefix is what distinguishes the recording styles, and the backend's
    own manifest builder keys off the same thing:

    * ``user_camera_recordings/…/{stem}.webm``      -> CAMERA_VIDEO (pure camera)
    * ``user_session_recordings/…/v3/{stem}.webm``  -> provisionally SCREEN_CAMERA_VIDEO
    * ``user_session_recordings/…/{stem}.webm``     -> SCREEN_VIDEO (screen only)
    * ``user_session_recordings/…/{epoch}.json``    -> RRWEB_EVENT (DOM batch)

    The v3 case is only provisional: a ``v3`` session recording is the combined
    screen+camera clip *unless* the attempt also has its own camera recordings,
    in which case the session video is screen-only. ``resolve_media_types``
    settles it once every url has been seen.
    """

    path = unquote(urlparse(url).path)
    s3_key = path.lstrip("/")
    filename = PurePosixPath(path).name
    lowered = filename.lower()
    is_camera_prefix = "user_camera_recordings" in path
    is_v3 = "/v3/" in path

    role = "unknown"
    media_type: str | None = None
    if lowered.endswith("__metadata.json.gz"):
        role, stem = "metadata", filename[: -len("__metadata.json.gz")]
    elif lowered.endswith("__events.json.gz"):
        role, stem = "events", filename[: -len("__events.json.gz")]
    elif lowered.endswith(".webm"):
        role, stem = "media", filename[: -len(".webm")]
        media_type = (
            "CAMERA_VIDEO"
            if is_camera_prefix
            else "SCREEN_CAMERA_VIDEO"
            if is_v3
            else "SCREEN_VIDEO"
        )
    elif lowered.endswith((".json", ".json.gz")):
        role = "rrweb"
        media_type = "RRWEB_EVENT"
        stem = re.sub(r"\.(?:json\.gz|json)$", "", filename, flags=re.IGNORECASE)
    else:
        stem = re.sub(r"\.(?:json\.gz|json|webm)$", "", filename, flags=re.IGNORECASE)

    match = STEM_RE.match(stem)
    duration = match.group("duration") if match else None
    return ChunkUrl(
        url=url,
        s3_key=s3_key,
        filename=filename,
        stem=stem,
        role=role,
        media_type=media_type,
        epoch_ms=int(match.group("epoch")) if match else None,
        duration_ms=int(duration) if duration is not None else None,
    )


def resolve_media_types(chunks: list[ChunkUrl]) -> list[ChunkUrl]:
    """Settle the combined-vs-screen-only question across the whole input.

    The backend contract (see ``tests/payloads.combined_payload``) is that an
    attempt never has separate CAMERA_VIDEO chunks alongside
    SCREEN_CAMERA_VIDEO ones. Read the other way round: if the attempt *does*
    ship its own camera recordings, the session video is plain screen, not a
    combined clip. Getting this wrong would analyse the camera twice — once
    standalone and once inside the "combined" clip — doubling Gemini cost and
    letting one camera event be counted as two.
    """

    has_camera = any(chunk.media_type == "CAMERA_VIDEO" for chunk in chunks)
    if not has_camera:
        return chunks
    return [
        replace(chunk, media_type="SCREEN_VIDEO")
        if chunk.media_type == "SCREEN_CAMERA_VIDEO"
        else chunk
        for chunk in chunks
    ]


def identity_from_key(s3_key: str) -> tuple[str, str, str]:
    """Recover (attempt_user_id, org_assess_id, exam_attempt_id) from the key.

    v3 keys are ``.../user_session_recordings/{a}/{b}/{c}/v3/{stem}.webm`` and
    the sidecar metadata's ``uploaderPathIdentifier`` is exactly ``a/b/c``.
    """

    parts = PurePosixPath(s3_key).parts
    # Camera recordings carry the same {a}/{b}/{c} triple under their own
    # prefix, so anchoring only on the session prefix silently reduced a
    # camera-only attempt to the "local-*" placeholders — and with it the
    # exam_attempt_id that separates one section from the next.
    anchor = next(
        (
            parts.index(name)
            for name in ("user_session_recordings", "user_camera_recordings")
            if name in parts
        ),
        None,
    )
    if anchor is None:
        return ("local-candidate", "local-assessment", "local-attempt")
    tail = parts[anchor + 1 : anchor + 4]
    if len(tail) < 3:
        return ("local-candidate", "local-assessment", "local-attempt")
    return (tail[0], tail[1], tail[2])


def iso_utc(epoch_ms: int) -> str:
    """ISO-8601 with an explicit ``Z``.

    The offset matters: ``parse_activity_epoch_ms`` treats a timestamp with no
    timezone as IST-naive and subtracts 5:30, which would put T0 five and a
    half hours away from the chunk epochs.
    """

    return datetime.fromtimestamp(epoch_ms / 1000, tz=UTC).isoformat().replace("+00:00", "Z")


# --------------------------------------------------------------------------
# 2. Manifest chunks -> the payload the backend would have staged
# --------------------------------------------------------------------------


def rrweb_event_span(cache_dir: Path, chunk: ChunkUrl) -> tuple[int, int] | None:
    """First/last DOM event timestamp of a cached rrweb batch.

    An rrweb filename carries only the upload epoch, so without this the
    timeline has no duration for the batch and falls back to a flat
    ``KEYSTROKE_FALLBACK_DURATION_MS``. The backend's own manifest does supply
    ``durationMs`` for RRWEB_EVENT, so deriving the real span here is the
    faithful thing to do rather than an embellishment.
    """

    path = cache_dir / chunk.filename
    if not path.is_file():
        return None
    try:
        events = json.loads(maybe_gunzip(path.read_bytes()).decode("utf-8"))
    except (OSError, ValueError):
        return None
    if isinstance(events, dict):
        events = events.get("events")
    if not isinstance(events, list):
        return None
    stamps = [
        int(event["timestamp"])
        for event in events
        if isinstance(event, dict) and isinstance(event.get("timestamp"), (int, float))
    ]
    return (min(stamps), max(stamps)) if stamps else None


def build_staged_payload(
    chunks: list[ChunkUrl],
    *,
    media_type: str | None,
    section_id: str,
    section_type: str,
    cache_dir: Path,
    overrides: dict[str, Any],
) -> dict[str, Any]:
    usable = [chunk for chunk in chunks if chunk.role in ("media", "rrweb")]
    if not usable:
        raise SystemExit("input has no .webm or rrweb .json chunk urls")

    placeable = [chunk for chunk in usable if chunk.epoch_ms is not None]
    if not placeable:
        raise SystemExit(
            "no chunk filename matched '{epoch13}[__{duration}]'; the pipeline "
            "cannot place these chunks on a timeline"
        )

    candidate_id, assessment_id, _ = identity_from_key(placeable[0].s3_key)
    candidate_id = overrides.get("candidateId") or candidate_id
    assessment_id = overrides.get("assessmentId") or assessment_id
    # One review can span several sections, and the backend tells them apart by
    # examAttemptId — the third path segment. Reading it per chunk instead of
    # once from the first url is what keeps a two-section sitting a single
    # review with two sections, rather than one section wearing both.
    attempt_by_key = {c.s3_key: identity_from_key(c.s3_key)[2] for c in placeable}

    # An rrweb batch's own event timestamps beat the filename epoch, which is
    # only the upload time; media chunks already carry a real duration.
    rrweb_spans = {
        chunk.stem: rrweb_event_span(cache_dir, chunk)
        for chunk in placeable
        if chunk.role == "rrweb"
    }

    manifest_chunks: list[dict[str, Any]] = []
    starts: list[int] = []
    ends: list[int] = []
    for chunk in sorted(placeable, key=lambda item: item.epoch_ms or 0):
        duration = chunk.duration_ms
        span = rrweb_spans.get(chunk.stem) if chunk.role == "rrweb" else None
        if span is not None:
            duration = max(0, span[1] - span[0])
            starts.append(span[0])
            ends.append(max(span[1], chunk.epoch_ms or span[1]))
        else:
            start = chunk.start_ms
            if start is not None:
                starts.append(start)
            ends.append(chunk.epoch_ms or 0)
        manifest_chunks.append(
            {
                "mediaType": media_type or chunk.media_type,
                "examAttemptId": attempt_by_key[chunk.s3_key],
                "s3Key": chunk.s3_key,
                "epochMs": chunk.epoch_ms,
                "durationMs": duration,
            }
        )

    t0_ms = min(starts) if starts else min(c.epoch_ms for c in placeable)
    end_ms = max(ends)
    # One section per examAttemptId, in the order they were sat. A single
    # attempt keeps the plain --section-id so existing runs are unchanged;
    # several get "<section-id>1", "<section-id>2", ... so a card's sectionId
    # still reads as a section rather than a UUID.
    spans_by_attempt: dict[str, list[int]] = {}
    for chunk in placeable:
        attempt = attempt_by_key[chunk.s3_key]
        span = rrweb_spans.get(chunk.stem) if chunk.role == "rrweb" else None
        start = span[0] if span else (chunk.start_ms if chunk.start_ms is not None else chunk.epoch_ms)
        end = span[1] if span else chunk.epoch_ms
        bounds = spans_by_attempt.setdefault(attempt, [start or 0, end or 0])
        bounds[0] = min(bounds[0], start or bounds[0])
        bounds[1] = max(bounds[1], end or bounds[1])
    ordered_attempts = sorted(spans_by_attempt, key=lambda a: spans_by_attempt[a][0])
    section_ids = {
        attempt: (section_id if len(ordered_attempts) == 1 else f"{section_id}{index + 1}")
        for index, attempt in enumerate(ordered_attempts)
    }
    sections = overrides.get("sections") or [
        {
            "sectionId": section_ids[attempt],
            "examId": f"{assessment_id}-exam",
            "order": index,
            "examAttemptId": attempt,
            "startDatetime": iso_utc(spans_by_attempt[attempt][0]),
            "endDatetime": iso_utc(spans_by_attempt[attempt][1]),
            "sectionType": section_type,
        }
        for index, attempt in enumerate(ordered_attempts)
    ]
    activity_logs = overrides.get("activityLogs") or [
        {
            "order": 0,
            "activityType": "ASSESSMENT_STARTED",
            "creationDatetime": iso_utc(t0_ms),
            "offsetInSeconds": 0,
            "metadata": {},
        },
        *[
            entry
            for index, attempt in enumerate(ordered_attempts)
            for entry in (
                {
                    "order": 1 + index * 2,
                    "activityType": "SECTION_STARTED",
                    "creationDatetime": iso_utc(spans_by_attempt[attempt][0]),
                    "metadata": {"sectionId": section_ids[attempt]},
                },
                {
                    "order": 2 + index * 2,
                    "activityType": "SECTION_SUBMITTED",
                    "creationDatetime": iso_utc(spans_by_attempt[attempt][1]),
                    "metadata": {"sectionId": section_ids[attempt]},
                },
            )
        ],
        {
            "order": 1 + len(ordered_attempts) * 2,
            "activityType": "ALL_SECTIONS_COMPLETED",
            "creationDatetime": iso_utc(end_ms),
            "metadata": {},
        },
    ]

    return {
        "reviewId": overrides.get("reviewId") or f"local-{t0_ms}",
        "orgAssessId": assessment_id,
        "attemptUserId": candidate_id,
        "manifest": {"chunks": manifest_chunks},
        "activityTimeline": {"sections": sections, "activityLogs": activity_logs},
        "shouldAnalyseSebLogs": False,
        "shouldAnalyseVideo": True,
    }


# --------------------------------------------------------------------------
# 3. Local stand-ins for the S3/presign boundaries
# --------------------------------------------------------------------------


class UrlMediaUriProvider:
    """``MediaUriProvider`` backed by the signed URLs from the input file.

    Production presigns ``chunk.source_ref`` here; locally the URL is already
    signed, so this is a lookup by S3 key.
    """

    def __init__(self, by_key: dict[str, str]) -> None:
        self._by_key = by_key

    async def media_uri_for(self, chunk: EvidenceChunkRef) -> str:
        try:
            return self._by_key[chunk.source_ref]
        except KeyError:
            raise RuntimeError(f"no signed url in input.json for {chunk.source_ref}") from None


class UrlObjectStore:
    """``ObjectStore`` that GETs the signed URL, preferring the local cache."""

    def __init__(self, by_key: dict[str, str], cache_dir: Path) -> None:
        self._by_key = by_key
        self._cache_dir = cache_dir

    async def get_bytes(self, ref: str) -> bytes:
        cached = self._cache_dir / PurePosixPath(ref).name
        if cached.is_file():
            return await asyncio.to_thread(cached.read_bytes)
        url = self._by_key.get(ref)
        if url is None:
            raise FileNotFoundError(f"no signed url in input.json for {ref}")
        async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.content

    async def get_json(self, ref: str) -> Any:
        return json.loads(maybe_gunzip(await self.get_bytes(ref)).decode("utf-8"))


class FakeGemini:
    """Scripted stand-in, so the plumbing can be exercised without an API key.

    Returns a canned combined camera+screen half for every flash (perception)
    call and an empty adjudication for the pro (deliberation) call. Useful for
    proving the wiring, the timeline and the bundle assembly; useless for
    judging prompt or model quality.
    """

    PERCEPTION = json.dumps(
        {
            "camera": {
                "chunkDurationMs": 60_000,
                "chunkFullyReviewed": True,
                "captureQualityTier": "HIGH",
                "events": [
                    {
                        "eventId": "fake-1",
                        "kind": "second_person_present",
                        "startMsLocal": 4_000,
                        "endMsLocal": 9_000,
                        "attrs": {"secondPersonVisible": "yes"},
                        "confidence": 0.8,
                    }
                ],
            },
            "screen": {
                "examUiVisible": "yes",
                "foregroundAppClass": "browser",
                "externalResourceLabels": ["stackoverflow.com"],
                "aiAssistantUiVisible": "yes",
                "secondaryWorkspaceVisible": "no",
                "fullscreenExamLikely": "no",
                "pasteCueVisible": "yes",
                "pastedTextExcerpt": "def solve(n):",
                "visibleQuestionRef": "Q3",
                "confidence": 0.9,
            },
        }
    )

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def generate(self, *, model: str, contents: Any, config: Any) -> dict[str, Any]:
        self.calls.append(model)
        text = "{}" if "pro" in model else self.PERCEPTION
        return {"text": text, "usage": {"prompt_tk": 0, "completion_tk": 0}}


class SemaphoreLimiter:
    def __init__(self, slots: int) -> None:
        self._semaphore = asyncio.Semaphore(slots)

    async def __aenter__(self) -> None:
        await self._semaphore.acquire()

    async def __aexit__(self, *_exc: object) -> None:
        self._semaphore.release()


# --------------------------------------------------------------------------
# 4. Media delivery into Gemini
# --------------------------------------------------------------------------


async def download_all(chunks: list[ChunkUrl], cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(timeout=300.0, follow_redirects=True) as client:

        async def fetch(chunk: ChunkUrl) -> None:
            target = cache_dir / chunk.filename
            if target.is_file() and target.stat().st_size:
                print(f"  cached  {chunk.filename} ({target.stat().st_size:,} bytes)")
                return
            response = await client.get(chunk.url)
            response.raise_for_status()
            target.write_bytes(response.content)
            print(f"  fetched {chunk.filename} ({len(response.content):,} bytes)")

        await asyncio.gather(*(fetch(chunk) for chunk in chunks))


def install_media_delivery(
    mode: str,
    *,
    cache_dir: Path,
    url_to_filename: dict[str, str],
    api_key: str | None,
    logger: StructuredLogger,
) -> None:
    """Override how a chunk's bytes reach Gemini.

    ``auto`` keeps the shipped behaviour: probe for EBML magic and, when the
    object really is a WebM, hand Gemini the URL as ``fileData.fileUri``. That
    is the production path, but the Gemini *developer* API only resolves a
    ``fileUri`` that belongs to its own Files API (or YouTube), so a raw
    presigned S3 URL is rejected there. ``inline`` and ``files`` are the two
    modes that work with a plain ``GEMINI_API_KEY``.
    """

    from nw_assessments_ai_analysis_backend.perception import chunk_job

    if mode == "auto":
        chunk_job.resolve_media_parts = STOCK_RESOLVE_MEDIA_PARTS
        return

    def local_path(signed_url: str) -> Path | None:
        name = url_to_filename.get(signed_url)
        if name is None:
            return None
        candidate = cache_dir / name
        return candidate if candidate.is_file() and candidate.stat().st_size else None

    async def read_media(signed_url: str) -> bytes:
        cached = local_path(signed_url)
        if cached is not None:
            return await asyncio.to_thread(cached.read_bytes)
        async with httpx.AsyncClient(timeout=300.0, follow_redirects=True) as client:
            response = await client.get(signed_url)
            response.raise_for_status()
            return response.content

    if mode == "inline":

        async def resolve_inline(
            signed_url: str, mime_type: str, text_parts: Any, **_: Any
        ) -> tuple[list[dict[str, Any]], str]:
            decoded = decode_screen_recording(maybe_gunzip(await read_media(signed_url)))
            if not decoded:
                raise ValueError("perception media decoded to empty buffer")
            resolved = "video/webm" if is_ebml_magic(decoded) else mime_type
            logger.info(
                "local-runner: inlining media",
                bytes=len(decoded),
                mime_type=resolved,
                cached=local_path(signed_url) is not None,
            )
            return build_parts_from_inline(decoded, resolved, text_parts), "inline_data"

        chunk_job.resolve_media_parts = resolve_inline
        return

    if mode == "files":
        if not api_key:
            raise SystemExit("--media-delivery files needs GEMINI_API_KEY")
        from google import genai

        client = genai.Client(api_key=api_key)
        uploads: dict[str, str] = {}
        lock = asyncio.Lock()

        async def upload(signed_url: str) -> str:
            async with lock:
                if signed_url in uploads:
                    return uploads[signed_url]
                path = local_path(signed_url)
                if path is None:
                    path = cache_dir / (url_to_filename.get(signed_url) or "chunk.webm")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(await read_media(signed_url))
                handle = await client.aio.files.upload(
                    file=str(path), config={"mime_type": "video/webm"}
                )
                # A freshly uploaded video is PROCESSING; Gemini rejects it
                # until it turns ACTIVE.
                for _ in range(60):
                    state = str(getattr(handle.state, "name", handle.state) or "")
                    if state == "ACTIVE":
                        break
                    if state == "FAILED":
                        raise RuntimeError(f"Gemini Files API failed to process {path.name}")
                    await asyncio.sleep(2)
                    handle = await client.aio.files.get(name=handle.name)
                logger.info(
                    "local-runner: uploaded media to Files API",
                    file=path.name,
                    uri=handle.uri,
                )
                uploads[signed_url] = handle.uri
                return handle.uri

        async def resolve_files(
            signed_url: str, mime_type: str, text_parts: Any, **_: Any
        ) -> tuple[list[dict[str, Any]], str]:
            uri = await upload(signed_url)
            return build_parts_from_url(uri, "video/webm", text_parts), "file_data"

        chunk_job.resolve_media_parts = resolve_files


# --------------------------------------------------------------------------
# 5. Wiring
# --------------------------------------------------------------------------


def rel(path: Path) -> str:
    """Display path, tolerating one passed in relative or outside the repo."""

    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n")
    print(f"  wrote {rel(path)}")


def read_sidecar_metadata(cache_dir: Path, chunks: list[ChunkUrl]) -> list[dict[str, Any]]:
    """Report what the ``__metadata.json.gz`` sidecars say, for cross-checking.

    The pipeline never reads these — it derives every chunk's span from the
    ``{epoch}__{duration}`` filename stem — so this is purely a sanity check
    that the stems agree with the recorder's own timestamps.
    """

    out: list[dict[str, Any]] = []
    for chunk in chunks:
        if chunk.role != "metadata":
            continue
        path = cache_dir / chunk.filename
        if not path.is_file():
            continue
        try:
            # S3 stores these with Content-Encoding: gzip, so an HTTP client
            # that advertises gzip hands back plain JSON already.
            meta = json.loads(maybe_gunzip(path.read_bytes()).decode("utf-8"))
        except (OSError, ValueError) as error:
            out.append({"stem": chunk.stem, "error": str(error)})
            continue
        time_object = meta.get("time_object") or {}
        out.append(
            {
                "stem": chunk.stem,
                "chunkIndex": meta.get("chunkIndex"),
                "mimeType": meta.get("mimeType"),
                "start": time_object.get("start"),
                "end": time_object.get("end"),
                "durationFromMetadata": time_object.get("duration"),
                "durationFromStem": chunk.duration_ms,
                "stemMatchesMetadata": str(time_object.get("duration") or "")
                == str(chunk.duration_ms),
            }
        )
    return sorted(out, key=lambda item: str(item.get("stem")))


def read_client_reported_events(
    cache_dir: Path, chunks: list[ChunkUrl], t0_ms: int
) -> list[dict[str, Any]]:
    """Decode the ``__events.json.gz`` sidecars.

    These are the client's own proctoring events (``MULTIPLE_FACES_DETECTED``
    and friends), not rrweb DOM events. ``run_behavioral_analysis`` has a
    ``client_reported_events`` parameter for exactly this shape, but
    ``run_integrity_review`` never fills it — so today these are read here for
    visibility only and reach no stage of the review.
    """

    out: list[dict[str, Any]] = []
    for chunk in chunks:
        if chunk.role != "events":
            continue
        path = cache_dir / chunk.filename
        if not path.is_file():
            continue
        try:
            payload = json.loads(maybe_gunzip(path.read_bytes()).decode("utf-8"))
        except (OSError, ValueError) as error:
            out.append({"stem": chunk.stem, "error": str(error)})
            continue
        for event in payload.get("events") or []:
            # The recorder emits this as a float (sub-millisecond precision),
            # not always an int, so accept both.
            absolute = event.get("absoluteTimestamp")
            offset = (
                round(absolute - t0_ms)
                if isinstance(absolute, (int, float)) and not isinstance(absolute, bool)
                else None
            )
            out.append(
                {
                    "stem": chunk.stem,
                    "identifier": event.get("identifier"),
                    "absoluteTimestamp": absolute,
                    "sessionOffsetMs": offset,
                    "message": (event.get("payload") or {}).get("message"),
                }
            )
    return out


def summarize(normalized: NormalizedReviewRequest, timeline: Any) -> dict[str, Any]:
    request = normalized.request
    return {
        "payloadSummary": normalized.summary.as_log_fields(),
        "examMode": request.evidence.exam_mode.value,
        "manifests": {
            manifest.evidence_type.value: {
                "totalChunks": manifest.total_chunks,
                "totalDurationMs": manifest.total_duration_ms,
                "chunkIds": [chunk.chunk_id for chunk in manifest.chunks],
            }
            for manifest in request.evidence.manifests
        },
        "timeline": {
            "durationMs": timeline.duration_ms,
            "t0Source": getattr(timeline.sync_report, "t0_source", None),
            "durationSource": getattr(timeline.sync_report, "duration_source", None),
            "sections": [
                {
                    "sectionId": section.section_id,
                    "label": section.label,
                    "sectionType": section.section_type,
                    "startMs": section.start_ms,
                    "endMs": section.end_ms,
                }
                for section in timeline.sections
            ],
            "videoChunkSpans": len(timeline.video_chunk_spans),
            "screenChunkSpans": [
                {
                    "chunkId": span.chunk_id,
                    "startOffsetMs": span.start_offset_ms,
                    "endOffsetMs": span.end_offset_ms,
                    "durationMs": span.duration_ms,
                    "sectionId": span.section_id,
                }
                for span in timeline.screen_chunk_spans
            ],
        },
    }


async def run(args: argparse.Namespace) -> int:
    input_paths: list[Path] = list(args.input) or [ROOT / "input" / "input.json"]
    out_dir: Path = args.output_dir
    cache_dir: Path = args.cache_dir
    prefix = f"{args.run_name}_" if args.run_name else ""

    def out(name: str) -> Path:
        return out_dir / f"{prefix}{name}"

    # One attempt's evidence often arrives as several lists (camera in one,
    # session recordings in another), so accept --input more than once and
    # merge them into a single review. Later files' overrides win.
    raw: dict[str, Any] = {}
    urls: list[str] = []
    seen: set[str] = set()
    for path in input_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        found = payload.get("chunk_urls") or payload.get("chunkUrls") or []
        if not found:
            raise SystemExit(f"{path} has no 'chunk_urls'")
        for url in found:
            # Same object signed twice across files is one chunk, not two.
            key = unquote(urlparse(url).path)
            if key in seen:
                continue
            seen.add(key)
            urls.append(url)
        raw.update({k: v for k, v in payload.items() if k not in ("chunk_urls", "chunkUrls")})

    chunks = resolve_media_types([classify_url(url) for url in urls])
    print(f"input: {len(chunks)} urls from {', '.join(rel(p) for p in input_paths)}")
    for role in ("media", "rrweb", "metadata", "events", "unknown"):
        matching = [chunk for chunk in chunks if chunk.role == role]
        if matching:
            print(f"  {role:9s} {len(matching):2d}  {', '.join(c.stem for c in matching)}")

    if args.download:
        print("\ndownloading media into", rel(cache_dir))
        await download_all(chunks, cache_dir)

    staged = build_staged_payload(
        chunks,
        media_type=args.media_type,
        section_id=args.section_id,
        section_type=args.section_type,
        cache_dir=cache_dir,
        overrides=raw,
    )
    normalized = build_review_request(StagedReviewPayload.model_validate(staged))
    timeline = build_master_timeline(normalized.request)

    print("\ningest:")
    dump(out("staged_payload.json"), staged)
    dump(
        out("review_request.json"),
        normalized.request.model_dump(mode="json", by_alias=True),
    )
    dump(out("ingest_summary.json"), summarize(normalized, timeline))
    sidecars = read_sidecar_metadata(cache_dir, chunks)
    if sidecars:
        dump(out("sidecar_metadata.json"), sidecars)
    client_events = read_client_reported_events(
        cache_dir, chunks, timeline.canonical_timeline.t0
    )
    if client_events:
        dump(out("client_reported_events.json"), client_events)

    summary = normalized.summary
    print(
        f"  exam_mode={summary.exam_mode} screen_camera={summary.screen_camera_chunks} "
        f"screen={summary.screen_chunks} camera={summary.camera_chunks} "
        f"rrweb={summary.rrweb_chunks} dropped={summary.dropped_chunks}"
    )
    for warning in summary.warnings:
        print(f"  warning: {warning}")
    print(
        f"  timeline duration={timeline.duration_ms}ms "
        f"screen_spans={len(timeline.screen_chunk_spans)} "
        f"video_spans={len(timeline.video_chunk_spans)}"
    )

    if args.dry_run:
        print("\n--dry-run: stopping before any Gemini call")
        return 0

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key and not args.fake_gemini and not args.openrouter:
        raise SystemExit("GEMINI_API_KEY is required (or pass --dry-run / --fake-gemini)")
    if args.openrouter and not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit("--openrouter needs OPENROUTER_API_KEY")

    by_key = {chunk.s3_key: chunk.url for chunk in chunks}
    url_to_filename = {chunk.url: chunk.filename for chunk in chunks}
    logger = StructuredLogger(
        staged["reviewId"],
        candidateId=normalized.request.candidate_id,
    )
    if args.perception_model:
        # chunk_job._gemini_text is the single perception call site, so
        # rebinding the module's model constant swaps the model for every
        # camera / screen / combined chunk without touching the library
        # default the worker ships with.
        _chunk_job.DEFAULT_GEMINI_FLASH_MODEL = args.perception_model
        logger.info("local-runner: perception model overridden", model=args.perception_model)

    if args.deliberation_model:
        # pipeline.py reads DEFAULT_GEMINI_PRO_MODEL from its own module
        # namespace at call time, so rebinding it there swaps the model for the
        # deliberation call. The rules module's copy is rebound too, so the
        # bundle's provenance records the model that actually ran rather than
        # the pro default it no longer used.
        _pipeline.DEFAULT_GEMINI_PRO_MODEL = args.deliberation_model
        _rules.DELIBERATION_MODEL_VERSION = args.deliberation_model
        logger.info(
            "local-runner: deliberation model overridden", model=args.deliberation_model
        )

    if args.openrouter:
        # Deliberation goes to an OpenRouter-hosted model instead of Gemini.
        # Perception must come from --perception-cache: this client speaks text
        # only, so a cache miss fails loudly rather than silently costing money
        # on a provider that cannot serve the media call anyway.
        gemini: Any = OpenRouterClient(
            os.environ["OPENROUTER_API_KEY"],
            base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            only_model=args.deliberation_model,
        )
        delivery = "n/a (openrouter)"
        install_media_delivery(
            "inline",
            cache_dir=cache_dir,
            url_to_filename=url_to_filename,
            api_key=api_key,
            logger=logger,
        )
    elif args.fake_gemini:
        # No real call is made, so skip fetching or uploading a single byte.
        gemini: Any = FakeGemini()
        delivery = "stubbed"
        install_media_delivery(
            "inline",
            cache_dir=cache_dir,
            url_to_filename=url_to_filename,
            api_key=api_key,
            logger=logger,
        )
    else:
        gemini = GoogleGeminiClient(api_key)
        delivery = args.media_delivery
        install_media_delivery(
            args.media_delivery,
            cache_dir=cache_dir,
            url_to_filename=url_to_filename,
            api_key=api_key,
            logger=logger,
        )
    print(
        f"\nrunning pipeline (gemini: {'FAKE' if args.fake_gemini else 'live'}, "
        f"media: {delivery}, perception model: "
        f"{args.perception_model or _chunk_job.DEFAULT_GEMINI_FLASH_MODEL})"
    )

    cache = InMemoryPerceptionCache()
    if args.perception_cache:
        # Replay a previous run's per-chunk perception so the perception stage
        # is a cache hit and makes no Gemini calls. Isolates a deliberation or
        # bundle-assembly change from Gemini's run-to-run variance, which is
        # large enough to move the episode inventory on its own.
        cache._values = json.loads(Path(args.perception_cache).read_text())
        print(f"  replaying perception from {rel(Path(args.perception_cache))}"
              f" ({len(cache._values)} entries) — perception stage will not call Gemini")
    deps = PipelineDeps(
        object_store=UrlObjectStore(by_key, cache_dir),
        gemini=gemini,
        cache=cache,
        limiter=SemaphoreLimiter(args.concurrency),
        logger=logger,
        media_uri_provider=UrlMediaUriProvider(by_key),
        organization_id=os.environ.get("ORGANIZATION_ID", "local"),
        ai_usage_logger=NullAiUsageLogger(),
    )

    try:
        bundle = await run_integrity_review(normalized.request, deps)
    finally:
        # The per-chunk perception output is where a bad prompt or a bad clip
        # shows up first, so keep it even when deliberation blows up.
        if cache._values:
            dump(out("perception_cache.json"), dict(cache._values))

    print("\nresult:")
    dump(
        out(args.bundle_name),
        bundle.model_dump(mode="json", by_alias=True, exclude_none=True),
    )
    print(f"  recommendation   {bundle.recommendation.category}")
    print(f"  detected signals {len(bundle.detected_signals.signals)}")
    print(f"  media index      {len(bundle.media_index)}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        default=[],
        help=(
            "input JSON with chunk_urls; repeatable, so one attempt's camera and "
            "session URL lists can be passed as separate files and merged"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output")
    parser.add_argument(
        "--bundle-name",
        default="evidence_bundle.json",
        help="filename for the final bundle inside --output-dir",
    )
    parser.add_argument(
        "--run-name",
        default="",
        help=(
            "prefix for every output filename, so a second attempt's artifacts "
            "never overwrite the first's (e.g. --run-name attempt2)"
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "input" / "media",
        help="local copies of the chunks, so a run survives signed-URL expiry",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="fetch every url into --cache-dir before running",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="build the payload, request and timeline only; no Gemini calls",
    )
    parser.add_argument(
        "--media-type",
        choices=MEDIA_TYPE_CHOICES,
        default=None,
        help=(
            "force one mediaType for every chunk. Default is to classify each "
            "url by its S3 prefix (camera / v3-combined / screen / rrweb)."
        ),
    )
    parser.add_argument(
        "--media-delivery",
        choices=("inline", "files", "auto"),
        default="inline",
        help=(
            "how the clip reaches Gemini. inline: base64 in the request "
            "(default, always works). files: Gemini Files API. auto: the "
            "shipped presigned-fileUri path."
        ),
    )
    parser.add_argument("--section-id", default="mcq", help="synthetic section id")
    parser.add_argument("--section-type", default="mcq", help="synthetic section type")
    parser.add_argument(
        "--fake-gemini",
        action="store_true",
        help="run the whole flow with scripted Gemini responses (no API key, no cost)",
    )
    parser.add_argument(
        "--perception-model",
        default=None,
        help=(
            "override the flash model used for every perception chunk "
            "(e.g. gemini-3.8-flash); deliberation still uses the pro model"
        ),
    )
    parser.add_argument(
        "--openrouter",
        action="store_true",
        help=(
            "send the deliberation call to OpenRouter (OPENROUTER_API_KEY) instead "
            "of Gemini; requires --perception-cache and an OpenRouter model id "
            "such as anthropic/claude-opus-5 in --deliberation-model"
        ),
    )
    parser.add_argument(
        "--deliberation-model",
        default=None,
        help=(
            "override the pro model used for the deliberation call "
            "(e.g. gemini-3.8-flash); perception still uses the flash model"
        ),
    )
    parser.add_argument(
        "--perception-cache",
        type=Path,
        default=None,
        help=(
            "replay a previous run's *_perception_cache.json so only deliberation "
            "calls Gemini; use it to A/B a deliberation change on fixed perception"
        ),
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=12,
        help=(
            "concurrent Gemini calls. Defaults to the worker's "
            "gemini_per_review_limit, so one local run has the same number of "
            "calls in flight as one review on the ECS task."
        ),
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    sys.exit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
