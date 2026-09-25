from __future__ import annotations

import base64
import binascii
import gzip
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import httpx

EBML_MAGIC = b"\x1a\x45\xdf\xa3"
GZIP_MAGIC = b"\x1f\x8b"

MediaPart = dict[str, Any]
MediaDelivery = Literal["file_data", "inline_data"]


@dataclass(frozen=True, slots=True)
class MediaPayload:
    bytes: bytes
    mime_type: str


def maybe_gunzip(data: bytes) -> bytes:
    return gzip.decompress(data) if data.startswith(GZIP_MAGIC) else data


def decode_screen_recording(data: bytes) -> bytes:
    if data.startswith(EBML_MAGIC):
        return data
    if not data or data[:1] not in (b'"', b"{"):
        return data
    try:
        parsed: Any = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return data

    data_url: str | None = None
    if isinstance(parsed, str) and parsed.startswith("data:video/"):
        data_url = parsed
    elif isinstance(parsed, dict):
        data_url = next(
            (
                value
                for value in parsed.values()
                if isinstance(value, str) and value.startswith("data:video/")
            ),
            None,
        )
    if not data_url or "," not in data_url:
        return data
    try:
        return base64.b64decode(data_url.split(",", 1)[1], validate=True)
    except (ValueError, binascii.Error):
        return data


def normalize_perception_media(raw: bytes) -> MediaPayload:
    data = decode_screen_recording(maybe_gunzip(raw))
    if not data:
        raise ValueError("perception media is empty after decoding")
    return MediaPayload(
        bytes=data,
        mime_type="video/webm" if data.startswith(EBML_MAGIC) else "image/jpeg",
    )


def is_ebml_magic(data: bytes) -> bool:
    return len(data) >= 4 and data[:4] == EBML_MAGIC


_AUDIO_CODEC_IDS = (b"A_OPUS", b"A_VORBIS", b"A_AAC", b"A_MPEG", b"A_PCM")

SILENT_CLIP_NOTE = (
    "AUDIO: this clip has NO audio track — there is nothing to hear. "
    "Set audio.speechPresent=no, audio.speechContentClass=silence, "
    "audio.conversationSummaryEn=null, and every other audio.* field to UNKNOWN. "
    "Do not emit a speech event and do not describe anything as heard, said, "
    "whispered or dictated."
)


def webm_has_audio_track(data: bytes) -> bool:
    if not is_ebml_magic(data):
        return False
    return any(codec in data for codec in _AUDIO_CODEC_IDS)


def build_parts_from_url(
    signed_url: str,
    mime_type: str,
    text_parts: Sequence[str],
) -> list[MediaPart]:
    parts: list[MediaPart] = [{"text": text} for text in text_parts]
    parts.append({"fileData": {"fileUri": signed_url, "mimeType": mime_type}})
    return parts


def build_parts_from_inline(
    data: bytes,
    mime_type: str,
    text_parts: Sequence[str],
) -> list[MediaPart]:
    parts: list[MediaPart] = [{"text": text} for text in text_parts]
    parts.append(
        {
            "inlineData": {
                "mimeType": mime_type,
                "data": base64.b64encode(data).decode("ascii"),
            }
        }
    )
    return parts


async def resolve_media_parts(
    signed_url: str,
    mime_type: str,
    text_parts: Sequence[str],
    *,
    client: httpx.AsyncClient | None = None,
) -> tuple[list[MediaPart], MediaDelivery]:
    async def _fetch_range() -> bytes | None:
        owns_client = client is None
        active = client or httpx.AsyncClient(timeout=30.0)
        try:
            response = await active.get(
                signed_url,
                headers={"Range": "bytes=0-3"},
            )
            if response.status_code not in (200, 206):
                return None
            return response.content
        finally:
            if owns_client:
                await active.aclose()

    async def _fetch_all() -> bytes:
        owns_client = client is None
        active = client or httpx.AsyncClient(timeout=120.0)
        try:
            response = await active.get(signed_url)
            response.raise_for_status()
            return response.content
        finally:
            if owns_client:
                await active.aclose()

    probe = await _fetch_range()
    if probe is not None and is_ebml_magic(probe):
        return build_parts_from_url(signed_url, mime_type, text_parts), "file_data"

    raw = await _fetch_all()
    decoded = decode_screen_recording(maybe_gunzip(raw))
    if not decoded:
        raise ValueError("perception media decoded to empty buffer")
    resolved_mime = "video/webm" if is_ebml_magic(decoded) else mime_type
    parts = list(text_parts)
    if is_ebml_magic(decoded) and not webm_has_audio_track(decoded):
        parts.append(SILENT_CLIP_NOTE)
    return build_parts_from_inline(decoded, resolved_mime, parts), "inline_data"
