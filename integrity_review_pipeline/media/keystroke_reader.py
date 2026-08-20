"""Decode the two rrweb chunk encodings observed in production."""

from __future__ import annotations

import gzip
import json
import zlib
from typing import Any

from .perception_media import GZIP_MAGIC

_ZLIB_SECOND_BYTES = {0x01, 0x5E, 0x9C, 0xDA}


def decode_rrweb_chunk(raw: bytes) -> list[dict[str, Any]]:
    if raw.startswith(GZIP_MAGIC):
        raw = gzip.decompress(raw)
    parsed: Any = json.loads(raw.decode("utf-8"))
    if isinstance(parsed, dict):
        parsed = parsed.get("events")
    if not isinstance(parsed, list):
        raise ValueError("rrweb chunk must be an event array or {events: [...]}")

    events: list[dict[str, Any]] = []
    for element in parsed:
        if isinstance(element, dict):
            events.append(element)
            continue
        if not isinstance(element, str):
            continue
        compressed = element.encode("latin-1")
        if (
            len(compressed) < 2
            or compressed[0] != 0x78
            or compressed[1] not in _ZLIB_SECOND_BYTES
        ):
            continue
        try:
            event = json.loads(zlib.decompress(compressed).decode("utf-8"))
        except (zlib.error, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(event, dict):
            events.append(event)
    return events
