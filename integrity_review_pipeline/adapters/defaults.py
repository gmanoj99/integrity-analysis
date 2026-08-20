"""Production-shaped adapters used by the local signed-URL test runner."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import Mapping, Sequence
from typing import Any

import httpx

from ..contracts.evidence import EvidenceChunkRef
from ..deps import PipelineDeps


class SignedUrlObjectStore:
    def __init__(self, timeout_seconds: float = 120.0) -> None:
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_seconds),
            follow_redirects=True,
        )

    async def get_bytes(self, ref: str) -> bytes:
        response = await self._client.get(ref)
        response.raise_for_status()
        return response.content

    async def get_json(self, ref: str) -> Any:
        return json.loads((await self.get_bytes(ref)).decode("utf-8"))


class InMemoryPerceptionCache:
    def __init__(self) -> None:
        self._values: dict[str, Mapping[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Mapping[str, Any] | None:
        async with self._lock:
            return self._values.get(key)

    async def set(self, key: str, value: Mapping[str, Any]) -> None:
        async with self._lock:
            self._values[key] = value


class SemaphoreLimiter:
    def __init__(self, slots: int) -> None:
        self._semaphore = asyncio.Semaphore(slots)

    async def __aenter__(self) -> None:
        await self._semaphore.acquire()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: Any,
    ) -> None:
        self._semaphore.release()


class StructuredLogger:
    def __init__(self, review_id: str) -> None:
        self._review_id = review_id
        self._logger = logging.getLogger("integrity_review_pipeline")

    def _log(self, level: int, message: str, fields: dict[str, Any]) -> None:
        self._logger.log(
            level,
            "%s %s",
            message,
            json.dumps({"reviewId": self._review_id, **fields}, default=str),
        )

    def debug(self, message: str, **fields: Any) -> None:
        self._log(logging.DEBUG, message, fields)

    def info(self, message: str, **fields: Any) -> None:
        self._log(logging.INFO, message, fields)

    def warning(self, message: str, **fields: Any) -> None:
        self._log(logging.WARNING, message, fields)

    def error(self, message: str, **fields: Any) -> None:
        self._log(logging.ERROR, message, fields)


class SignedUrlMediaProvider:
    async def media_uri_for(self, chunk: EvidenceChunkRef) -> str:
        return str(chunk.signed_url)


class GoogleGeminiClient:
    def __init__(self, api_key: str) -> None:
        try:
            from google import genai
        except ImportError as error:  # pragma: no cover - dependency error
            raise RuntimeError("google-genai is required for live analysis") from error
        self._client = genai.Client(api_key=api_key)

    async def generate(
        self,
        *,
        model: str,
        contents: Sequence[Mapping[str, Any] | Any],
        config: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        response = await self._client.aio.models.generate_content(
            model=model,
            contents=list(contents),
            config=dict(config),
        )
        text = response.text or "{}"
        return {"text": text}


def build_default_deps(review_id: str) -> PipelineDeps:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is required")
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    return PipelineDeps(
        object_store=SignedUrlObjectStore(),
        gemini=GoogleGeminiClient(api_key),
        cache=InMemoryPerceptionCache(),
        limiter=SemaphoreLimiter(12),
        logger=StructuredLogger(review_id),
        media_uri_provider=SignedUrlMediaProvider(),
    )
