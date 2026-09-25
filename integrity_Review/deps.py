from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from .contracts.evidence import EvidenceChunkRef


class ObjectStore(Protocol):
    async def get_bytes(self, ref: str) -> bytes: ...

    async def get_json(self, ref: str) -> Any: ...


class GeminiClient(Protocol):
    async def generate(
        self,
        *,
        model: str,
        contents: Sequence[Mapping[str, Any] | Any],
        config: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...


class PerceptionCache(Protocol):
    async def get(self, key: str) -> Mapping[str, Any] | None: ...

    async def set(self, key: str, value: Mapping[str, Any]) -> None: ...


class GeminiLimiter(Protocol):
    async def __aenter__(self) -> None: ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: Any,
    ) -> None: ...


class Logger(Protocol):
    def debug(self, message: str, **fields: Any) -> None: ...

    def info(self, message: str, **fields: Any) -> None: ...

    def warning(self, message: str, **fields: Any) -> None: ...

    def error(self, message: str, **fields: Any) -> None: ...


class AiUsageLogger(Protocol):
    def record(
        self,
        *,
        step: str,
        model_name: str,
        model_version: str,
        status: str,
        latency_seconds: float,
        usage: Mapping[str, int] | None,
        extra_meta: Mapping[str, Any],
    ) -> None: ...


class MediaUriProvider(Protocol):
    async def media_uri_for(self, chunk: EvidenceChunkRef) -> str: ...


@dataclass(frozen=True, slots=True)
class PipelineDeps:
    object_store: ObjectStore
    gemini: GeminiClient
    cache: PerceptionCache
    limiter: GeminiLimiter
    logger: Logger
    media_uri_provider: MediaUriProvider
    organization_id: str
    ai_usage_logger: AiUsageLogger
