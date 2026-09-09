"""Dependency protocols for SEB log analysis pipeline."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol


class SebLogObjectStore(Protocol):
    """Protocol for accessing SEB log files in object storage.

    Designed for streaming access to potentially large log files
    without loading entire files into memory.
    """

    def list_keys(self, prefix: str) -> Iterator[str]:
        """List all object keys under the given prefix.

        Yields keys one at a time to avoid materializing large lists.
        """
        ...

    def iter_lines(self, key: str) -> Iterator[str]:
        """Stream lines from an object.

        Yields decoded lines (UTF-8) one at a time without loading
        the entire object into memory. Handles chunked streaming
        and splits on newlines.
        """
        ...

    def get_json(self, key: str) -> dict[str, Any]:
        """Load and parse a JSON object.

        For small files like manifest.json, this loads the entire
        content and parses it as JSON.
        """
        ...

    def get_object_size(self, key: str) -> int:
        """Get the size of an object in bytes."""
        ...


class Logger(Protocol):
    """Protocol for structured logging."""

    def debug(self, message: str, **fields: Any) -> None: ...

    def info(self, message: str, **fields: Any) -> None: ...

    def warning(self, message: str, **fields: Any) -> None: ...

    def error(self, message: str, **fields: Any) -> None: ...


@dataclass(frozen=True, slots=True)
class SebLogPipelineDeps:
    """Dependencies for the SEB log reduction pipeline (deterministic)."""

    object_store: SebLogObjectStore
    logger: Logger


class GeminiClient(Protocol):
    """Protocol for Gemini client."""

    async def generate(
        self,
        *,
        model: str,
        contents: Sequence[Mapping[str, Any] | Any],
        config: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...


class GeminiLimiter(Protocol):
    """Protocol for Gemini rate limiter."""

    async def __aenter__(self) -> None: ...
    async def __aexit__(self, *args: Any) -> None: ...


class AiUsageLogger(Protocol):
    """Protocol for AI usage logger."""

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


@dataclass(frozen=True, slots=True)
class SebLogAiDeps:
    """Dependencies for the SEB log AI analysis layer.

    Satisfies the AiDeps protocol expected by gemini_call.generate_and_log.
    """

    gemini: GeminiClient
    limiter: GeminiLimiter
    logger: Logger
    ai_usage_logger: AiUsageLogger
