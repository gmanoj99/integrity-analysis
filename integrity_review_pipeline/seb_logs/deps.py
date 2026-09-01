"""Dependency protocols for SEB log analysis pipeline."""

from __future__ import annotations

from collections.abc import Iterator
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
    """Dependencies for the SEB log analysis pipeline."""

    object_store: SebLogObjectStore
    logger: Logger
