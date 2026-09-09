"""Base grammar interface for log parsing."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol


@dataclass
class ParsedLogEntry:
    """A single parsed log entry."""

    timestamp: datetime | None
    thread: str | None
    severity: str | None
    module: str | None
    message: str
    raw_line: str
    line_number: int
    is_continuation: bool = False


class LogGrammar(Protocol):
    """Protocol for log file grammar parsers."""

    def parse_line(self, line: str, line_number: int) -> ParsedLogEntry | None:
        """Parse a single log line.

        Returns None if the line is a header/comment line that should be skipped.
        Returns ParsedLogEntry with is_continuation=True for continuation lines
        that should be appended to the previous entry.
        """
        ...

    def parse_header(self, lines: list[str]) -> dict[str, str]:
        """Extract metadata from header lines at the start of the log file.

        Returns a dict with keys like 'program_version', 'os_info', 'machine_name',
        'runtime_id', etc.
        """
        ...


class BaseLogGrammar(ABC):
    """Base implementation of LogGrammar with common utilities."""

    @abstractmethod
    def parse_line(self, line: str, line_number: int) -> ParsedLogEntry | None:
        """Parse a single log line."""
        ...

    @abstractmethod
    def parse_header(self, lines: list[str]) -> dict[str, str]:
        """Extract metadata from header lines."""
        ...

    def normalize_severity(self, severity: str) -> str:
        """Normalize severity string to standard form."""
        mapping = {
            "WARNING": "warning",
            "WARN": "warning",
            "ERROR": "error",
            "ERR": "error",
            "INFO": "info",
            "DEBUG": "debug",
            "VERBOSE": "verbose",
            "FATAL": "fatal",
        }
        return mapping.get(severity.upper(), severity.lower())
