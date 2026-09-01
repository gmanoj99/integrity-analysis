"""Grammar for macOS SEB log files (stub).

This module is a placeholder for macOS log parsing support.
macOS SEB uses different log formats:
- 5-level SEBLogLevel enum (vs. 4-level on Windows)
- WebKit console format (vs. CEF/Chromium)
- One-time process snapshot (vs. continuous monitoring)

Implementation will be completed once macOS sample logs are available.
"""

from __future__ import annotations

from .base import BaseLogGrammar, ParsedLogEntry


class MacOSLogGrammar(BaseLogGrammar):
    """Grammar for macOS SEB log files.

    Note: This is a stub implementation. macOS SEB uses different log formats
    than Windows, including:
    - Different timestamp formats
    - SEBLogLevel enum with 5 levels
    - WebKit-based browser logs instead of CEF
    - Different process monitoring approach

    This will be implemented once macOS sample logs are provided.
    """

    def __init__(self) -> None:
        pass

    def parse_line(self, line: str, line_number: int) -> ParsedLogEntry | None:
        """Parse a single macOS log line.

        Currently returns None for all lines since the format is unknown.
        """
        stripped = line.rstrip("\r\n")

        if not stripped:
            return None

        return ParsedLogEntry(
            timestamp=None,
            thread=None,
            severity=None,
            module=None,
            message=stripped,
            raw_line=stripped,
            line_number=line_number,
            is_continuation=False,
        )

    def parse_header(self, lines: list[str]) -> dict[str, str]:
        """Extract metadata from macOS log header.

        Currently returns empty dict since format is unknown.
        """
        return {}
