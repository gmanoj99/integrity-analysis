"""Grammar for CEF/Chromium-style Browser.log files."""

from __future__ import annotations

import re
from datetime import datetime

from .base import BaseLogGrammar, ParsedLogEntry

CEF_MAIN_PATTERN = re.compile(
    r"^\[(?P<pid>\d+):(?P<tid>\d+):"
    r"(?P<month>\d{2})(?P<day>\d{2})/"
    r"(?P<hour>\d{2})(?P<min>\d{2})(?P<sec>\d{2})\.(?P<ms>\d{3}):"
    r"(?P<severity>VERBOSE\d*|INFO|WARNING|ERROR|FATAL):"
    r"(?P<source>[^\]]+)\]"
    r"\s*(?P<message>.*)$"
)

CONSOLE_LOG_PATTERN = re.compile(
    r'^INFO:CONSOLE:\d+\]\s*"(?P<message>.+)",\s*source:\s*(?P<url>\S+)\s*\((?P<line>\d+)\)$'
)


class CefBrowserGrammar(BaseLogGrammar):
    """Grammar for CEF/Chromium Browser.log files.

    Format: [pid:tid:MMDD/HHMMSS.mmm:LEVEL:file.cc:line] message

    Note: The timestamp does not include a year. We infer the year from
    the manifest's startTime or use the current year as fallback.
    """

    def __init__(self, reference_year: int | None = None) -> None:
        """Initialize the grammar.

        Args:
            reference_year: The year to use for timestamps (from manifest.startTime).
                           If None, uses the current year.
        """
        self._reference_year = reference_year or datetime.now().year

    def set_reference_year(self, year: int) -> None:
        """Set the reference year for timestamp parsing."""
        self._reference_year = year

    def parse_line(self, line: str, line_number: int) -> ParsedLogEntry | None:
        stripped = line.rstrip("\r\n")

        if not stripped:
            return None

        match = CEF_MAIN_PATTERN.match(stripped)
        if match:
            try:
                timestamp = datetime(
                    year=self._reference_year,
                    month=int(match.group("month")),
                    day=int(match.group("day")),
                    hour=int(match.group("hour")),
                    minute=int(match.group("min")),
                    second=int(match.group("sec")),
                    microsecond=int(match.group("ms")) * 1000,
                )
            except ValueError:
                timestamp = None

            pid = match.group("pid")
            tid = match.group("tid")
            source = match.group("source")
            message = match.group("message").strip()

            is_console_log = ":CONSOLE:" in stripped
            if is_console_log:
                module = "CONSOLE"
            else:
                module = self._extract_module_from_source(source)

            return ParsedLogEntry(
                timestamp=timestamp,
                thread=f"{pid}:{tid}",
                severity=self.normalize_severity(match.group("severity")),
                module=module,
                message=message,
                raw_line=stripped,
                line_number=line_number,
                is_continuation=False,
            )

        return ParsedLogEntry(
            timestamp=None,
            thread=None,
            severity=None,
            module=None,
            message=stripped,
            raw_line=stripped,
            line_number=line_number,
            is_continuation=True,
        )

    def _extract_module_from_source(self, source: str) -> str | None:
        """Extract module name from CEF source path (e.g., 'chrome\\browser\\foo.cc:123')."""
        if not source:
            return None

        parts = source.rsplit(":", 1)[0]

        if "\\" in parts:
            parts = parts.replace("\\", "/")

        path_parts = parts.rsplit("/", 1)
        filename = path_parts[-1] if path_parts else parts

        if filename.endswith((".cc", ".cpp", ".h", ".mm")):
            filename = filename.rsplit(".", 1)[0]

        return filename

    def parse_header(self, lines: list[str]) -> dict[str, str]:
        """CEF Browser.log files don't have a header section.

        Version info may be extracted from log content instead.
        """
        result: dict[str, str] = {}

        for line in lines[:20]:
            if "CefSharp" in line:
                if match := re.search(r"CefSharp(?:\s+|/)(\d+\.\d+\.\d+)", line):
                    result["cefsharp_version"] = match.group(1)
            if "CEF/" in line or "CEF " in line:
                if match := re.search(r"CEF[/ ](\d+\.\d+\.\d+)", line):
                    result["cef_version"] = match.group(1)
            if "Chromium/" in line or "Chromium " in line:
                if match := re.search(r"Chromium[/ ](\d+\.\d+\.\d+\.\d+)", line):
                    result["chromium_version"] = match.group(1)

        return result

    def extract_console_message(self, message: str) -> tuple[str, str | None, int | None]:
        """Extract console message details from a CONSOLE log entry.

        Returns: (message_content, source_url, line_number)
        """
        if match := CONSOLE_LOG_PATTERN.search(message):
            return (
                match.group("message"),
                match.group("url"),
                int(match.group("line")),
            )
        return (message, None, None)
