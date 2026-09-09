"""Grammar for Windows .NET-style log files (Runtime.log, Client.log, Service.log)."""

from __future__ import annotations

import re
from datetime import datetime

from .base import BaseLogGrammar, ParsedLogEntry

TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S.%f"

MAIN_PATTERN = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})"
    r"\s+\[(?P<thread>\d+)\]"
    r"\s+-\s+(?P<severity>DEBUG|INFO|WARNING|ERROR):"
    r"\s*(?:\[(?P<module>[^\]]+)\])?"
    r"\s*(?P<message>.*)$"
)

HEADER_COMMENT_PATTERN = re.compile(r"^(?:/\*|#)")


def _strip_utf8_bom(text: str) -> str:
    return text.lstrip("\ufeff")

EXCEPTION_CONTINUATION_PATTERN = re.compile(
    r"^(?:Exception Message:|Exception Type:|at |   at |\s+---> )"
)

HEADER_APP_STARTED = re.compile(
    r"^#\s*Application started at (?P<start_time>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})"
)

HEADER_OS_INFO = re.compile(
    r"^#\s*Running on (?P<os_name>[^,]+), (?P<os_version>[^\(]+)\s*\((?P<arch>[^)]+)\)"
)

HEADER_MACHINE_INFO = re.compile(
    r"^#\s*Computer '(?P<machine_name>[^']+)' is a (?P<machine_model>.+) manufactured by (?P<manufacturer>.+)\."
)

HEADER_RUNTIME_ID = re.compile(r"^#\s*Runtime-ID:\s*(?P<runtime_id>.+)$")

HEADER_VERSION = re.compile(
    r"^/\*\s*(?:Topin Secure Browser|Safe Exam Browser),\s*Version\s*(?P<version>[^,]+),\s*Build\s*(?P<build>.+)$"
)


class WindowsDotNetGrammar(BaseLogGrammar):
    """Grammar for Windows .NET-style SEB log files.

    Handles:
    - Runtime.log
    - Client.log
    - Service.log

    Format: yyyy-MM-dd HH:mm:ss.fff [thread] - SEVERITY: [Module] message
    """

    def __init__(self) -> None:
        self._pending_multiline: ParsedLogEntry | None = None

    def parse_line(self, line: str, line_number: int) -> ParsedLogEntry | None:
        stripped = _strip_utf8_bom(line.rstrip("\r\n"))

        if not stripped:
            return None

        if HEADER_COMMENT_PATTERN.match(stripped):
            return None

        match = MAIN_PATTERN.match(stripped)
        if match:
            ts_str = match.group("timestamp")
            try:
                timestamp = datetime.strptime(ts_str, TIMESTAMP_FORMAT)
            except ValueError:
                timestamp = None

            return ParsedLogEntry(
                timestamp=timestamp,
                thread=match.group("thread"),
                severity=self.normalize_severity(match.group("severity")),
                module=match.group("module"),
                message=match.group("message").strip(),
                raw_line=stripped,
                line_number=line_number,
                is_continuation=False,
            )

        if EXCEPTION_CONTINUATION_PATTERN.match(stripped.lstrip()):
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

        if stripped.startswith("   ") or stripped.startswith("\t"):
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
        """Extract metadata from the header lines of a Windows .NET log file."""
        result: dict[str, str] = {}

        for line in lines:
            stripped = _strip_utf8_bom(line.strip())
            if not stripped:
                continue

            if not stripped.startswith(("/", "#")):
                break

            if match := HEADER_VERSION.match(stripped):
                result["program_version"] = match.group("version")
                result["program_build"] = match.group("build")
                continue

            if match := HEADER_APP_STARTED.match(stripped):
                result["app_started_at"] = match.group("start_time")
                continue

            if match := HEADER_OS_INFO.match(stripped):
                result["os_name"] = match.group("os_name")
                result["os_version"] = match.group("os_version").strip()
                result["architecture"] = match.group("arch")
                continue

            if match := HEADER_MACHINE_INFO.match(stripped):
                result["machine_name"] = match.group("machine_name")
                result["machine_model"] = match.group("machine_model").strip()
                result["machine_manufacturer"] = match.group("manufacturer")
                continue

            if match := HEADER_RUNTIME_ID.match(stripped):
                result["runtime_id"] = match.group("runtime_id")
                continue

        return result
