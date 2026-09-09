"""Grammar for macOS SEB log files.

macOS SEB uses a unified log format different from Windows:
- Timestamp format: YYYY/MM/DD HH:MM:SS:mmm  message
- No thread ID, severity, or module in the line format
- Multi-line blocks for Objective-C plist dictionaries
- Severity is inferred from message keywords
"""

from __future__ import annotations

import re
from datetime import datetime

from .base import BaseLogGrammar, ParsedLogEntry

MACOS_TIMESTAMP_PATTERN = re.compile(
    r"^(?P<ts>\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}:\d{3})  (?P<message>.*)$"
)

VERSION_PATTERN = re.compile(
    r"Safe Exam Browser version (?P<version>[\d.]+) \(Build (?P<build>\d+)\)"
)
BUNDLE_ID_PATTERN = re.compile(
    r"Bundle ID: (?P<bundle_id>[^,]+), executable: (?P<executable>.+)"
)
OS_VERSION_PATTERN = re.compile(
    r"OS version (?P<os_name>macOS) Version (?P<os_version>[\d.]+) \(Build (?P<build>\w+)\)"
)
LOCAL_HOSTNAME_PATTERN = re.compile(r"Local hostname: (?P<hostname>.+)")
DEVICE_NAME_PATTERN = re.compile(r"Device name: (?P<device_name>.+)")
USER_NAME_PATTERN = re.compile(r"User name: (?P<user_name>.+)")
FULL_USER_NAME_PATTERN = re.compile(r"Full user name: (?P<full_user_name>.+)")

ERROR_KEYWORDS = frozenset([
    "error", "failed", "failure", "cannot", "unable", "invalid", "exception",
    "crash", "fatal", "denied", "rejected", "not allowed", "wrong",
])
WARNING_KEYWORDS = frozenset([
    "warning", "warn", "deprecated", "caution", "attention", "not found",
    "missing", "unexpected", "retry", "timeout",
])

BLOCK_START_CHARS = {"(", "{"}
BLOCK_END_MAP = {"(": ")", "{": "}"}
MAX_BLOCK_CHARS = 2000
MAX_BLOCK_LINES = 100


class MacOSLogGrammar(BaseLogGrammar):
    """Grammar for macOS SEB unified log files.

    Format: YYYY/MM/DD HH:MM:SS:mmm  message

    Handles multi-line Objective-C plist blocks that span multiple lines,
    e.g.:
        2026/08/19 17:38:37:397  This not allowed process is running: {
            executable = "Google Chrome Helper";
            strongKill = 1;
        }

    Also handles unindented continuation lines for display info blocks.
    """

    def __init__(self) -> None:
        self._pending_block: list[str] | None = None
        self._pending_header_line: str | None = None
        self._pending_header_ts: datetime | None = None
        self._pending_header_line_number: int = 0
        self._pending_block_char: str | None = None
        self._pending_block_end_char: str | None = None
        self._block_depth: int = 0
        self._block_line_count: int = 0

    def parse_line(self, line: str, line_number: int) -> ParsedLogEntry | None:
        """Parse a single macOS log line.

        Handles multi-line block reassembly for plist dictionaries and arrays.
        """
        stripped = line.rstrip("\r\n")

        if not stripped:
            return None

        # If we're in the middle of a block, collect lines
        if self._pending_block is not None:
            return self._handle_block_continuation(stripped, line_number)

        # Try to match a timestamped line
        match = MACOS_TIMESTAMP_PATTERN.match(stripped)
        if match:
            ts_str = match.group("ts")
            message = match.group("message")
            timestamp = self._parse_timestamp(ts_str)

            # Check if this line starts a multi-line block
            trimmed_msg = message.rstrip()
            if trimmed_msg and trimmed_msg[-1] in BLOCK_START_CHARS:
                self._start_block(stripped, timestamp, line_number, trimmed_msg[-1])
                return None

            severity = self._infer_severity(message)
            return ParsedLogEntry(
                timestamp=timestamp,
                thread=None,
                severity=severity,
                module=None,
                message=message,
                raw_line=stripped,
                line_number=line_number,
                is_continuation=False,
            )

        # Unindented continuation line (display info like " is not built-in")
        if stripped.startswith(" ") and not stripped.startswith("  "):
            return ParsedLogEntry(
                timestamp=None,
                thread=None,
                severity="info",
                module=None,
                message=stripped.strip(),
                raw_line=stripped,
                line_number=line_number,
                is_continuation=True,
            )

        # Line doesn't match expected format, return as-is
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

    def _start_block(
        self,
        header_line: str,
        timestamp: datetime | None,
        line_number: int,
        start_char: str,
    ) -> None:
        """Start collecting a multi-line block."""
        self._pending_block = [header_line]
        self._pending_header_line = header_line
        self._pending_header_ts = timestamp
        self._pending_header_line_number = line_number
        self._pending_block_char = start_char
        self._pending_block_end_char = BLOCK_END_MAP[start_char]
        self._block_depth = 1
        self._block_line_count = 1

    def _handle_block_continuation(
        self, line: str, line_number: int  # noqa: ARG002
    ) -> ParsedLogEntry | None:
        """Handle a line that's part of a multi-line block."""
        assert self._pending_block is not None

        self._block_line_count += 1

        # Check for nested blocks
        for char in line:
            if char == self._pending_block_char:
                self._block_depth += 1
            elif char == self._pending_block_end_char:
                self._block_depth -= 1

        # Check if we should continue or emit
        total_chars = sum(len(l) for l in self._pending_block) + len(line)
        should_emit = (
            self._block_depth <= 0
            or total_chars >= MAX_BLOCK_CHARS
            or self._block_line_count >= MAX_BLOCK_LINES
        )

        if total_chars < MAX_BLOCK_CHARS:
            self._pending_block.append(line)

        if should_emit:
            # Emit the complete block
            full_message = "\n".join(self._pending_block)
            header_line = self._pending_header_line or ""
            timestamp = self._pending_header_ts
            header_line_number = self._pending_header_line_number

            severity = self._infer_severity(full_message)

            # Reset block state
            self._pending_block = None
            self._pending_header_line = None
            self._pending_header_ts = None
            self._pending_header_line_number = 0
            self._pending_block_char = None
            self._pending_block_end_char = None
            self._block_depth = 0
            self._block_line_count = 0

            return ParsedLogEntry(
                timestamp=timestamp,
                thread=None,
                severity=severity,
                module=None,
                message=full_message[:MAX_BLOCK_CHARS],
                raw_line=header_line,
                line_number=header_line_number,
                is_continuation=False,
            )

        return None

    def _parse_timestamp(self, ts_str: str) -> datetime | None:
        """Parse macOS timestamp format: YYYY/MM/DD HH:MM:SS:mmm"""
        try:
            # Replace last : with . to match strptime microseconds format
            normalized = ts_str[:19] + "." + ts_str[20:] + "000"
            return datetime.strptime(normalized, "%Y/%m/%d %H:%M:%S.%f")
        except (ValueError, IndexError):
            return None

    def _infer_severity(self, message: str) -> str:
        """Infer severity from message content keywords."""
        lower_msg = message.lower()

        for keyword in ERROR_KEYWORDS:
            if keyword in lower_msg:
                return "error"

        for keyword in WARNING_KEYWORDS:
            if keyword in lower_msg:
                return "warning"

        return "info"

    def parse_header(self, lines: list[str]) -> dict[str, str]:
        """Extract metadata from macOS log header lines.

        macOS SEB logs start with:
        - INITIALIZING SEB banner
        - Safe Exam Browser version X.Y.Z (Build NNNNN)
        - Bundle ID: org.safeexambrowser.SafeExamBrowser
        - OS version macOS Version X.Y.Z (Build XXXXX)
        - Local hostname: ...
        - Device name: ...
        - User name: ...
        """
        result: dict[str, str] = {}

        for line in lines[:20]:
            stripped = line.strip()
            if not stripped:
                continue

            # Remove timestamp prefix if present
            match = MACOS_TIMESTAMP_PATTERN.match(stripped)
            if match:
                content = match.group("message")
            else:
                content = stripped

            if ver_match := VERSION_PATTERN.search(content):
                result["program_version"] = ver_match.group("version")
                result["program_build"] = ver_match.group("build")

            if bundle_match := BUNDLE_ID_PATTERN.search(content):
                result["bundle_id"] = bundle_match.group("bundle_id")
                result["executable"] = bundle_match.group("executable")

            if os_match := OS_VERSION_PATTERN.search(content):
                result["os_name"] = os_match.group("os_name")
                result["os_version"] = os_match.group("os_version")
                result["os_build"] = os_match.group("build")

            if host_match := LOCAL_HOSTNAME_PATTERN.search(content):
                result["machine_name"] = host_match.group("hostname")

            if device_match := DEVICE_NAME_PATTERN.search(content):
                result["device_name"] = device_match.group("device_name")

            if user_match := USER_NAME_PATTERN.search(content):
                result["user_name"] = user_match.group("user_name")

            if full_user_match := FULL_USER_NAME_PATTERN.search(content):
                result["full_user_name"] = full_user_match.group("full_user_name")

        return result

    def reset(self) -> None:
        """Reset block parsing state between files."""
        self._pending_block = None
        self._pending_header_line = None
        self._pending_header_ts = None
        self._pending_header_line_number = 0
        self._pending_block_char = None
        self._pending_block_end_char = None
        self._block_depth = 0
        self._block_line_count = 0
