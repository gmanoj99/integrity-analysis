"""Bulk event aggregator for process termination sweeps.

The Client.log can contain tens of thousands of process termination entries
during startup when blacklisted applications are terminated. This module
aggregates these into summary events per application rather than keeping
every single log line.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from ..contracts import BulkTerminationEvent, LogFileType
from ..grammars.base import ParsedLogEntry

BLACKLIST_BELONGS_PATTERN = re.compile(
    r"Process '(?P<name>[^']+)' \((?P<pid>\d+)\) belongs to "
    r"(?:application '(?P<app>[^']+)'|blacklisted application)"
)

NEEDS_TERMINATION_PATTERN = re.compile(
    r"Process '(?P<name>[^']+)' \((?P<pid>\d+)\).*needs to be terminated"
)

CLOSE_MESSAGE_FAILED_PATTERN = re.compile(
    r"\[Process '(?P<name>[^']+)' \((?P<pid>\d+)\)\].*Failed to send close message"
)

CLOSE_WINDOW_FAILED_PATTERN = re.compile(
    r"\[Process '(?P<name>[^']+)' \((?P<pid>\d+)\)\].*Failed to close main window"
)

KILL_FAILED_PATTERN = re.compile(
    r"\[Process '(?P<name>[^']+)' \((?P<pid>\d+)\)\].*Failed to kill process"
)

ATTEMPTING_TO_KILL_PATTERN = re.compile(
    r"\[Process '(?P<name>[^']+)' \((?P<pid>\d+)\)\].*Attempting to kill process"
)

SUCCESSFULLY_TERMINATED_PATTERN = re.compile(
    r"Successfully terminated process '(?P<name>[^']+)' \((?P<pid>\d+)\)"
)

ATTEMPTING_TO_CLOSE_PATTERN = re.compile(
    r"\[Process '(?P<name>[^']+)' \((?P<pid>\d+)\)\].*Attempting to close process"
)


@dataclass
class ProcessTerminationRecord:
    """Tracks termination attempts for a single process instance."""

    application_name: str
    pid: int
    first_seen: datetime | None = None
    last_seen: datetime | None = None
    close_attempts: int = 0
    close_message_failures: int = 0
    close_window_failures: int = 0
    kill_attempts: int = 0
    kill_failures: int = 0
    terminated: bool = False

    def update_timestamp(self, ts: datetime | None) -> None:
        """Update first/last seen timestamps."""
        if ts is None:
            return
        if self.first_seen is None or ts < self.first_seen:
            self.first_seen = ts
        if self.last_seen is None or ts > self.last_seen:
            self.last_seen = ts


@dataclass
class ApplicationTerminationGroup:
    """Groups termination records by application name."""

    application_name: str
    processes: dict[int, ProcessTerminationRecord] = field(default_factory=dict)
    first_seen: datetime | None = None
    last_seen: datetime | None = None

    def get_or_create_process(self, pid: int) -> ProcessTerminationRecord:
        """Get or create a process record."""
        if pid not in self.processes:
            self.processes[pid] = ProcessTerminationRecord(
                application_name=self.application_name,
                pid=pid,
            )
        return self.processes[pid]

    def update_timestamp(self, ts: datetime | None) -> None:
        """Update group-level timestamps."""
        if ts is None:
            return
        if self.first_seen is None or ts < self.first_seen:
            self.first_seen = ts
        if self.last_seen is None or ts > self.last_seen:
            self.last_seen = ts

    def to_bulk_event(self) -> BulkTerminationEvent:
        """Convert to BulkTerminationEvent contract."""
        close_msg_failures = sum(p.close_message_failures for p in self.processes.values())
        close_window_failures = sum(p.close_window_failures for p in self.processes.values())
        kill_failures = sum(p.kill_failures for p in self.processes.values())
        kill_attempts = sum(p.kill_attempts for p in self.processes.values())
        all_terminated = all(p.terminated for p in self.processes.values())

        return BulkTerminationEvent(
            application_name=self.application_name,
            instance_count=len(self.processes),
            close_message_failures=close_msg_failures,
            close_window_failures=close_window_failures,
            kill_failures=kill_failures,
            force_kill_needed=kill_attempts > 0,
            first_seen=self.first_seen,
            last_seen=self.last_seen,
            all_terminated=all_terminated,
        )


class BulkEventAggregator:
    """Aggregates bulk process termination events.

    This aggregator detects and groups the startup process termination sweep
    that can produce tens of thousands of log lines when multiple blacklisted
    applications are running.
    """

    def __init__(
        self,
        time_window_seconds: float = 60.0,
    ) -> None:
        """Initialize aggregator.

        Args:
            time_window_seconds: Maximum time span for events to be considered
                                part of the same bulk operation.
        """
        self._time_window = timedelta(seconds=time_window_seconds)
        self._groups: dict[str, ApplicationTerminationGroup] = defaultdict(
            lambda: ApplicationTerminationGroup(application_name="")
        )
        self._in_bulk_operation = False
        self._bulk_start_time: datetime | None = None

    def check_entry(
        self, entry: ParsedLogEntry, file_type: LogFileType
    ) -> bool:
        """Check if an entry is part of a bulk termination operation.

        Returns True if the entry was absorbed into a bulk event, False if
        it should be processed normally.
        """
        if file_type != LogFileType.CLIENT:
            return False

        message = entry.message
        raw_line = entry.raw_line

        if match := BLACKLIST_BELONGS_PATTERN.search(message):
            self._handle_blacklist_match(entry, match)
            return True

        if match := NEEDS_TERMINATION_PATTERN.search(message):
            self._handle_needs_termination(entry, match)
            return True

        if match := ATTEMPTING_TO_CLOSE_PATTERN.search(raw_line):
            self._handle_close_attempt(entry, match)
            return True

        if match := CLOSE_MESSAGE_FAILED_PATTERN.search(raw_line):
            self._handle_close_message_failed(entry, match)
            return True

        if match := CLOSE_WINDOW_FAILED_PATTERN.search(raw_line):
            self._handle_close_window_failed(entry, match)
            return True

        if match := ATTEMPTING_TO_KILL_PATTERN.search(raw_line):
            self._handle_kill_attempt(entry, match)
            return True

        if match := KILL_FAILED_PATTERN.search(raw_line):
            self._handle_kill_failed(entry, match)
            return True

        if match := SUCCESSFULLY_TERMINATED_PATTERN.search(message):
            self._handle_successfully_terminated(entry, match)
            return True

        return False

    def _get_or_create_group(self, app_name: str) -> ApplicationTerminationGroup:
        """Get or create an application group."""
        normalized = app_name.lower()
        if normalized not in self._groups:
            self._groups[normalized] = ApplicationTerminationGroup(
                application_name=app_name
            )
        return self._groups[normalized]

    def _handle_blacklist_match(
        self, entry: ParsedLogEntry, match: re.Match[str]
    ) -> None:
        """Handle a 'belongs to blacklisted application' log entry."""
        name = match.group("name")
        pid = int(match.group("pid"))
        app = match.group("app") or name

        group = self._get_or_create_group(app)
        group.application_name = app
        record = group.get_or_create_process(pid)
        record.update_timestamp(entry.timestamp)
        group.update_timestamp(entry.timestamp)

    def _handle_needs_termination(
        self, entry: ParsedLogEntry, match: re.Match[str]
    ) -> None:
        """Handle a 'needs to be terminated' log entry."""
        name = match.group("name")
        pid = int(match.group("pid"))

        group = self._get_or_create_group(name)
        group.application_name = name
        record = group.get_or_create_process(pid)
        record.update_timestamp(entry.timestamp)
        group.update_timestamp(entry.timestamp)

    def _handle_close_attempt(
        self, entry: ParsedLogEntry, match: re.Match[str]
    ) -> None:
        """Handle an 'Attempting to close process' log entry."""
        name = match.group("name")
        pid = int(match.group("pid"))

        group = self._get_or_create_group(name)
        record = group.get_or_create_process(pid)
        record.close_attempts += 1
        record.update_timestamp(entry.timestamp)
        group.update_timestamp(entry.timestamp)

    def _handle_close_message_failed(
        self, entry: ParsedLogEntry, match: re.Match[str]
    ) -> None:
        """Handle a 'Failed to send close message' log entry."""
        name = match.group("name")
        pid = int(match.group("pid"))

        group = self._get_or_create_group(name)
        record = group.get_or_create_process(pid)
        record.close_message_failures += 1
        record.update_timestamp(entry.timestamp)
        group.update_timestamp(entry.timestamp)

    def _handle_close_window_failed(
        self, entry: ParsedLogEntry, match: re.Match[str]
    ) -> None:
        """Handle a 'Failed to close main window' log entry."""
        name = match.group("name")
        pid = int(match.group("pid"))

        group = self._get_or_create_group(name)
        record = group.get_or_create_process(pid)
        record.close_window_failures += 1
        record.update_timestamp(entry.timestamp)
        group.update_timestamp(entry.timestamp)

    def _handle_kill_attempt(
        self, entry: ParsedLogEntry, match: re.Match[str]
    ) -> None:
        """Handle an 'Attempting to kill process' log entry."""
        name = match.group("name")
        pid = int(match.group("pid"))

        group = self._get_or_create_group(name)
        record = group.get_or_create_process(pid)
        record.kill_attempts += 1
        record.update_timestamp(entry.timestamp)
        group.update_timestamp(entry.timestamp)

    def _handle_kill_failed(
        self, entry: ParsedLogEntry, match: re.Match[str]
    ) -> None:
        """Handle a 'Failed to kill process' log entry."""
        name = match.group("name")
        pid = int(match.group("pid"))

        group = self._get_or_create_group(name)
        record = group.get_or_create_process(pid)
        record.kill_failures += 1
        record.update_timestamp(entry.timestamp)
        group.update_timestamp(entry.timestamp)

    def _handle_successfully_terminated(
        self, entry: ParsedLogEntry, match: re.Match[str]
    ) -> None:
        """Handle a 'Successfully terminated process' log entry."""
        name = match.group("name")
        pid = int(match.group("pid"))

        group = self._get_or_create_group(name)
        record = group.get_or_create_process(pid)
        record.terminated = True
        record.update_timestamp(entry.timestamp)
        group.update_timestamp(entry.timestamp)

    def get_bulk_events(self) -> list[BulkTerminationEvent]:
        """Get all aggregated bulk termination events."""
        events = []
        for group in self._groups.values():
            if group.processes:
                events.append(group.to_bulk_event())
        return sorted(events, key=lambda e: e.instance_count, reverse=True)

    def get_absorbed_entry_count(self) -> int:
        """Get total number of log entries absorbed into bulk events."""
        count = 0
        for group in self._groups.values():
            for proc in group.processes.values():
                count += 1
                count += proc.close_attempts
                count += proc.close_message_failures
                count += proc.close_window_failures
                count += proc.kill_attempts
                count += proc.kill_failures
                if proc.terminated:
                    count += 1
        return count

    def reset(self) -> None:
        """Reset aggregator for a new session."""
        self._groups.clear()
        self._in_bulk_operation = False
        self._bulk_start_time = None
