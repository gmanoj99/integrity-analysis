"""Session reducer - orchestrates log reduction for a single session.

This module is the main orchestrator for Process 2 (Deterministic Analysis).
It streams each log file through the appropriate grammar, routes entries through
noise/bulk/signal rules, and assembles the final ReducedSessionEvidence.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime

from ..contracts import (
    BrowserActivitySummary,
    DisplayInfo,
    EffectiveConfigField,
    EnvironmentInfo,
    ForegroundWindowSpan,
    LogFileType,
    MonitoringCoverage,
    ProcessRecord,
    RawLogFileMeta,
    ReducedSessionEvidence,
    SebLogSessionRef,
    Severity,
    TimelineEvent,
)
from ..deps import SebLogPipelineDeps
from ..discovery import get_log_file_key
from ..grammars.base import ParsedLogEntry
from ..grammars.cef_browser import CefBrowserGrammar
from ..grammars.windows_dotnet import WindowsDotNetGrammar
from .bulk_event_aggregator import BulkEventAggregator
from .correlator import CoverageGapDetector, PhaseTracker, SignalCorrelator
from .noise_rules import NoiseDetector
from .signal_rules import SignalDetector


PROCESS_STARTED_PATTERN = re.compile(
    r"Process '(?P<name>[^']+)' \((?P<pid>\d+)\) has been started"
    r"(?:.*Original Name: '(?P<orig>[^']*)')?"
    r"(?:.*Path: '(?P<path>[^']*)')?"
    r"(?:.*Signature: (?P<sig>[^\]]+))?"
)

WINDOW_CHANGE_PATTERN = re.compile(
    r"Window has changed from '(?P<from>[^']+)' \((?P<from_h>\d+)\) "
    r"to '(?P<to>[^']+)' \((?P<to_h>\d+)\)"
)

DISPLAY_DETECTED_PATTERN = re.compile(
    r"Detected (?P<count>\d+) active displays?, (?P<allowed>\d+) (?:is|are) allowed"
)

DISPLAY_INFO_PATTERN = re.compile(
    r"Display '(?P<id>[^']+)'.*?(?:internal|external)?.*?(?:active|inactive)?"
)

MONITOR_STARTED_PATTERN = re.compile(
    r"Started monitoring (?P<monitor>[^.]+)\."
)

KEYBOARD_INTERCEPTION_PATTERN = re.compile(
    r"Starting keyboard interception"
)

MOUSE_INTERCEPTION_PATTERN = re.compile(
    r"Starting mouse interception"
)

WINDOW_GUARD_PATTERN = re.compile(
    r"(?:Activated|Deactivated) window guard"
)

BROWSER_WINDOW_CREATED_PATTERN = re.compile(
    r"Created (?:new )?(?:browser )?window"
)

NAVIGATED_TO_PATTERN = re.compile(
    r"Navigated to '(?P<url>[^']+)'"
)

BLOCKED_REQUEST_PATTERN = re.compile(
    r"Blocked (?:navigation|request)"
)

BLOCKED_POPUP_PATTERN = re.compile(
    r"Blocked popup"
)

DOWNLOAD_ATTEMPTED_PATTERN = re.compile(
    r"Download.*(?:started|requested|attempted)"
)

UPLOAD_BLOCKED_PATTERN = re.compile(
    r"Upload.*blocked"
)

USER_IDENTIFIER_PATTERN = re.compile(
    r"User identifier '(?P<id>[^']+)' detected"
)

ENGINE_VERSION_PATTERN = re.compile(
    r"Engine Version:.*Chromium (?P<chromium>[\d.]+).*CEF (?P<cef>[\d.]+).*CefSharp (?P<cefsharp>[\d.]+)"
)


@dataclass
class SessionReducerState:
    """Mutable state accumulated during session reduction."""

    processes: dict[int, ProcessRecord] = field(default_factory=dict)
    foreground_spans: list[ForegroundWindowSpan] = field(default_factory=list)
    current_window_title: str | None = None
    current_window_start: datetime | None = None
    displays: list[DisplayInfo] = field(default_factory=list)
    monitors_armed: list[str] = field(default_factory=list)
    browser_windows_created: int = 0
    navigations: list[str] = field(default_factory=list)
    blocked_requests: int = 0
    blocked_popups: int = 0
    downloads_attempted: int = 0
    uploads_blocked: int = 0
    user_identifiers: list[str] = field(default_factory=list)
    page_errors: list[str] = field(default_factory=list)
    chromium_version: str | None = None
    cef_version: str | None = None
    cefsharp_version: str | None = None
    timeline_events: list[TimelineEvent] = field(default_factory=list)
    environment: EnvironmentInfo | None = None
    effective_config: list[EffectiveConfigField] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)


class SessionReducer:
    """Reduces a single session's logs into structured evidence."""

    def __init__(
        self,
        deps: SebLogPipelineDeps,
    ) -> None:
        self._deps = deps
        self._noise_detector = NoiseDetector()
        self._bulk_aggregator = BulkEventAggregator()
        self._signal_detector = SignalDetector()
        self._correlator = SignalCorrelator()
        self._gap_detector = CoverageGapDetector()
        self._phase_tracker = PhaseTracker()
        self._state = SessionReducerState()
        self._windows_grammar = WindowsDotNetGrammar()
        self._browser_grammar = CefBrowserGrammar()

    def reduce_session(
        self,
        session_ref: SebLogSessionRef,
    ) -> ReducedSessionEvidence:
        """Reduce a session's logs to structured evidence."""
        self._reset()

        if session_ref.manifest and session_ref.manifest.start_time:
            self._browser_grammar.set_reference_year(
                session_ref.manifest.start_time.year
            )

        analysed_files: list[RawLogFileMeta] = []
        total_raw_bytes = 0
        last_timestamp: datetime | None = None

        for file_type in [
            LogFileType.RUNTIME,
            LogFileType.SERVICE,
            LogFileType.CLIENT,
            LogFileType.BROWSER,
        ]:
            if file_type in session_ref.files_present:
                meta, file_last_ts = self._process_log_file(session_ref, file_type)
                analysed_files.append(meta)
                total_raw_bytes += meta.size_bytes
                if file_last_ts and (last_timestamp is None or file_last_ts > last_timestamp):
                    last_timestamp = file_last_ts
            else:
                analysed_files.append(
                    RawLogFileMeta(
                        file_type=file_type,
                        s3_key="",
                        missing=True,
                    )
                )

        phases = self._phase_tracker.finalize(last_timestamp)

        self._finalize_foreground_spans(last_timestamp)

        signals = self._signal_detector.get_signals()
        for signal in signals:
            self._correlator.add_signal(signal)
        correlated_incidents = self._correlator.correlate()

        coverage_gaps = self._gap_detector.detect_gaps()

        noise_summary = self._noise_detector.get_summaries()

        bulk_terminations = self._bulk_aggregator.get_bulk_events()

        browser_activity = BrowserActivitySummary(
            windows_created=self._state.browser_windows_created,
            navigations=self._state.navigations[:20],
            blocked_requests=self._state.blocked_requests,
            blocked_popups=self._state.blocked_popups,
            downloads_attempted=self._state.downloads_attempted,
            uploads_blocked=self._state.uploads_blocked,
            user_identifiers_detected=self._state.user_identifiers,
            page_errors=self._state.page_errors[:10],
            chromium_version=self._state.chromium_version,
            cef_version=self._state.cef_version,
            cefsharp_version=self._state.cefsharp_version,
        )

        monitoring = MonitoringCoverage(
            monitors_armed=self._state.monitors_armed,
            coverage_gaps=coverage_gaps,
        )

        if LogFileType.SERVICE not in session_ref.files_present:
            monitoring.service_ignored = True
            self._state.caveats.append(
                "Service log missing - OS lockdowns may not have been applied"
            )

        reduced_json_estimate = 5000

        reduction_stats = {
            "raw_bytes": total_raw_bytes,
            "reduced_estimate_bytes": reduced_json_estimate,
            "reduction_ratio": (
                total_raw_bytes / reduced_json_estimate if reduced_json_estimate > 0 else 0
            ),
            "signals_extracted": len(signals),
            "noise_entries_collapsed": sum(ns.count for ns in noise_summary),
            "bulk_entries_collapsed": self._bulk_aggregator.get_absorbed_entry_count(),
            "correlated_incidents": len(correlated_incidents),
        }

        return ReducedSessionEvidence(
            session_ref=session_ref,
            analysed_files=analysed_files,
            environment=self._state.environment,
            displays=self._state.displays,
            effective_config=self._state.effective_config,
            session_phases=phases,
            bulk_terminations=bulk_terminations,
            processes=list(self._state.processes.values()),
            foreground_activity=self._state.foreground_spans,
            browser_activity=browser_activity,
            monitoring_coverage=monitoring,
            noise_summary=noise_summary,
            signals=signals,
            correlated_incidents=correlated_incidents,
            timeline=self._state.timeline_events[:100],
            unresolved=self._state.unresolved,
            caveats=self._state.caveats,
            reduction_stats=reduction_stats,
        )

    def _reset(self) -> None:
        """Reset all state for a new session."""
        self._noise_detector.reset()
        self._bulk_aggregator.reset()
        self._signal_detector.reset()
        self._correlator.reset()
        self._gap_detector.reset()
        self._phase_tracker = PhaseTracker()
        self._state = SessionReducerState()

    def _process_log_file(
        self,
        session_ref: SebLogSessionRef,
        file_type: LogFileType,
    ) -> tuple[RawLogFileMeta, datetime | None]:
        """Process a single log file."""
        key = get_log_file_key(session_ref, file_type)

        try:
            size_bytes = self._deps.object_store.get_object_size(key)
        except Exception:
            size_bytes = 0

        grammar = self._get_grammar_for_file(file_type)

        header_lines: list[str] = []
        line_count = 0
        first_timestamp: datetime | None = None
        last_timestamp: datetime | None = None

        try:
            lines_iter: Iterator[str] = self._deps.object_store.iter_lines(key)

            for line in lines_iter:
                line_count += 1

                if line_count <= 15:
                    header_lines.append(line)

                entry = grammar.parse_line(line, line_count)
                if entry is None:
                    continue

                if entry.timestamp:
                    if first_timestamp is None:
                        first_timestamp = entry.timestamp
                    last_timestamp = entry.timestamp

                    self._gap_detector.record_timestamp(entry.timestamp, file_type)

                self._phase_tracker.check_line(entry.raw_line, entry.timestamp)

                self._process_entry(entry, file_type)

        except Exception as e:
            self._deps.logger.error(
                "Error processing log file",
                key=key,
                file_type=file_type.value,
                error=str(e),
            )
            self._state.caveats.append(f"Error processing {file_type.value}: {str(e)}")

        if header_lines and file_type in (LogFileType.RUNTIME, LogFileType.CLIENT):
            self._extract_environment_from_header(header_lines, grammar)

        return (
            RawLogFileMeta(
                file_type=file_type,
                s3_key=key,
                line_count=line_count,
                first_timestamp=first_timestamp,
                last_timestamp=last_timestamp,
                missing=False,
                size_bytes=size_bytes,
            ),
            last_timestamp,
        )

    def _get_grammar_for_file(self, file_type: LogFileType):
        """Get the appropriate grammar for a file type."""
        if file_type == LogFileType.BROWSER:
            return self._browser_grammar
        return self._windows_grammar

    def _process_entry(
        self, entry: ParsedLogEntry, file_type: LogFileType
    ) -> None:
        """Process a single parsed log entry through all detectors."""
        if self._bulk_aggregator.check_entry(entry, file_type):
            return

        if self._noise_detector.check_entry(entry, file_type):
            return

        self._signal_detector.check_entry(entry, file_type)

        self._extract_metadata(entry, file_type)

    def _extract_metadata(
        self, entry: ParsedLogEntry, file_type: LogFileType
    ) -> None:
        """Extract metadata from log entries (processes, windows, etc.)."""
        message = entry.message
        raw_line = entry.raw_line
        
        if match := PROCESS_STARTED_PATTERN.search(raw_line):
            pid = int(match.group("pid"))
            if pid not in self._state.processes:
                self._state.processes[pid] = ProcessRecord(
                    name=match.group("name"),
                    pid=pid,
                    path=match.group("path"),
                    original_name=match.group("orig"),
                    first_seen=entry.timestamp,
                    last_seen=entry.timestamp,
                )
            else:
                self._state.processes[pid].last_seen = entry.timestamp

        if match := WINDOW_CHANGE_PATTERN.search(raw_line):
            self._record_window_change(
                match.group("from"),
                int(match.group("from_h")),
                match.group("to"),
                int(match.group("to_h")),
                entry.timestamp,
            )

        if match := MONITOR_STARTED_PATTERN.search(message):
            monitor_name = match.group("monitor")
            if monitor_name not in self._state.monitors_armed:
                self._state.monitors_armed.append(monitor_name)

        if KEYBOARD_INTERCEPTION_PATTERN.search(message):
            if "keyboard" not in self._state.monitors_armed:
                self._state.monitors_armed.append("keyboard")

        if MOUSE_INTERCEPTION_PATTERN.search(message):
            if "mouse" not in self._state.monitors_armed:
                self._state.monitors_armed.append("mouse")

        if file_type == LogFileType.BROWSER or file_type == LogFileType.CLIENT:
            if BROWSER_WINDOW_CREATED_PATTERN.search(message):
                self._state.browser_windows_created += 1

            if match := NAVIGATED_TO_PATTERN.search(message):
                url = match.group("url")
                if len(self._state.navigations) < 50:
                    self._state.navigations.append(url)

            if BLOCKED_REQUEST_PATTERN.search(message):
                self._state.blocked_requests += 1

            if BLOCKED_POPUP_PATTERN.search(message):
                self._state.blocked_popups += 1

            if DOWNLOAD_ATTEMPTED_PATTERN.search(message):
                self._state.downloads_attempted += 1

            if UPLOAD_BLOCKED_PATTERN.search(message):
                self._state.uploads_blocked += 1

            if match := USER_IDENTIFIER_PATTERN.search(message):
                user_id = match.group("id")
                if user_id not in self._state.user_identifiers:
                    self._state.user_identifiers.append(user_id)

            if match := ENGINE_VERSION_PATTERN.search(raw_line):
                self._state.chromium_version = match.group("chromium")
                self._state.cef_version = match.group("cef")
                self._state.cefsharp_version = match.group("cefsharp")

        if entry.severity in ("error", "warning") and entry.timestamp:
            self._state.timeline_events.append(
                TimelineEvent(
                    timestamp=entry.timestamp,
                    source_file=file_type,
                    severity=Severity.MEDIUM if entry.severity == "warning" else Severity.HIGH,
                    event_type=entry.module or "unknown",
                    description=message[:200],
                )
            )

    def _record_window_change(
        self,
        from_title: str,
        from_handle: int,
        to_title: str,
        to_handle: int,
        timestamp: datetime | None,
    ) -> None:
        """Record a foreground window change."""
        _ = from_title, from_handle, to_handle
        if self._state.current_window_title is not None:
            self._state.foreground_spans.append(
                ForegroundWindowSpan(
                    window_title=self._state.current_window_title,
                    start_time=self._state.current_window_start,
                    end_time=timestamp,
                )
            )

        self._state.current_window_title = to_title
        self._state.current_window_start = timestamp

    def _finalize_foreground_spans(self, last_timestamp: datetime | None) -> None:
        """Finalize the last foreground window span."""
        if self._state.current_window_title is not None:
            self._state.foreground_spans.append(
                ForegroundWindowSpan(
                    window_title=self._state.current_window_title,
                    start_time=self._state.current_window_start,
                    end_time=last_timestamp,
                )
            )

    def _extract_environment_from_header(
        self, header_lines: list[str], grammar
    ) -> None:
        """Extract environment info from log file header."""
        header_data = grammar.parse_header(header_lines)

        if header_data and self._state.environment is None:
            self._state.environment = EnvironmentInfo(
                program_version=header_data.get("program_version"),
                program_build=header_data.get("program_build"),
                architecture=header_data.get("architecture"),
                os_name=header_data.get("os_name"),
                os_version=header_data.get("os_version"),
                machine_name=header_data.get("machine_name"),
                machine_model=header_data.get("machine_model"),
                machine_manufacturer=header_data.get("machine_manufacturer"),
                runtime_id=header_data.get("runtime_id"),
            )
