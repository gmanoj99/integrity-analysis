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
    ConfigDeterminedBy,
    DisplayInfo,
    EffectiveConfigField,
    EvidenceLine,
    EnvironmentInfo,
    ForegroundWindowSpan,
    LogFileType,
    MonitoringCoverage,
    Platform,
    ProcessRecord,
    RawLogFileMeta,
    ReducedSessionEvidence,
    SebLogSessionRef,
    Severity,
    TimelineEvent,
)
from ..deps import SebLogPipelineDeps
from ..discovery import EXPECTED_LOG_FILES_BY_PLATFORM, get_log_file_key
from ..grammars.base import ParsedLogEntry
from ..grammars.cef_browser import CefBrowserGrammar
from ..grammars.macos import MacOSLogGrammar
from ..grammars.windows_dotnet import WindowsDotNetGrammar
from .bulk_event_aggregator import BulkEventAggregator, is_blacklist_detection
from .correlator import CoverageGapDetector, PhaseTracker, SignalCorrelator
from .noise_rules import NoiseDetector
from .redaction import redact_navigation_url, redact_text, redact_window_title
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

PAGE_ERROR_PATTERN = re.compile(
    r"(?:Page error|JavaScript error|Script error|Error loading|Failed to load).*?(?P<error>[^\n]{1,200})",
    re.IGNORECASE,
)

VM_INDICATOR_PATTERNS = [
    re.compile(r"(?:Virtual machine|VM).*detected.*(?P<vm_type>VMware|VirtualBox|Hyper-V|QEMU|Parallels)", re.IGNORECASE),
    re.compile(r"Running (?:on|in) (?P<vm_type>VMware|VirtualBox|Hyper-V|QEMU|Parallels)", re.IGNORECASE),
    re.compile(r"(?P<vm_type>VMware|VirtualBox|Hyper-V|QEMU|Parallels).*(?:environment|detected)", re.IGNORECASE),
    re.compile(
        r"\[VirtualMachineDetector\].*appears to be a virtual machine"
        r"(?:.*(?P<vm_type>VMware|VirtualBox|Hyper-V|QEMU|Parallels))?",
        re.IGNORECASE,
    ),
]

REMOTE_SESSION_INDICATOR_PATTERNS = [
    re.compile(r"Remote session detected", re.IGNORECASE),
    re.compile(r"(?:RDP|Remote Desktop|Terminal Services).*session", re.IGNORECASE),
]

DISPLAY_DETAIL_PATTERN = re.compile(
    r"Display.*?'(?P<id>[^']+)'.*?(?P<internal>internal|external)?.*?(?P<active>active|inactive)?",
    re.IGNORECASE,
)

CONFIG_FIELD_PATTERN = re.compile(
    r"(?:Setting|Config|Configuration).*?'(?P<field>[^']+)'"
    r"\s*(?:=>|=|:)\s*'(?P<value>[^']*)'",
    re.IGNORECASE,
)

LOG_LEVEL_PATTERN = re.compile(
    r"Log level.*?(?:set to|is|:)\s*(?P<level>DEBUG|INFO|WARNING|ERROR|VERBOSE)",
    re.IGNORECASE,
)

OS_LOCKDOWN_APPLIED_PATTERN = re.compile(
    r"(?:Applied|Enabled|Activated).*?(?:lockdown|lock down|restriction)",
    re.IGNORECASE,
)

WINDOW_HIDDEN_PATTERN = re.compile(
    r"(?:Hiding|Hidden|Covered|Masked).*window.*'(?P<title>[^']+)'",
    re.IGNORECASE,
)

WINDOW_ALLOWED_PATTERN = re.compile(
    r"(?:Allowing|Permitted|Approved).*window.*'(?P<title>[^']+)'",
    re.IGNORECASE,
)

EXPECTED_MONITORS_WINDOWS = [
    "keyboard",
    "mouse",
    "clipboard",
    "process",
    "window",
    "display",
    "network",
    "registry",
]

EXPECTED_MONITORS_MACOS = [
    "process",
    "window",
    "display",
    "screencapture",
    "kiosk",
]

EXPECTED_MONITORS_BY_PLATFORM = {
    Platform.WINDOWS: EXPECTED_MONITORS_WINDOWS,
    Platform.MACOS: EXPECTED_MONITORS_MACOS,
}

# SEB names some monitors after the thing being watched rather than the
# capability, so a substring test on the capability name alone would miss them.
MONITOR_ALIASES = {
    "process": ("process", "application"),
    "window": ("window", "desktop"),
    "screencapture": ("screencapture", "screen capture", "capturing"),
}


def _monitor_is_armed(monitor: str, monitors_armed: list[str]) -> bool:
    """Whether an expected monitor is covered by any armed-monitor entry.

    SEB logs "Started monitoring <free text>.", so the armed entries are
    phrases like "the network adapter" or "value 'Arrow' from registry key
    '...'" -- never bare capability names. Comparing for equality reports
    armed monitors as unarmed, which surfaces as fabricated coverage gaps.
    """
    tokens = MONITOR_ALIASES.get(monitor, (monitor,))
    return any(token in armed.lower() for armed in monitors_armed for token in tokens)


@dataclass
class SessionReducerState:
    """Mutable state accumulated during session reduction."""

    processes: dict[int, ProcessRecord] = field(default_factory=dict)
    foreground_spans: list[ForegroundWindowSpan] = field(default_factory=list)
    current_window_title: str | None = None
    current_window_raw_title: str | None = None
    current_window_handle: int | None = None
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
    vm_indicators: list[str] = field(default_factory=list)
    remote_session_detected: bool = False
    log_level: str | None = None
    os_lockdowns_applied: bool | None = None
    allowed_windows: set[str] = field(default_factory=set)
    hidden_windows: set[str] = field(default_factory=set)


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
        self._macos_grammar = MacOSLogGrammar()
        self._current_platform: Platform = Platform.WINDOWS

    def reduce_session(
        self,
        session_ref: SebLogSessionRef,
    ) -> ReducedSessionEvidence:
        """Reduce a session's logs to structured evidence."""
        self._reset()
        self._current_platform = session_ref.platform

        if session_ref.manifest and session_ref.manifest.start_time:
            self._browser_grammar.set_reference_year(
                session_ref.manifest.start_time.year
            )

        analysed_files: list[RawLogFileMeta] = []
        total_raw_bytes = 0
        last_timestamp: datetime | None = None

        # Determine which file types to process based on platform
        expected_files = EXPECTED_LOG_FILES_BY_PLATFORM.get(
            session_ref.platform, EXPECTED_LOG_FILES_BY_PLATFORM[Platform.WINDOWS]
        )

        # Process files that are present
        for file_type in session_ref.files_present:
            if file_type == LogFileType.MANIFEST:
                continue  # Skip manifest, not a log file
            meta, file_last_ts = self._process_log_file(session_ref, file_type)
            analysed_files.append(meta)
            total_raw_bytes += meta.size_bytes
            if file_last_ts and (last_timestamp is None or file_last_ts > last_timestamp):
                last_timestamp = file_last_ts

        # Mark expected files that are missing
        for file_type in expected_files:
            if file_type not in session_ref.files_present:
                analysed_files.append(
                    RawLogFileMeta(
                        file_type=file_type,
                        s3_key="",
                        missing=True,
                    )
                )

        phases = self._phase_tracker.finalize(last_timestamp)

        self._finalize_foreground_spans(last_timestamp)

        if self._state.environment:
            self._state.environment.virtual_machine_indicators = self._state.vm_indicators
            if self._state.remote_session_detected:
                self._state.environment.remote_session = True

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

        # Use platform-specific expected monitors list
        expected_monitors = EXPECTED_MONITORS_BY_PLATFORM.get(
            session_ref.platform, EXPECTED_MONITORS_WINDOWS
        )
        monitors_not_armed = [
            m for m in expected_monitors
            if not _monitor_is_armed(m.lower(), self._state.monitors_armed)
        ]

        monitoring = MonitoringCoverage(
            monitors_armed=self._state.monitors_armed,
            monitors_not_armed=monitors_not_armed,
            log_level=self._state.log_level,
            os_lockdowns_applied=self._state.os_lockdowns_applied,
            coverage_gaps=coverage_gaps,
        )

        # Service log caveat is Windows-only
        if session_ref.platform == Platform.WINDOWS:
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
            "platform": session_ref.platform.value,
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

        # Extract environment from header for Windows (Runtime/Client) or macOS unified log
        if header_lines and file_type in (
            LogFileType.RUNTIME, LogFileType.CLIENT, LogFileType.MACOS_UNIFIED
        ):
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
        if file_type == LogFileType.MACOS_UNIFIED:
            return self._macos_grammar
        if file_type == LogFileType.BROWSER:
            return self._browser_grammar
        return self._windows_grammar

    def _process_entry(
        self, entry: ParsedLogEntry, file_type: LogFileType
    ) -> None:
        """Process a single parsed log entry through all detectors."""
        if self._bulk_aggregator.check_entry(entry, file_type):
            if is_blacklist_detection(entry):
                self._signal_detector.check_entry(entry, file_type)
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
            sig_str = match.group("sig")
            signed = None
            signature_thumbprint = None
            if sig_str:
                sig_lower = sig_str.lower().strip()
                if "not signed" in sig_lower or "unsigned" in sig_lower:
                    signed = False
                elif "signed" in sig_lower:
                    signed = True
                    tp_match = re.search(r"[0-9A-Fa-f]{40}", sig_str)
                    if tp_match:
                        signature_thumbprint = tp_match.group(0)

            if pid not in self._state.processes:
                self._state.processes[pid] = ProcessRecord(
                    name=match.group("name"),
                    pid=pid,
                    path=match.group("path"),
                    original_name=match.group("orig"),
                    signed=signed,
                    signature_thumbprint=signature_thumbprint,
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

        for pattern in VM_INDICATOR_PATTERNS:
            if match := pattern.search(raw_line):
                vm_type = match.groupdict().get("vm_type") or "detected"
                if vm_type not in self._state.vm_indicators:
                    self._state.vm_indicators.append(vm_type)

        for pattern in REMOTE_SESSION_INDICATOR_PATTERNS:
            if pattern.search(raw_line):
                self._state.remote_session_detected = True
                break

        if match := DISPLAY_DETAIL_PATTERN.search(raw_line):
            display_id = match.group("id")
            if not any(d.identifier == display_id for d in self._state.displays):
                internal_str = match.group("internal")
                active_str = match.group("active")
                self._state.displays.append(
                    DisplayInfo(
                        identifier=display_id,
                        active=active_str is None or active_str.lower() == "active",
                        internal=internal_str.lower() == "internal" if internal_str else None,
                    )
                )

        if match := CONFIG_FIELD_PATTERN.search(raw_line):
            field_name = match.group("field")
            value = match.group("value")
            if not any(c.field_name == field_name for c in self._state.effective_config):
                self._state.effective_config.append(
                    EffectiveConfigField(
                        field_name=field_name,
                        value=value,
                        determined_by=ConfigDeterminedBy.LOG_EVIDENCE,
                        evidence=[
                            EvidenceLine(
                                file_type=file_type,
                                line_number=entry.line_number,
                                timestamp=entry.timestamp,
                                text=redact_text(raw_line[:300]),
                            )
                        ],
                    )
                )

        if self._state.log_level is None:
            if match := LOG_LEVEL_PATTERN.search(raw_line):
                self._state.log_level = match.group("level").upper()

        if self._state.os_lockdowns_applied is None:
            if OS_LOCKDOWN_APPLIED_PATTERN.search(raw_line):
                self._state.os_lockdowns_applied = True

        if match := WINDOW_HIDDEN_PATTERN.search(raw_line):
            self._state.hidden_windows.add(match.group("title"))

        if match := WINDOW_ALLOWED_PATTERN.search(raw_line):
            self._state.allowed_windows.add(match.group("title"))

        if file_type == LogFileType.BROWSER or file_type == LogFileType.CLIENT:
            if BROWSER_WINDOW_CREATED_PATTERN.search(message):
                self._state.browser_windows_created += 1

            if match := NAVIGATED_TO_PATTERN.search(message):
                url = match.group("url")
                if len(self._state.navigations) < 50:
                    self._state.navigations.append(redact_navigation_url(url))

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

            if match := PAGE_ERROR_PATTERN.search(raw_line):
                error_msg = match.group("error").strip()
                if error_msg and len(self._state.page_errors) < 50:
                    self._state.page_errors.append(redact_text(error_msg[:200]))

        if entry.severity in ("error", "warning") and entry.timestamp:
            self._state.timeline_events.append(
                TimelineEvent(
                    timestamp=entry.timestamp,
                    source_file=file_type,
                    severity=Severity.MEDIUM if entry.severity == "warning" else Severity.HIGH,
                    event_type=entry.module or "unknown",
                    description=redact_text(message[:200]),
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
        _ = from_handle
        if self._state.current_window_title is not None:
            raw_title = self._state.current_window_raw_title or self._state.current_window_title
            allowed = self._window_allowed(raw_title)
            action = "none"
            if raw_title in self._state.hidden_windows:
                action = "hidden"
            elif allowed is False:
                action = "blocked"

            self._state.foreground_spans.append(
                ForegroundWindowSpan(
                    window_title=self._state.current_window_title,
                    window_handle=self._state.current_window_handle,
                    start_time=self._state.current_window_start,
                    end_time=timestamp,
                    allowed=allowed,
                    action_taken=action,
                )
            )

        self._state.current_window_title = redact_window_title(to_title)
        self._state.current_window_raw_title = to_title
        self._state.current_window_handle = to_handle
        self._state.current_window_start = timestamp

    def _window_allowed(self, raw_title: str | None) -> bool | None:
        """Whether SEB allowed this window, or None when it never decided.

        `allowed_windows` / `hidden_windows` only fill in when SEB logs an
        explicit decision about a title. Treating "never mentioned" as "not
        allowed" marks every window as blocked -- including the exam window
        itself -- which reads downstream as a policy violation.
        """
        if not raw_title:
            return None
        if raw_title in self._state.allowed_windows:
            return True
        if raw_title in self._state.hidden_windows:
            return False
        return None

    def _finalize_foreground_spans(self, last_timestamp: datetime | None) -> None:
        """Finalize the last foreground window span."""
        if self._state.current_window_title is not None:
            raw_title = self._state.current_window_raw_title or self._state.current_window_title
            allowed = self._window_allowed(raw_title)
            action = "none"
            if raw_title in self._state.hidden_windows:
                action = "hidden"
            elif allowed is False:
                action = "blocked"

            self._state.foreground_spans.append(
                ForegroundWindowSpan(
                    window_title=self._state.current_window_title,
                    window_handle=self._state.current_window_handle,
                    start_time=self._state.current_window_start,
                    end_time=last_timestamp,
                    allowed=allowed,
                    action_taken=action,
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
                virtual_machine_indicators=self._state.vm_indicators,
                remote_session=self._state.remote_session_detected if self._state.remote_session_detected else None,
            )
        elif self._state.environment is not None:
            self._state.environment.virtual_machine_indicators = self._state.vm_indicators
            if self._state.remote_session_detected:
                self._state.environment.remote_session = True
