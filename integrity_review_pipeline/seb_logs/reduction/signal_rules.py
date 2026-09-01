"""Signal extraction rules for security-relevant log events.

This module contains patterns to identify security-relevant signals
from SEB/TSB log files, including policy violations, blocked attempts,
integrity anomalies, and configuration deviations.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime

from ..contracts import (
    EvidenceLine,
    ExtractedSignal,
    LogFileType,
    Severity,
    SignalCategory,
)
from ..grammars.base import ParsedLogEntry


@dataclass
class SignalMatch:
    """A signal extracted from a log entry."""

    signal_id: str
    category: SignalCategory
    severity: Severity
    confidence: str
    title: str
    description: str
    baseline_rule: str | None
    enforced_by_tsb: bool | None
    entry: ParsedLogEntry
    file_type: LogFileType


@dataclass
class SignalRule:
    """A rule for extracting signals from log entries."""

    title: str
    category: SignalCategory
    severity: Severity
    confidence: str
    file_types: set[LogFileType]
    patterns: list[re.Pattern[str]]
    description_template: str
    baseline_rule: str | None = None
    enforced_by_tsb: bool | None = None
    module_patterns: list[str] = field(default_factory=list)
    severity_patterns: list[str] = field(default_factory=list)

    def matches(self, entry: ParsedLogEntry, file_type: LogFileType) -> bool:
        """Check if an entry matches this signal rule."""
        if file_type not in self.file_types:
            return False

        if self.severity_patterns:
            if entry.severity and entry.severity.lower() in [
                s.lower() for s in self.severity_patterns
            ]:
                pass
            else:
                return False

        if self.module_patterns:
            if entry.module and any(
                mp.lower() in entry.module.lower() for mp in self.module_patterns
            ):
                pass
            else:
                return False

        for pattern in self.patterns:
            if pattern.search(entry.message) or pattern.search(entry.raw_line):
                return True

        return False


BLOCKED_KEYSTROKE_PATTERNS = [
    re.compile(r"Blocked '(?P<key>[^']+)'", re.IGNORECASE),
    re.compile(r"Blocked keystroke.*'(?P<key>[^']+)'", re.IGNORECASE),
]

BLOCKED_MOUSE_PATTERNS = [
    re.compile(r"Blocked mouse", re.IGNORECASE),
    re.compile(r"Mouse.*blocked", re.IGNORECASE),
]

INJECTED_KEYSTROKE_PATTERNS = [
    re.compile(r"Detected.*injected", re.IGNORECASE),
    re.compile(r"Keystroke was injected", re.IGNORECASE),
    re.compile(r"software-synthesised", re.IGNORECASE),
]

VM_DETECTED_PATTERNS = [
    re.compile(r"Virtual machine detected", re.IGNORECASE),
    re.compile(r"VM.*detected", re.IGNORECASE),
    re.compile(r"running.*virtual machine", re.IGNORECASE),
    re.compile(r"VMware|VirtualBox|Hyper-V|QEMU|Parallels", re.IGNORECASE),
]

REMOTE_SESSION_PATTERNS = [
    re.compile(r"Remote session detected", re.IGNORECASE),
    re.compile(r"Remote desktop.*detected", re.IGNORECASE),
    re.compile(r"RDP session", re.IGNORECASE),
]

INTEGRITY_COMPROMISE_PATTERNS = [
    re.compile(r"integrity.*compromised", re.IGNORECASE),
    re.compile(r"Application integrity is compromised", re.IGNORECASE),
    re.compile(r"Failed to verify application integrity", re.IGNORECASE),
]

INTEGRITY_MODULE_UNAVAILABLE_PATTERNS = [
    re.compile(r"Integrity module is not available", re.IGNORECASE),
]

BLACKLIST_TERMINATED_PATTERNS = [
    re.compile(r"Successfully terminated.*blacklist", re.IGNORECASE),
    re.compile(r"terminated.*blacklisted application", re.IGNORECASE),
]

BLACKLIST_TERMINATION_FAILED_PATTERNS = [
    re.compile(r"Failed to terminate.*blacklist", re.IGNORECASE),
    re.compile(r"Could not terminate.*blacklisted", re.IGNORECASE),
]

FOREGROUND_WINDOW_CHANGE_PATTERNS = [
    re.compile(r"Window has changed from '(?P<from>[^']+)'.*to '(?P<to>[^']+)'"),
    re.compile(r"Foreground window changed"),
]

BLOCKED_NAVIGATION_PATTERNS = [
    re.compile(r"Blocked navigation", re.IGNORECASE),
    re.compile(r"Navigation.*blocked", re.IGNORECASE),
    re.compile(r"Request was blocked by filter", re.IGNORECASE),
]

BLOCKED_DOWNLOAD_PATTERNS = [
    re.compile(r"Blocked download", re.IGNORECASE),
    re.compile(r"Download.*blocked", re.IGNORECASE),
]

BLOCKED_POPUP_PATTERNS = [
    re.compile(r"Blocked popup", re.IGNORECASE),
    re.compile(r"Popup.*blocked", re.IGNORECASE),
]

DEV_CONSOLE_OPENED_PATTERNS = [
    re.compile(r"Developer console.*opened", re.IGNORECASE),
    re.compile(r"DevTools.*opened", re.IGNORECASE),
    re.compile(r"Console.*opened", re.IGNORECASE),
]

QUIT_URL_PATTERNS = [
    re.compile(r"Quit URL.*triggered", re.IGNORECASE),
    re.compile(r"Detected quit URL", re.IGNORECASE),
]

DISPLAY_POLICY_VIOLATION_PATTERNS = [
    re.compile(
        r"Detected (?P<count>\d+) active displays?, (?P<allowed>\d+) (?:is|are) allowed"
    ),
    re.compile(r"external display.*only internal.*allowed", re.IGNORECASE),
]

SESSION_LOCK_PATTERNS = [
    re.compile(r"Session.*locked", re.IGNORECASE),
    re.compile(r"User session lock", re.IGNORECASE),
    re.compile(r"Workstation.*locked", re.IGNORECASE),
]

STICKY_KEYS_PATTERNS = [
    re.compile(r"Sticky keys.*changed", re.IGNORECASE),
    re.compile(r"StickyKeys", re.IGNORECASE),
]

CURSOR_TAMPERING_PATTERNS = [
    re.compile(r"Cursor.*tampering", re.IGNORECASE),
    re.compile(r"Cursor.*modified", re.IGNORECASE),
]

EASE_OF_ACCESS_PATTERNS = [
    re.compile(r"Ease of access.*changed", re.IGNORECASE),
    re.compile(r"EaseOfAccess.*tampering", re.IGNORECASE),
]

PROCESS_STARTED_PATTERNS = [
    re.compile(
        r"Process '(?P<name>[^']+)' \((?P<pid>\d+)\) has been started"
        r".*Original Name: '(?P<orig>[^']*)'.*Path: '(?P<path>[^']*)'.*Signature: (?P<sig>[^\]]+)"
    ),
]

EXPLORER_STARTED_PATTERNS = [
    re.compile(r"explorer\.exe.*has been started", re.IGNORECASE),
    re.compile(r"New Explorer instance", re.IGNORECASE),
]

OS_LOCKDOWN_DRIFT_PATTERNS = [
    re.compile(r"is enabled instead of disabled", re.IGNORECASE),
    re.compile(
        r"\[FeatureConfigurationMonitor\].*(?P<config>\w+).*enabled instead of disabled"
    ),
]

BEK_SALT_MISSING_PATTERNS = [
    re.compile(r"does not contain a salt value for the browser exam key"),
    re.compile(r"salt.*missing.*browser exam key", re.IGNORECASE),
]

BEK_FALLBACK_CALCULATION_PATTERNS = [
    re.compile(r"Failed to calculate browser exam key using integrity module"),
    re.compile(r"Falling back to simplified calculation"),
]

RECONFIGURATION_PATTERNS = [
    re.compile(r"Reconfiguration.*accepted", re.IGNORECASE),
    re.compile(r"Configuration.*changed.*during session", re.IGNORECASE),
]

WINDOW_GUARD_DEACTIVATED_PATTERNS = [
    re.compile(r"Window guard.*deactivated", re.IGNORECASE),
    re.compile(r"Deactivated window guard", re.IGNORECASE),
]

SESSION_BANNER_PATTERNS = [
    re.compile(r"### ----+ Session Start Procedure ----+ ###"),
    re.compile(r"### ----+ Session Stop Procedure ----+ ###"),
]

BROWSER_ERROR_PATTERNS = [
    re.compile(r"ERROR:.*browser", re.IGNORECASE),
    re.compile(r"Failed to load.*page", re.IGNORECASE),
    re.compile(r"Page load failed", re.IGNORECASE),
]

NETWORK_ERROR_PATTERNS = [
    re.compile(r"Network error", re.IGNORECASE),
    re.compile(r"Connection failed", re.IGNORECASE),
    re.compile(r"Unable to connect", re.IGNORECASE),
]

QUIT_PASSWORD_ATTEMPT_PATTERNS = [
    re.compile(r"Quit password.*incorrect", re.IGNORECASE),
    re.compile(r"Failed.*quit password", re.IGNORECASE),
    re.compile(r"Invalid quit password", re.IGNORECASE),
]


SIGNAL_RULES: list[SignalRule] = [
    SignalRule(
        title="Blocked keystroke",
        category=SignalCategory.BLOCKED_ATTEMPT,
        severity=Severity.LOW,
        confidence="high",
        file_types={LogFileType.CLIENT, LogFileType.RUNTIME},
        patterns=BLOCKED_KEYSTROKE_PATTERNS,
        description_template="Keystroke blocked by keyboard interceptor",
        enforced_by_tsb=True,
    ),
    SignalRule(
        title="Blocked mouse action",
        category=SignalCategory.BLOCKED_ATTEMPT,
        severity=Severity.LOW,
        confidence="high",
        file_types={LogFileType.CLIENT},
        patterns=BLOCKED_MOUSE_PATTERNS,
        description_template="Mouse action blocked by interceptor",
        enforced_by_tsb=True,
    ),
    SignalRule(
        title="Injected keystroke detected",
        category=SignalCategory.INTEGRITY_ANOMALY,
        severity=Severity.CRITICAL,
        confidence="high",
        file_types={LogFileType.CLIENT, LogFileType.RUNTIME},
        patterns=INJECTED_KEYSTROKE_PATTERNS,
        description_template="Software-synthesised/injected keystroke detected",
        baseline_rule="Injected keystrokes indicate automation or remote control",
    ),
    SignalRule(
        title="Virtual machine detected",
        category=SignalCategory.ENVIRONMENT_ISSUE,
        severity=Severity.HIGH,
        confidence="high",
        file_types={LogFileType.RUNTIME, LogFileType.CLIENT},
        patterns=VM_DETECTED_PATTERNS,
        description_template="Running inside a virtual machine",
        baseline_rule="VM policy determines if this is a violation",
    ),
    SignalRule(
        title="Remote session detected",
        category=SignalCategory.INTEGRITY_ANOMALY,
        severity=Severity.CRITICAL,
        confidence="high",
        file_types={LogFileType.RUNTIME, LogFileType.CLIENT},
        patterns=REMOTE_SESSION_PATTERNS,
        description_template="Remote desktop/session detected",
        baseline_rule="Remote sessions allow external control",
    ),
    SignalRule(
        title="Application integrity compromised",
        category=SignalCategory.INTEGRITY_ANOMALY,
        severity=Severity.HIGH,
        confidence="high",
        file_types={LogFileType.RUNTIME, LogFileType.CLIENT},
        patterns=INTEGRITY_COMPROMISE_PATTERNS,
        description_template="Application integrity verification failed",
        baseline_rule="Depends on signing status baseline",
    ),
    SignalRule(
        title="Integrity module unavailable",
        category=SignalCategory.COVERAGE_GAP,
        severity=Severity.MEDIUM,
        confidence="high",
        file_types={LogFileType.RUNTIME, LogFileType.CLIENT},
        patterns=INTEGRITY_MODULE_UNAVAILABLE_PATTERNS,
        description_template="Integrity verification module not available",
        baseline_rule="No tamper detection for this session",
    ),
    SignalRule(
        title="Blacklisted application terminated",
        category=SignalCategory.BLOCKED_ATTEMPT,
        severity=Severity.MEDIUM,
        confidence="high",
        file_types={LogFileType.CLIENT},
        patterns=BLACKLIST_TERMINATED_PATTERNS,
        description_template="Blacklisted application was terminated",
        enforced_by_tsb=True,
    ),
    SignalRule(
        title="Blacklist termination failed",
        category=SignalCategory.POLICY_VIOLATION,
        severity=Severity.HIGH,
        confidence="high",
        file_types={LogFileType.CLIENT},
        patterns=BLACKLIST_TERMINATION_FAILED_PATTERNS,
        description_template="Failed to terminate blacklisted application",
        enforced_by_tsb=False,
    ),
    SignalRule(
        title="Foreground window change",
        category=SignalCategory.BLOCKED_ATTEMPT,
        severity=Severity.INFO,
        confidence="high",
        file_types={LogFileType.CLIENT, LogFileType.RUNTIME},
        patterns=FOREGROUND_WINDOW_CHANGE_PATTERNS,
        description_template="Window focus changed",
    ),
    SignalRule(
        title="Blocked navigation",
        category=SignalCategory.BLOCKED_ATTEMPT,
        severity=Severity.LOW,
        confidence="high",
        file_types={LogFileType.BROWSER, LogFileType.CLIENT},
        patterns=BLOCKED_NAVIGATION_PATTERNS,
        description_template="Navigation was blocked by URL filter",
        enforced_by_tsb=True,
    ),
    SignalRule(
        title="Blocked download",
        category=SignalCategory.BLOCKED_ATTEMPT,
        severity=Severity.LOW,
        confidence="high",
        file_types={LogFileType.BROWSER, LogFileType.CLIENT},
        patterns=BLOCKED_DOWNLOAD_PATTERNS,
        description_template="Download was blocked",
        enforced_by_tsb=True,
    ),
    SignalRule(
        title="Blocked popup",
        category=SignalCategory.BLOCKED_ATTEMPT,
        severity=Severity.LOW,
        confidence="high",
        file_types={LogFileType.BROWSER},
        patterns=BLOCKED_POPUP_PATTERNS,
        description_template="Popup window was blocked",
        enforced_by_tsb=True,
    ),
    SignalRule(
        title="Developer console opened",
        category=SignalCategory.POLICY_VIOLATION,
        severity=Severity.HIGH,
        confidence="high",
        file_types={LogFileType.BROWSER, LogFileType.CLIENT},
        patterns=DEV_CONSOLE_OPENED_PATTERNS,
        description_template="Browser developer console was opened",
        baseline_rule="dev_console should be disabled in exam config",
    ),
    SignalRule(
        title="Quit URL triggered",
        category=SignalCategory.BLOCKED_ATTEMPT,
        severity=Severity.INFO,
        confidence="high",
        file_types={LogFileType.CLIENT, LogFileType.RUNTIME},
        patterns=QUIT_URL_PATTERNS,
        description_template="Quit URL was triggered",
    ),
    SignalRule(
        title="Display policy violation",
        category=SignalCategory.POLICY_VIOLATION,
        severity=Severity.HIGH,
        confidence="high",
        file_types={LogFileType.RUNTIME, LogFileType.CLIENT},
        patterns=DISPLAY_POLICY_VIOLATION_PATTERNS,
        description_template="More displays active than allowed",
        baseline_rule="Display count should match configured allowance",
    ),
    SignalRule(
        title="Session locked/switched",
        category=SignalCategory.POLICY_VIOLATION,
        severity=Severity.MEDIUM,
        confidence="high",
        file_types={LogFileType.CLIENT, LogFileType.RUNTIME, LogFileType.SERVICE},
        patterns=SESSION_LOCK_PATTERNS,
        description_template="User session was locked or switched",
    ),
    SignalRule(
        title="Sticky keys change",
        category=SignalCategory.INTEGRITY_ANOMALY,
        severity=Severity.MEDIUM,
        confidence="high",
        file_types={LogFileType.CLIENT, LogFileType.SERVICE},
        patterns=STICKY_KEYS_PATTERNS,
        description_template="Sticky keys setting changed",
    ),
    SignalRule(
        title="Cursor tampering",
        category=SignalCategory.INTEGRITY_ANOMALY,
        severity=Severity.HIGH,
        confidence="high",
        file_types={LogFileType.CLIENT, LogFileType.SERVICE},
        patterns=CURSOR_TAMPERING_PATTERNS,
        description_template="Cursor configuration was modified",
    ),
    SignalRule(
        title="Ease of access tampering",
        category=SignalCategory.INTEGRITY_ANOMALY,
        severity=Severity.HIGH,
        confidence="high",
        file_types={LogFileType.CLIENT, LogFileType.SERVICE},
        patterns=EASE_OF_ACCESS_PATTERNS,
        description_template="Ease of access settings were modified",
    ),
    SignalRule(
        title="Process started",
        category=SignalCategory.BLOCKED_ATTEMPT,
        severity=Severity.INFO,
        confidence="high",
        file_types={LogFileType.CLIENT},
        patterns=PROCESS_STARTED_PATTERNS,
        description_template="New process started during session",
    ),
    SignalRule(
        title="New Explorer instance",
        category=SignalCategory.POLICY_VIOLATION,
        severity=Severity.MEDIUM,
        confidence="high",
        file_types={LogFileType.CLIENT},
        patterns=EXPLORER_STARTED_PATTERNS,
        description_template="New Windows Explorer instance started",
    ),
    SignalRule(
        title="OS lockdown drift",
        category=SignalCategory.CONFIG_DEVIATION,
        severity=Severity.MEDIUM,
        confidence="high",
        file_types={LogFileType.SERVICE},
        patterns=OS_LOCKDOWN_DRIFT_PATTERNS,
        module_patterns=["FeatureConfigurationMonitor"],
        description_template="OS lockdown reverted - feature re-enabled during session",
        baseline_rule="OS lockdowns should remain disabled throughout session",
    ),
    SignalRule(
        title="BEK salt missing",
        category=SignalCategory.CONFIG_DEVIATION,
        severity=Severity.MEDIUM,
        confidence="high",
        file_types={LogFileType.CLIENT, LogFileType.RUNTIME},
        patterns=BEK_SALT_MISSING_PATTERNS,
        module_patterns=["KeyGenerator"],
        description_template="Browser exam key salt not configured",
    ),
    SignalRule(
        title="BEK calculation fallback",
        category=SignalCategory.COVERAGE_GAP,
        severity=Severity.LOW,
        confidence="high",
        file_types={LogFileType.CLIENT, LogFileType.RUNTIME},
        patterns=BEK_FALLBACK_CALCULATION_PATTERNS,
        module_patterns=["KeyGenerator"],
        description_template="Browser exam key using simplified calculation (integrity module failed)",
    ),
    SignalRule(
        title="Reconfiguration during session",
        category=SignalCategory.POLICY_VIOLATION,
        severity=Severity.HIGH,
        confidence="high",
        file_types={LogFileType.RUNTIME, LogFileType.CLIENT},
        patterns=RECONFIGURATION_PATTERNS,
        description_template="Configuration was changed during active session",
        baseline_rule="Reconfiguration should be disabled",
    ),
    SignalRule(
        title="Window guard deactivated",
        category=SignalCategory.COVERAGE_GAP,
        severity=Severity.MEDIUM,
        confidence="high",
        file_types={LogFileType.CLIENT},
        patterns=WINDOW_GUARD_DEACTIVATED_PATTERNS,
        description_template="Window guard protection deactivated",
    ),
    SignalRule(
        title="Quit password attempt failed",
        category=SignalCategory.BLOCKED_ATTEMPT,
        severity=Severity.LOW,
        confidence="high",
        file_types={LogFileType.CLIENT, LogFileType.RUNTIME},
        patterns=QUIT_PASSWORD_ATTEMPT_PATTERNS,
        description_template="Incorrect quit password entered",
        enforced_by_tsb=True,
    ),
]


@dataclass
class SignalAccumulator:
    """Accumulates multiple occurrences of the same signal type."""

    signal_id: str
    title: str
    category: SignalCategory
    severity: Severity
    confidence: str
    description: str
    baseline_rule: str | None
    enforced_by_tsb: bool | None
    count: int = 0
    first_seen: datetime | None = None
    last_seen: datetime | None = None
    evidence: list[EvidenceLine] = field(default_factory=list)
    max_evidence: int = 5

    def add(self, entry: ParsedLogEntry, file_type: LogFileType) -> None:
        """Add an occurrence to this accumulator."""
        self.count += 1
        if self.first_seen is None or (
            entry.timestamp and entry.timestamp < self.first_seen
        ):
            self.first_seen = entry.timestamp
        if self.last_seen is None or (
            entry.timestamp and entry.timestamp > self.last_seen
        ):
            self.last_seen = entry.timestamp

        if len(self.evidence) < self.max_evidence:
            self.evidence.append(
                EvidenceLine(
                    file_type=file_type,
                    line_number=entry.line_number,
                    timestamp=entry.timestamp,
                    text=entry.raw_line[:300],
                )
            )

    def to_signal(self) -> ExtractedSignal:
        """Convert to ExtractedSignal contract."""
        return ExtractedSignal(
            signal_id=self.signal_id,
            category=self.category,
            severity=self.severity,
            confidence=self.confidence,
            title=self.title,
            description=self.description,
            baseline_rule=self.baseline_rule,
            enforced_by_tsb=self.enforced_by_tsb,
            occurrence_count=self.count,
            first_seen=self.first_seen,
            last_seen=self.last_seen,
            evidence=self.evidence,
        )


class SignalDetector:
    """Detects and accumulates security signals from log entries."""

    def __init__(self) -> None:
        self._accumulators: dict[str, SignalAccumulator] = {}

    def check_entry(
        self, entry: ParsedLogEntry, file_type: LogFileType
    ) -> SignalMatch | None:
        """Check if an entry contains a signal.

        Returns SignalMatch if the entry matches a signal rule, None otherwise.
        """
        for rule in SIGNAL_RULES:
            if rule.matches(entry, file_type):
                match = SignalMatch(
                    signal_id=self._get_or_create_id(rule.title),
                    category=rule.category,
                    severity=rule.severity,
                    confidence=rule.confidence,
                    title=rule.title,
                    description=rule.description_template,
                    baseline_rule=rule.baseline_rule,
                    enforced_by_tsb=rule.enforced_by_tsb,
                    entry=entry,
                    file_type=file_type,
                )
                self._accumulate(match, rule)
                return match

        return None

    def _get_or_create_id(self, title: str) -> str:
        """Get or create a signal ID for a title."""
        key = title.lower().replace(" ", "_")
        if key not in self._accumulators:
            return str(uuid.uuid4())[:8]
        return self._accumulators[key].signal_id

    def _accumulate(self, match: SignalMatch, rule: SignalRule) -> None:
        """Add a match to the appropriate accumulator."""
        key = rule.title.lower().replace(" ", "_")
        if key not in self._accumulators:
            self._accumulators[key] = SignalAccumulator(
                signal_id=match.signal_id,
                title=match.title,
                category=match.category,
                severity=match.severity,
                confidence=match.confidence,
                description=match.description,
                baseline_rule=match.baseline_rule,
                enforced_by_tsb=match.enforced_by_tsb,
            )
        self._accumulators[key].add(match.entry, match.file_type)

    def get_signals(self) -> list[ExtractedSignal]:
        """Get all accumulated signals."""
        return [acc.to_signal() for acc in self._accumulators.values() if acc.count > 0]

    def reset(self) -> None:
        """Reset all accumulators for a new session."""
        self._accumulators.clear()
