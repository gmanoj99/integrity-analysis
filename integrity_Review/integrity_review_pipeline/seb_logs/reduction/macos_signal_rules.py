"""macOS-specific signal extraction rules for SEB log analysis.

These rules extract security-relevant signals from macOS SEB unified log files.
They are scoped to LogFileType.MACOS_UNIFIED to avoid affecting Windows behaviour.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..contracts import Severity, SignalCategory


@dataclass
class MacOSSignalRule:
    """A rule for extracting signals from macOS log entries."""

    title: str
    category: SignalCategory
    severity: Severity
    confidence: str
    patterns: list[re.Pattern[str]]
    description_template: str
    baseline_rule: str | None = None
    enforced_by_tsb: bool | None = None

    def matches(self, message: str, raw_line: str) -> dict[str, str] | None:
        """Check if a message matches this signal rule.

        Returns captured groups dict if matched, None otherwise.
        """
        for pattern in self.patterns:
            match = pattern.search(message) or pattern.search(raw_line)
            if match:
                return match.groupdict()
        return None


# Prohibited app/process detection patterns
NOT_ALLOWED_APP_PATTERNS = [
    re.compile(
        r"This not allowed application is running and has to be quit first",
        re.IGNORECASE,
    ),
]

NOT_ALLOWED_PROCESS_PATTERNS = [
    re.compile(
        r"This not allowed process is running and has to terminated.*?executable\s*=\s*\"?(?P<executable>[^\";\n]+)",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"This not allowed process is running",
        re.IGNORECASE,
    ),
]

# Termination patterns
SUCCESSFULLY_TERMINATED_PATTERNS = [
    re.compile(
        r"Successfully terminated application/process:.*?name\s*=\s*\"?(?P<name>[^\";\n}]+)",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"Running process (?P<name>\w+) successfully force terminated",
        re.IGNORECASE,
    ),
]

SESSION_TERMINATED_PROCESSES_PATTERNS = [
    re.compile(
        r"These processes were terminated by SEB during this session",
        re.IGNORECASE,
    ),
]

# Remote management
REMOTE_MANAGEMENT_PATTERNS = [
    re.compile(r"RemoteManagementAgent", re.IGNORECASE),
]

# Display policy
DISPLAY_SETTINGS_PATTERNS = [
    re.compile(
        r"Current Settings: Maximum allowed displays:\s*(?P<count>\d+)",
        re.IGNORECASE,
    ),
]

NON_BUILTIN_DISPLAY_PATTERNS = [
    re.compile(
        r"Found matching non built-in screen.*for main display",
        re.IGNORECASE,
    ),
    re.compile(
        r"is not built-in\s+is main",
        re.IGNORECASE,
    ),
]

BUILTIN_REQUIRED_UNAVAILABLE_PATTERNS = [
    re.compile(
        r"noRequiredBuiltInScreenAvailable.*1",
        re.IGNORECASE,
    ),
]

# Exam lock / interruption
LOCK_SEB_PATTERNS = [
    re.compile(r"lockSEB:\s*(?P<reason>\w+)", re.IGNORECASE),
]

REOPENING_LOCKED_EXAM_PATTERNS = [
    re.compile(
        r"Re-opening an exam which was locked before",
        re.IGNORECASE,
    ),
    re.compile(
        r"Attempting to start an exam which is on the list of previously interrupted",
        re.IGNORECASE,
    ),
]

SEB_LOCKED_DURATION_PATTERNS = [
    re.compile(
        r"SEB was locked \(exam interrupted\) for (?P<duration>[\d:]+)",
        re.IGNORECASE,
    ),
]

# Password events
LOCK_PASSWORD_ENTERED_PATTERNS = [
    re.compile(r"Password entered in lock view alert", re.IGNORECASE),
]

CORRECT_PASSWORD_PATTERNS = [
    re.compile(r"Correct password entered", re.IGNORECASE),
]

QUIT_PASSWORD_PATTERNS = [
    re.compile(r"Displaying quit password alert", re.IGNORECASE),
]

CORRECT_QUIT_PASSWORD_PATTERNS = [
    re.compile(r"Correct quit password entered", re.IGNORECASE),
]

# Screen capture
SCREEN_CAPTURE_ACCESS_PATTERNS = [
    re.compile(
        r"User has to grant screen capture access",
        re.IGNORECASE,
    ),
]

SCREEN_CAPTURE_LEFTOVER_PATTERNS = [
    re.compile(
        r"There was a persistently saved redirected screencapture location.*Looks like SEB didn't quit properly",
        re.IGNORECASE,
    ),
]

SCREEN_CAPTURE_RESTORED_PATTERNS = [
    re.compile(r"Success of restoring SC:\s*(?P<success>\d)", re.IGNORECASE),
]

# VM detection (negative evidence)
NO_VM_PATTERNS = [
    re.compile(
        r"SEB is running on a native system \(no VM\)",
        re.IGNORECASE,
    ),
]

# Kiosk mode
KIOSK_MODE_PATTERNS = [
    re.compile(r"using SEB kiosk mode", re.IGNORECASE),
    re.compile(r"startKioskMode switchToApplications\s*(?P<value>\d)", re.IGNORECASE),
]

AAC_ENABLED_PATTERNS = [
    re.compile(r"isAACEnabled\s*=\s*(?P<value>\d)", re.IGNORECASE),
]

# Browser window close attempt
BROWSER_CLOSE_BLOCKED_PATTERNS = [
    re.compile(
        r"Web application requests the main browser window to be closed.*will be ignored",
        re.IGNORECASE,
    ),
]

# Configuration
CONFIG_LOADED_PATTERNS = [
    re.compile(
        r"openURLs event: Loading \.seb settings file with URL\s*(?P<url>\S+)",
        re.IGNORECASE,
    ),
]

SETTINGS_CHANGED_PATTERNS = [
    re.compile(r"Settings changed\.", re.IGNORECASE),
]

CONFIG_KEY_UPDATED_PATTERNS = [
    re.compile(r"Updated ConfigKey\.", re.IGNORECASE),
]

# Monitors started
PROCESS_WATCHER_PATTERNS = [
    re.compile(r"\[SEBController startProcessWatcher\]", re.IGNORECASE),
]

WINDOW_WATCHER_PATTERNS = [
    re.compile(r"\[SEBController startWindowWatcher\]", re.IGNORECASE),
]


MACOS_SIGNAL_RULES: list[MacOSSignalRule] = [
    MacOSSignalRule(
        title="Prohibited application detected (macOS)",
        category=SignalCategory.POLICY_VIOLATION,
        severity=Severity.MEDIUM,
        confidence="high",
        patterns=NOT_ALLOWED_APP_PATTERNS,
        description_template="Not allowed application was running and had to be quit",
    ),
    MacOSSignalRule(
        title="Prohibited process detected (macOS)",
        category=SignalCategory.POLICY_VIOLATION,
        severity=Severity.MEDIUM,
        confidence="high",
        patterns=NOT_ALLOWED_PROCESS_PATTERNS,
        description_template="Not allowed process was running and had to be terminated",
    ),
    MacOSSignalRule(
        title="Process terminated by SEB (macOS)",
        category=SignalCategory.BLOCKED_ATTEMPT,
        severity=Severity.LOW,
        confidence="high",
        patterns=SUCCESSFULLY_TERMINATED_PATTERNS,
        description_template="SEB successfully terminated a prohibited process",
        enforced_by_tsb=True,
    ),
    MacOSSignalRule(
        title="Session terminated processes summary (macOS)",
        category=SignalCategory.BLOCKED_ATTEMPT,
        severity=Severity.INFO,
        confidence="high",
        patterns=SESSION_TERMINATED_PROCESSES_PATTERNS,
        description_template="Summary of processes terminated during the session",
        enforced_by_tsb=True,
    ),
    MacOSSignalRule(
        title="Remote management agent detected (macOS)",
        category=SignalCategory.INTEGRITY_ANOMALY,
        severity=Severity.HIGH,
        confidence="high",
        patterns=REMOTE_MANAGEMENT_PATTERNS,
        description_template="RemoteManagementAgent was detected and terminated",
        baseline_rule="Remote management agents can enable external control",
    ),
    MacOSSignalRule(
        title="Non-built-in display used (macOS)",
        category=SignalCategory.ENVIRONMENT_ISSUE,
        severity=Severity.MEDIUM,
        confidence="high",
        patterns=NON_BUILTIN_DISPLAY_PATTERNS,
        description_template="Non-built-in display is being used as main display",
        baseline_rule="May indicate external monitor when built-in required",
    ),
    MacOSSignalRule(
        title="Built-in display required but unavailable (macOS)",
        category=SignalCategory.POLICY_VIOLATION,
        severity=Severity.HIGH,
        confidence="high",
        patterns=BUILTIN_REQUIRED_UNAVAILABLE_PATTERNS,
        description_template="Policy requires built-in display but none available",
    ),
    MacOSSignalRule(
        title="SEB locked session (macOS)",
        category=SignalCategory.POLICY_VIOLATION,
        severity=Severity.HIGH,
        confidence="high",
        patterns=LOCK_SEB_PATTERNS,
        description_template="SEB locked the session due to policy violation",
    ),
    MacOSSignalRule(
        title="Reopening locked exam (macOS)",
        category=SignalCategory.INTEGRITY_ANOMALY,
        severity=Severity.HIGH,
        confidence="high",
        patterns=REOPENING_LOCKED_EXAM_PATTERNS,
        description_template="Attempting to reopen an exam that was previously locked/interrupted",
    ),
    MacOSSignalRule(
        title="Session interruption duration (macOS)",
        category=SignalCategory.POLICY_VIOLATION,
        severity=Severity.MEDIUM,
        confidence="high",
        patterns=SEB_LOCKED_DURATION_PATTERNS,
        description_template="Session was locked for a period of time",
    ),
    MacOSSignalRule(
        title="Lock screen password entered (macOS)",
        category=SignalCategory.BLOCKED_ATTEMPT,
        severity=Severity.INFO,
        confidence="high",
        patterns=LOCK_PASSWORD_ENTERED_PATTERNS,
        description_template="Password was entered in the lock screen",
    ),
    MacOSSignalRule(
        title="Correct unlock password (macOS)",
        category=SignalCategory.BLOCKED_ATTEMPT,
        severity=Severity.INFO,
        confidence="high",
        patterns=CORRECT_PASSWORD_PATTERNS,
        description_template="Correct password entered to unlock session",
        enforced_by_tsb=True,
    ),
    MacOSSignalRule(
        title="Quit password requested (macOS)",
        category=SignalCategory.BLOCKED_ATTEMPT,
        severity=Severity.LOW,
        confidence="high",
        patterns=QUIT_PASSWORD_PATTERNS,
        description_template="Quit password dialog was displayed",
    ),
    MacOSSignalRule(
        title="Correct quit password entered (macOS)",
        category=SignalCategory.BLOCKED_ATTEMPT,
        severity=Severity.INFO,
        confidence="high",
        patterns=CORRECT_QUIT_PASSWORD_PATTERNS,
        description_template="Correct quit password was entered",
        enforced_by_tsb=True,
    ),
    MacOSSignalRule(
        title="Screen capture access required (macOS)",
        category=SignalCategory.COVERAGE_GAP,
        severity=Severity.MEDIUM,
        confidence="high",
        patterns=SCREEN_CAPTURE_ACCESS_PATTERNS,
        description_template="User needed to grant screen capture access",
    ),
    MacOSSignalRule(
        title="Previous session did not quit properly (macOS)",
        category=SignalCategory.TECHNICAL_PROBLEM,
        severity=Severity.LOW,
        confidence="high",
        patterns=SCREEN_CAPTURE_LEFTOVER_PATTERNS,
        description_template="Found leftover screen capture redirect from previous session",
    ),
    MacOSSignalRule(
        title="Native system confirmed (macOS)",
        category=SignalCategory.ENVIRONMENT_ISSUE,
        severity=Severity.INFO,
        confidence="high",
        patterns=NO_VM_PATTERNS,
        description_template="Confirmed running on native system, not a VM",
    ),
    MacOSSignalRule(
        title="Kiosk mode active (macOS)",
        category=SignalCategory.ENVIRONMENT_ISSUE,
        severity=Severity.INFO,
        confidence="high",
        patterns=KIOSK_MODE_PATTERNS,
        description_template="SEB kiosk mode is active",
    ),
    MacOSSignalRule(
        title="Browser window close blocked (macOS)",
        category=SignalCategory.BLOCKED_ATTEMPT,
        severity=Severity.LOW,
        confidence="high",
        patterns=BROWSER_CLOSE_BLOCKED_PATTERNS,
        description_template="Web app attempted to close browser window, blocked by SEB",
        enforced_by_tsb=True,
    ),
    MacOSSignalRule(
        title="Configuration loaded (macOS)",
        category=SignalCategory.ENVIRONMENT_ISSUE,
        severity=Severity.INFO,
        confidence="high",
        patterns=CONFIG_LOADED_PATTERNS,
        description_template="SEB configuration file was loaded",
    ),
    MacOSSignalRule(
        title="Settings changed (macOS)",
        category=SignalCategory.CONFIG_DEVIATION,
        severity=Severity.INFO,
        confidence="high",
        patterns=SETTINGS_CHANGED_PATTERNS,
        description_template="SEB settings were changed",
    ),
    MacOSSignalRule(
        title="Process watcher started (macOS)",
        category=SignalCategory.ENVIRONMENT_ISSUE,
        severity=Severity.INFO,
        confidence="high",
        patterns=PROCESS_WATCHER_PATTERNS,
        description_template="Process monitoring started",
    ),
    MacOSSignalRule(
        title="Window watcher started (macOS)",
        category=SignalCategory.ENVIRONMENT_ISSUE,
        severity=Severity.INFO,
        confidence="high",
        patterns=WINDOW_WATCHER_PATTERNS,
        description_template="Window monitoring started",
    ),
]


def _title_to_slug(title: str) -> str:
    """Convert a rule title to a deterministic signal ID slug."""
    return title.lower().replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
