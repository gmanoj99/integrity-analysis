"""macOS-specific noise identification rules for SEB log reduction.

These rules identify high-volume, low-signal log entries in macOS SEB logs
that should be collapsed into summary counts rather than preserved individually.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..contracts import NoiseCategory
from ..grammars.base import ParsedLogEntry


@dataclass
class MacOSNoiseRule:
    """A rule for identifying noise entries in macOS logs."""

    category: NoiseCategory
    patterns: list[re.Pattern[str]]

    def matches(self, message: str, raw_line: str) -> bool:
        """Check if a message matches this noise rule."""
        for pattern in self.patterns:
            if pattern.search(message) or pattern.search(raw_line):
                return True
        return False


# Window geometry churn
WINDOW_GEOMETRY_PATTERNS = [
    re.compile(r"Usable screen frame", re.IGNORECASE),
    re.compile(r"Adjusted window frame for new screen", re.IGNORECASE),
    re.compile(r"Height\s*=\s*\d+;", re.IGNORECASE),
    re.compile(r"Width\s*=\s*\d+;", re.IGNORECASE),
    re.compile(r"X\s*=\s*\d+;", re.IGNORECASE),
    re.compile(r"Y\s*=\s*\d+;", re.IGNORECASE),
    re.compile(r"windowDidChangeScreen from previous", re.IGNORECASE),
    re.compile(r"\[SEBBrowserWindowController adjustWindowForScreen", re.IGNORECASE),
]

# Covering window churn
COVERING_WINDOW_PATTERNS = [
    re.compile(r"Opening background covering window with frame", re.IGNORECASE),
    re.compile(r"Opening lockdown alert covering window", re.IGNORECASE),
    re.compile(r"\[SEBController closeCoveringWindows:\]", re.IGNORECASE),
    re.compile(r"\[SEBController coverScreens\]", re.IGNORECASE),
    re.compile(r"Cap window <CapWindowController:", re.IGNORECASE),
    re.compile(r"requestedReinforceKioskMode", re.IGNORECASE),
]

# Screen parameter change notifications
SCREEN_PARAMETER_PATTERNS = [
    re.compile(r"NSApplicationDidChangeScreenParametersNotification", re.IGNORECASE),
    re.compile(r"\[SEBController adjustScreenLocking:\]", re.IGNORECASE),
    re.compile(r"Screen is active, device description", re.IGNORECASE),
    re.compile(r"NSDeviceBitsPerSample", re.IGNORECASE),
    re.compile(r"NSDeviceColorSpaceName", re.IGNORECASE),
    re.compile(r"NSDeviceIsScreen", re.IGNORECASE),
    re.compile(r"NSDeviceResolution", re.IGNORECASE),
    re.compile(r"NSDeviceSize", re.IGNORECASE),
    re.compile(r"NSScreenNumber", re.IGNORECASE),
    re.compile(r"All available screens:", re.IGNORECASE),
    re.compile(r"Move all browser windows to new main screen", re.IGNORECASE),
    re.compile(r"Use built-in option set, using display", re.IGNORECASE),
    re.compile(r"Found matching non built-in screen.*for main display", re.IGNORECASE),
]

# NSUserDefaults suite additions
USERDEFAULTS_SUITE_PATTERNS = [
    re.compile(r"addSuiteNamed:", re.IGNORECASE),
    re.compile(r"\[NSUserDefaults\(SEBEncryptedUserDefaults\) valueForDefaultsDomain", re.IGNORECASE),
]

# Kiosk mode level adjustments
KIOSK_LEVEL_PATTERNS = [
    re.compile(r"NSApp setPresentationOptions:", re.IGNORECASE),
    re.compile(r"\[SEBController changeWindowLevels:\]", re.IGNORECASE),
    re.compile(r"\[SEBOSXBrowserController browserWindowsChangeLevelAllowApps:\]", re.IGNORECASE),
    re.compile(r"\[SEBOSXBrowserController setLevelForBrowserWindow:", re.IGNORECASE),
    re.compile(r"\[SEBController switchKioskModeAppsAllowed:", re.IGNORECASE),
    re.compile(r"\[SEBController startKioskModeThirdPartyAppsAllowed:", re.IGNORECASE),
    re.compile(r"\[SEBController adjustModalAlertWindowLevels:\]", re.IGNORECASE),
    re.compile(r"\[SEBDockController adjustDock\]", re.IGNORECASE),
    re.compile(r"\[SEBDockController moveDockToScreen:", re.IGNORECASE),
]

# Running app inventory (very high volume during startup)
RUNNING_APP_PATTERNS = [
    re.compile(r"^Running app:", re.IGNORECASE),
]

# Browser window focus changes
BROWSER_WINDOW_FOCUS_PATTERNS = [
    re.compile(r"BrowserWindow.*did become key", re.IGNORECASE),
    re.compile(r"BrowserWindow.*did resign key", re.IGNORECASE),
    re.compile(r"BrowserWindow.*did become main", re.IGNORECASE),
    re.compile(r"Current key window:", re.IGNORECASE),
]

# Regain active status churn
REGAIN_ACTIVE_PATTERNS = [
    re.compile(r"Regain active status after", re.IGNORECASE),
    re.compile(r"isActive property of SEB changed", re.IGNORECASE),
    re.compile(r"Active window:", re.IGNORECASE),
    re.compile(r"SEB got active", re.IGNORECASE),
]


MACOS_NOISE_RULES: list[MacOSNoiseRule] = [
    MacOSNoiseRule(
        category=NoiseCategory.MACOS_WINDOW_GEOMETRY,
        patterns=WINDOW_GEOMETRY_PATTERNS,
    ),
    MacOSNoiseRule(
        category=NoiseCategory.MACOS_COVERING_WINDOW_CHURN,
        patterns=COVERING_WINDOW_PATTERNS,
    ),
    MacOSNoiseRule(
        category=NoiseCategory.MACOS_SCREEN_PARAMETER_CHANGE,
        patterns=SCREEN_PARAMETER_PATTERNS,
    ),
    MacOSNoiseRule(
        category=NoiseCategory.MACOS_USERDEFAULTS_SUITE,
        patterns=USERDEFAULTS_SUITE_PATTERNS,
    ),
    MacOSNoiseRule(
        category=NoiseCategory.MACOS_KIOSK_LEVEL_ADJUSTMENT,
        patterns=KIOSK_LEVEL_PATTERNS,
    ),
]


def check_macos_noise(entry: ParsedLogEntry) -> NoiseCategory | None:
    """Check if a macOS log entry is noise.

    Returns the NoiseCategory if matched, None otherwise.
    """
    message = entry.message
    raw_line = entry.raw_line

    for rule in MACOS_NOISE_RULES:
        if rule.matches(message, raw_line):
            return rule.category

    return None
