"""Noise identification rules for log reduction.

This module identifies high-volume, low-signal log entries that should be
collapsed into summary counts rather than preserved individually.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime

from ..contracts import LogFileType, NoiseCategory, NoiseCategorySummary
from ..grammars.base import ParsedLogEntry
from .macos_noise_rules import check_macos_noise


@dataclass
class NoiseMatch:
    """A match from a noise rule."""

    category: NoiseCategory
    entry: ParsedLogEntry


@dataclass
class NoiseAccumulator:
    """Accumulates noise matches into summaries."""

    category: NoiseCategory
    count: int = 0
    first_seen: datetime | None = None
    last_seen: datetime | None = None
    example_line: str | None = None

    def add(self, entry: ParsedLogEntry) -> None:
        """Add a noise entry to this accumulator."""
        self.count += 1
        if self.first_seen is None or (
            entry.timestamp and entry.timestamp < self.first_seen
        ):
            self.first_seen = entry.timestamp
        if self.last_seen is None or (
            entry.timestamp and entry.timestamp > self.last_seen
        ):
            self.last_seen = entry.timestamp
        if self.example_line is None:
            self.example_line = entry.raw_line[:300]

    def to_summary(self) -> NoiseCategorySummary:
        """Convert accumulator to summary contract."""
        return NoiseCategorySummary(
            category=self.category,
            count=self.count,
            first_seen=self.first_seen,
            last_seen=self.last_seen,
            example_line=self.example_line,
        )


@dataclass
class NoiseRule:
    """A rule for identifying noise entries."""

    category: NoiseCategory
    file_types: set[LogFileType]
    patterns: list[re.Pattern[str]]
    module_patterns: list[str] = field(default_factory=list)

    def matches(self, entry: ParsedLogEntry, file_type: LogFileType) -> bool:
        """Check if an entry matches this noise rule."""
        if file_type not in self.file_types:
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


PING_KEEPALIVE_PATTERNS = [
    re.compile(r"Ping", re.IGNORECASE),
    re.compile(r"Sending message.*Ping", re.IGNORECASE),
    re.compile(r"Received response.*Ping", re.IGNORECASE),
    re.compile(r"Alive.*Alive", re.IGNORECASE),
]

CONFIG_MONITOR_PATTERNS = [
    re.compile(r"Checking \d+ configurations"),
    re.compile(r"\[FeatureConfigurationMonitor\].*Checking"),
]

MICROPHONE_ERROR_PATTERNS = [
    re.compile(r"Failed to read the microphone input level", re.IGNORECASE),
    re.compile(r"\[Microphone\].*Failed to read", re.IGNORECASE),
]

WEBRTC_CAPTURE_PATTERNS = [
    re.compile(r"wgc_capture_session.*ProcessFrame failed"),
    re.compile(r"desktop_capture.*failed"),
]

GCM_REGISTRATION_PATTERNS = [
    re.compile(r"Registration response error message:"),
    re.compile(r"gcm.*registration", re.IGNORECASE),
    re.compile(r"PHONE_REGISTRATION_ERROR"),
    re.compile(r"DEPRECATED_ENDPOINT"),
]

CEF_FRAME_DETACHED_PATTERNS = [
    re.compile(r"sent to detached frame.*will be ignored"),
    re.compile(r"CefFrameImpl::SendProcessMessage"),
]

TURN_ALLOCATE_PATTERNS = [
    re.compile(r"Received TURN allocate error"),
    re.compile(r"turn_port.*allocate error"),
]

NOISE_RULES: list[NoiseRule] = [
    NoiseRule(
        category=NoiseCategory.PING_KEEPALIVE,
        file_types={LogFileType.RUNTIME, LogFileType.CLIENT, LogFileType.SERVICE},
        patterns=PING_KEEPALIVE_PATTERNS,
    ),
    NoiseRule(
        category=NoiseCategory.CONFIG_MONITOR_CHECK,
        file_types={LogFileType.SERVICE},
        patterns=CONFIG_MONITOR_PATTERNS,
        module_patterns=["FeatureConfigurationMonitor"],
    ),
    NoiseRule(
        category=NoiseCategory.MICROPHONE_READ_ERROR,
        file_types={LogFileType.CLIENT},
        patterns=MICROPHONE_ERROR_PATTERNS,
        module_patterns=["Microphone"],
    ),
    NoiseRule(
        category=NoiseCategory.WEBRTC_CAPTURE_ERROR,
        file_types={LogFileType.BROWSER},
        patterns=WEBRTC_CAPTURE_PATTERNS,
    ),
    NoiseRule(
        category=NoiseCategory.GCM_REGISTRATION,
        file_types={LogFileType.BROWSER},
        patterns=GCM_REGISTRATION_PATTERNS,
    ),
    NoiseRule(
        category=NoiseCategory.CEF_FRAME_DETACHED,
        file_types={LogFileType.BROWSER},
        patterns=CEF_FRAME_DETACHED_PATTERNS,
    ),
    NoiseRule(
        category=NoiseCategory.TURN_ALLOCATE_ERROR,
        file_types={LogFileType.BROWSER},
        patterns=TURN_ALLOCATE_PATTERNS,
    ),
]


class NoiseDetector:
    """Detects and categorizes noise entries in log files."""

    def __init__(self) -> None:
        self._accumulators: dict[NoiseCategory, NoiseAccumulator] = {}
        self._ping_timestamps: list[datetime] = []

    def check_entry(
        self, entry: ParsedLogEntry, file_type: LogFileType
    ) -> NoiseMatch | None:
        """Check if an entry is noise.

        Returns NoiseMatch if the entry matches a noise rule, None otherwise.
        """
        # Check Windows noise rules
        for rule in NOISE_RULES:
            if rule.matches(entry, file_type):
                match = NoiseMatch(category=rule.category, entry=entry)
                self._accumulate(match)
                return match

        # Check macOS noise rules (only for macOS unified log type)
        if file_type == LogFileType.MACOS_UNIFIED:
            macos_category = check_macos_noise(entry)
            if macos_category is not None:
                match = NoiseMatch(category=macos_category, entry=entry)
                self._accumulate(match)
                return match

        return None

    def _accumulate(self, match: NoiseMatch) -> None:
        """Add a match to the appropriate accumulator."""
        if match.category not in self._accumulators:
            self._accumulators[match.category] = NoiseAccumulator(
                category=match.category
            )

        self._accumulators[match.category].add(match.entry)

        if (
            match.category == NoiseCategory.PING_KEEPALIVE
            and match.entry.timestamp
        ):
            self._ping_timestamps.append(match.entry.timestamp)

    def get_summaries(self) -> list[NoiseCategorySummary]:
        """Get summaries for all accumulated noise categories."""
        return [acc.to_summary() for acc in self._accumulators.values() if acc.count > 0]

    def get_ping_timestamps(self) -> list[datetime]:
        """Get timestamps of ping messages for cadence analysis."""
        return sorted(self._ping_timestamps)

    def get_expected_ping_interval_seconds(self) -> float | None:
        """Estimate the expected interval between pings.

        Returns None if not enough data points, otherwise returns the
        median interval in seconds.
        """
        if len(self._ping_timestamps) < 3:
            return None

        sorted_ts = sorted(self._ping_timestamps)
        intervals = []
        for i in range(1, len(sorted_ts)):
            delta = (sorted_ts[i] - sorted_ts[i - 1]).total_seconds()
            if 0 < delta < 120:
                intervals.append(delta)

        if not intervals:
            return None

        intervals.sort()
        mid = len(intervals) // 2
        return intervals[mid]

    def reset(self) -> None:
        """Reset all accumulators for a new session."""
        self._accumulators.clear()
        self._ping_timestamps.clear()
