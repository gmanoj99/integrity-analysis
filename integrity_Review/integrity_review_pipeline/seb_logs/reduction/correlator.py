"""Correlator for clustering signals into incidents and detecting coverage gaps.

This module handles:
1. Session phase segmentation via session banners
2. Clustering signals within time windows into correlated incidents
3. Detecting coverage gaps (silences in otherwise chatty logs)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from ..contracts import (
    CorrelatedIncident,
    CoverageGap,
    ExtractedSignal,
    LogFileType,
    SessionPhase,
)


# Windows session banners
SESSION_START_BANNER = re.compile(
    r"###\s*-+\s*Session Start Procedure\s*-+\s*###", re.IGNORECASE
)
SESSION_STOP_BANNER = re.compile(
    r"###\s*-+\s*Session Stop Procedure\s*-+\s*###", re.IGNORECASE
)
SESSION_BANNER_GENERIC = re.compile(
    r"###\s*-+\s*(?P<phase>[^#]+?)\s*-+\s*###"
)

# macOS session banners
MACOS_SESSION_START_BANNER = re.compile(
    r"-+\s*INITIALIZING SEB\s*-\s*STARTING SESSION\s*-+", re.IGNORECASE
)
MACOS_SESSION_STOP_BANNER = re.compile(
    r"-+\s*EXITING SEB\s*-\s*ENDING SESSION\s*-+", re.IGNORECASE
)


@dataclass
class TimestampedEvent:
    """An event with a timestamp for correlation."""

    timestamp: datetime
    signal_id: str
    description: str
    file_type: LogFileType


@dataclass
class PhaseTracker:
    """Tracks session phases based on banners."""

    phases: list[SessionPhase] = field(default_factory=list)
    current_phase: str | None = None
    current_phase_start: datetime | None = None

    def check_line(self, line: str, timestamp: datetime | None) -> bool:
        """Check if a line is a session banner and update phases accordingly.

        Returns True if the line was a banner.
        """
        # Windows banners
        if SESSION_START_BANNER.search(line):
            self._end_current_phase(timestamp)
            self.current_phase = "Session Start Procedure"
            self.current_phase_start = timestamp
            return True

        if SESSION_STOP_BANNER.search(line):
            self._end_current_phase(timestamp)
            self.current_phase = "Session Stop Procedure"
            self.current_phase_start = timestamp
            return True

        if match := SESSION_BANNER_GENERIC.search(line):
            phase_name = match.group("phase").strip()
            self._end_current_phase(timestamp)
            self.current_phase = phase_name
            self.current_phase_start = timestamp
            return True

        # macOS banners
        if MACOS_SESSION_START_BANNER.search(line):
            self._end_current_phase(timestamp)
            self.current_phase = "Session Start (macOS)"
            self.current_phase_start = timestamp
            return True

        if MACOS_SESSION_STOP_BANNER.search(line):
            self._end_current_phase(timestamp)
            self.current_phase = "Session Stop (macOS)"
            self.current_phase_start = timestamp
            return True

        return False

    def _end_current_phase(self, end_time: datetime | None) -> None:
        """End the current phase if one is active."""
        if self.current_phase and self.current_phase_start:
            self.phases.append(
                SessionPhase(
                    phase_name=self.current_phase,
                    started_at=self.current_phase_start,
                    ended_at=end_time,
                )
            )

    def finalize(self, last_timestamp: datetime | None) -> list[SessionPhase]:
        """Finalize and return all phases."""
        self._end_current_phase(last_timestamp)
        return self.phases


class SignalCorrelator:
    """Correlates signals into incidents based on time proximity."""

    def __init__(
        self,
        time_window_seconds: float = 10.0,
    ) -> None:
        """Initialize correlator.

        Args:
            time_window_seconds: Maximum time between signals to cluster them
                               into the same incident (default 10s per naive prompt).
        """
        self._time_window = timedelta(seconds=time_window_seconds)
        self._events: list[TimestampedEvent] = []
        self._incident_counter = 0

    def add_signal(
        self,
        signal: ExtractedSignal,
    ) -> None:
        """Add a signal for correlation."""
        if signal.first_seen:
            self._events.append(
                TimestampedEvent(
                    timestamp=signal.first_seen,
                    signal_id=signal.signal_id,
                    description=signal.title,
                    file_type=signal.evidence[0].file_type if signal.evidence else LogFileType.RUNTIME,
                )
            )
        if signal.last_seen and signal.last_seen != signal.first_seen:
            self._events.append(
                TimestampedEvent(
                    timestamp=signal.last_seen,
                    signal_id=signal.signal_id,
                    description=signal.title,
                    file_type=signal.evidence[-1].file_type if signal.evidence else LogFileType.RUNTIME,
                )
            )

    def correlate(self) -> list[CorrelatedIncident]:
        """Cluster events into correlated incidents.

        Returns a list of incidents where each incident contains at least
        two distinct signals that occurred within the configured time window.
        Single-signal clusters are not considered incidents.
        """
        if not self._events:
            return []

        sorted_events = sorted(self._events, key=lambda e: e.timestamp)

        incidents: list[CorrelatedIncident] = []
        current_cluster: list[TimestampedEvent] = []

        for event in sorted_events:
            if not current_cluster:
                current_cluster.append(event)
            else:
                time_diff = event.timestamp - current_cluster[-1].timestamp
                if time_diff <= self._time_window:
                    current_cluster.append(event)
                else:
                    incident = self._try_create_incident(current_cluster)
                    if incident:
                        incidents.append(incident)
                    current_cluster = [event]

        incident = self._try_create_incident(current_cluster)
        if incident:
            incidents.append(incident)

        return incidents

    def _try_create_incident(
        self, events: list[TimestampedEvent]
    ) -> CorrelatedIncident | None:
        """Try to create an incident from a cluster of events.

        Returns None if the cluster has fewer than 2 distinct signals.
        """
        if len(events) < 2:
            return None

        distinct_signal_ids = {e.signal_id for e in events}
        if len(distinct_signal_ids) < 2:
            return None

        return self._create_incident(events)

    def _create_incident(
        self, events: list[TimestampedEvent]
    ) -> CorrelatedIncident:
        """Create an incident from a cluster of events."""
        self._incident_counter += 1
        signal_ids = list({e.signal_id for e in events})
        descriptions = list({e.description for e in events})

        return CorrelatedIncident(
            incident_id=f"incident_{self._incident_counter:04d}",
            time_window_start=events[0].timestamp,
            time_window_end=events[-1].timestamp,
            signals=signal_ids,
            description="; ".join(descriptions),
        )

    def reset(self) -> None:
        """Reset correlator for a new session."""
        self._events.clear()
        self._incident_counter = 0


# CEF writes Browser.log only on errors and console output, so silence there is
# normal operation. Measuring gaps against it invents coverage loss on quiet
# sessions and drags the AI's conclusion qualifier down with it.
EVENT_DRIVEN_FILE_TYPES = frozenset({LogFileType.BROWSER})


class CoverageGapDetector:
    """Detects gaps in log coverage."""

    def __init__(
        self,
        max_gap_seconds: float = 120.0,
        min_entries_for_gap_detection: int = 10,
    ) -> None:
        """Initialize gap detector.

        Args:
            max_gap_seconds: Maximum expected silence before flagging a gap
                           (default 120s per naive prompt).
            min_entries_for_gap_detection: Minimum number of entries needed
                                          to detect gaps (to avoid false positives
                                          on sparse logs).
        """
        self._max_gap = timedelta(seconds=max_gap_seconds)
        self._min_entries = min_entries_for_gap_detection
        self._timestamps: dict[LogFileType, list[datetime]] = {}

    def record_timestamp(
        self, timestamp: datetime | None, file_type: LogFileType
    ) -> None:
        """Record a timestamp for coverage tracking."""
        if timestamp is None:
            return

        if file_type not in self._timestamps:
            self._timestamps[file_type] = []
        self._timestamps[file_type].append(timestamp)

    def set_expected_interval(
        self, file_type: LogFileType, interval_seconds: float
    ) -> None:
        """Set expected interval for a file type based on observed cadence.

        Can be used to calibrate gap detection based on ping/heartbeat cadence.
        """
        pass

    def detect_gaps(self) -> list[CoverageGap]:
        """Detect coverage gaps in recorded timestamps."""
        gaps: list[CoverageGap] = []

        for file_type, timestamps in self._timestamps.items():
            if file_type in EVENT_DRIVEN_FILE_TYPES:
                continue
            if len(timestamps) < self._min_entries:
                continue

            sorted_ts = sorted(timestamps)

            for i in range(1, len(sorted_ts)):
                delta = sorted_ts[i] - sorted_ts[i - 1]
                if delta > self._max_gap:
                    gaps.append(
                        CoverageGap(
                            file_type=file_type,
                            gap_start=sorted_ts[i - 1],
                            gap_end=sorted_ts[i],
                            duration_seconds=delta.total_seconds(),
                            reason=f"No log entries for {delta.total_seconds():.0f}s",
                        )
                    )

        return sorted(gaps, key=lambda g: g.gap_start)

    def reset(self) -> None:
        """Reset detector for a new session."""
        self._timestamps.clear()
