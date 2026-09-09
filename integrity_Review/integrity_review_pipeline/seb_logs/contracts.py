"""Pydantic contracts for SEB/TSB log analysis pipeline."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import Field

from ...contracts.base import ContractModel


class Platform(str, Enum):
    """Platform the SEB session ran on."""

    WINDOWS = "windows"
    MACOS = "macos"


class LogFileType(str, Enum):
    """Types of log files in a SEB/TSB session."""

    RUNTIME = "runtime"
    CLIENT = "client"
    BROWSER = "browser"
    SERVICE = "service"
    MANIFEST = "manifest"
    MACOS_UNIFIED = "macos_unified"


class Severity(str, Enum):
    """Log severity levels, ordered from most to least severe."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class SignalCategory(str, Enum):
    """Categories for extracted signals."""

    POLICY_VIOLATION = "policy_violation"
    BLOCKED_ATTEMPT = "blocked_attempt"
    INTEGRITY_ANOMALY = "integrity_anomaly"
    CONFIG_DEVIATION = "config_deviation"
    ENVIRONMENT_ISSUE = "environment_issue"
    COVERAGE_GAP = "coverage_gap"
    EXPECTED_NOISE = "expected_noise"
    TECHNICAL_PROBLEM = "technical_problem"


class NoiseCategory(str, Enum):
    """Categories for noise that gets deduplicated."""

    PING_KEEPALIVE = "ping_keepalive"
    CONFIG_MONITOR_CHECK = "config_monitor_check"
    WEBRTC_CAPTURE_ERROR = "webrtc_capture_error"
    GCM_REGISTRATION = "gcm_registration"
    MICROPHONE_READ_ERROR = "microphone_read_error"
    CEF_FRAME_DETACHED = "cef_frame_detached"
    TURN_ALLOCATE_ERROR = "turn_allocate_error"
    PROCESS_TERMINATION_ATTEMPT = "process_termination_attempt"
    # macOS-specific noise categories
    MACOS_WINDOW_GEOMETRY = "macos_window_geometry"
    MACOS_COVERING_WINDOW_CHURN = "macos_covering_window_churn"
    MACOS_SCREEN_PARAMETER_CHANGE = "macos_screen_parameter_change"
    MACOS_USERDEFAULTS_SUITE = "macos_userdefaults_suite"
    MACOS_KIOSK_LEVEL_ADJUSTMENT = "macos_kiosk_level_adjustment"


class ConfigDeterminedBy(str, Enum):
    """How an effective config value was determined."""

    CONFIG_FILE = "config_file"
    LOG_EVIDENCE = "log_evidence"
    FORK_DEFAULT = "fork_default"
    UNKNOWN = "unknown"


class RawLogFileMeta(ContractModel):
    """Metadata about a single log file in the session."""

    file_type: LogFileType
    s3_key: str
    line_count: int = 0
    first_timestamp: datetime | None = None
    last_timestamp: datetime | None = None
    missing: bool = False
    size_bytes: int = 0


class EvidenceLine(ContractModel):
    """A single line of evidence from a log file."""

    file_type: LogFileType
    line_number: int
    timestamp: datetime | None = None
    text: str = Field(..., max_length=500)


class ExtractedSignal(ContractModel):
    """A signal extracted from log analysis."""

    signal_id: str
    category: SignalCategory
    severity: Severity
    confidence: str = "high"
    title: str
    description: str
    baseline_rule: str | None = None
    enforced_by_tsb: bool | None = None
    occurrence_count: int = 1
    first_seen: datetime | None = None
    last_seen: datetime | None = None
    evidence: list[EvidenceLine] = Field(default_factory=list)
    captured_values: dict[str, list[str]] = Field(default_factory=dict)


class NoiseCategorySummary(ContractModel):
    """Summary of a deduplicated noise category."""

    category: NoiseCategory
    count: int
    first_seen: datetime | None = None
    last_seen: datetime | None = None
    example_line: str | None = None


class EffectiveConfigField(ContractModel):
    """A field from the effective configuration."""

    field_name: str
    value: Any
    determined_by: ConfigDeterminedBy = ConfigDeterminedBy.UNKNOWN
    evidence: list[EvidenceLine] = Field(default_factory=list)


class ProcessRecord(ContractModel):
    """Record of a process observed during the session."""

    name: str
    pid: int
    path: str | None = None
    original_name: str | None = None
    signed: bool | None = None
    signature_thumbprint: str | None = None
    first_seen: datetime | None = None
    last_seen: datetime | None = None
    classification: str = "unknown_third_party"
    interactive: bool | None = None
    terminated_by_tsb: bool = False


class ForegroundWindowSpan(ContractModel):
    """A span of time when a window was in the foreground."""

    window_title: str
    window_handle: int | None = None
    process_name: str | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    allowed: bool | None = None
    action_taken: str = "none"


class CoverageGap(ContractModel):
    """A gap in log coverage."""

    file_type: LogFileType
    gap_start: datetime
    gap_end: datetime
    duration_seconds: float
    reason: str | None = None


class CorrelatedIncident(ContractModel):
    """Multiple signals clustered into one incident."""

    incident_id: str
    time_window_start: datetime
    time_window_end: datetime
    signals: list[str] = Field(default_factory=list)
    description: str | None = None


class SessionPhase(ContractModel):
    """A phase of the session lifecycle."""

    phase_name: str
    started_at: datetime | None = None
    ended_at: datetime | None = None


class BulkTerminationEvent(ContractModel):
    """Summary of bulk process termination at startup."""

    application_name: str
    instance_count: int
    close_message_failures: int = 0
    close_window_failures: int = 0
    kill_failures: int = 0
    force_kill_needed: bool = False
    first_seen: datetime | None = None
    last_seen: datetime | None = None
    all_terminated: bool = True


class EnvironmentInfo(ContractModel):
    """Environment and machine identity information."""

    program_title: str | None = None
    program_version: str | None = None
    program_build: str | None = None
    architecture: str | None = None
    os_name: str | None = None
    os_version: str | None = None
    machine_name: str | None = None
    machine_model: str | None = None
    machine_manufacturer: str | None = None
    runtime_id: str | None = None
    client_id: str | None = None
    session_id: str | None = None
    virtual_machine_indicators: list[str] = Field(default_factory=list)
    remote_session: bool | None = None


class DisplayInfo(ContractModel):
    """Information about a display."""

    identifier: str
    active: bool = True
    internal: bool | None = None
    technology: str | None = None


class BrowserActivitySummary(ContractModel):
    """Summary of browser activity during the session."""

    windows_created: int = 0
    navigations: list[str] = Field(default_factory=list)
    blocked_requests: int = 0
    blocked_popups: int = 0
    downloads_attempted: int = 0
    uploads_blocked: int = 0
    dev_console_opened: bool = False
    quit_url_triggered: bool | None = None
    user_identifiers_detected: list[str] = Field(default_factory=list)
    page_errors: list[str] = Field(default_factory=list)
    chromium_version: str | None = None
    cef_version: str | None = None
    cefsharp_version: str | None = None


class MonitoringCoverage(ContractModel):
    """Information about what monitoring was active."""

    monitors_armed: list[str] = Field(default_factory=list)
    monitors_not_armed: list[str] = Field(default_factory=list)
    log_level: str | None = None
    os_lockdowns_applied: bool | None = None
    service_ignored: bool | None = None
    coverage_gaps: list[CoverageGap] = Field(default_factory=list)
    coverage_penalty_notes: list[str] = Field(default_factory=list)


class SessionManifest(ContractModel):
    """Parsed manifest.json content."""

    launch_id: str
    environment: str | None = None
    org_assessment_id: str
    user_id: str
    attempt_id: str | None = None
    app_version: str | None = None
    build_version: str | None = None
    os_version: str | None = None
    machine_name: str | None = None
    start_time: datetime | None = None
    declared_files: list[str] = Field(default_factory=list)


class SebLogSessionRef(ContractModel):
    """Reference to a session's log files in S3."""

    org_assessment_id: str
    user_id: str
    session_folder: str
    s3_prefix: str
    files_present: list[LogFileType] = Field(default_factory=list)
    files_missing: list[LogFileType] = Field(default_factory=list)
    manifest: SessionManifest | None = None
    platform: Platform = Platform.WINDOWS
    file_keys: dict[str, str] = Field(default_factory=dict)


class TimelineEvent(ContractModel):
    """An event in the session timeline."""

    timestamp: datetime
    source_file: LogFileType
    severity: Severity
    event_type: str
    description: str
    significance: str = "medium"


class ReducedSessionEvidence(ContractModel):
    """The final reduced output for one session."""

    session_ref: SebLogSessionRef
    analysed_files: list[RawLogFileMeta] = Field(default_factory=list)
    environment: EnvironmentInfo | None = None
    displays: list[DisplayInfo] = Field(default_factory=list)
    effective_config: list[EffectiveConfigField] = Field(default_factory=list)
    session_phases: list[SessionPhase] = Field(default_factory=list)
    bulk_terminations: list[BulkTerminationEvent] = Field(default_factory=list)
    processes: list[ProcessRecord] = Field(default_factory=list)
    foreground_activity: list[ForegroundWindowSpan] = Field(default_factory=list)
    browser_activity: BrowserActivitySummary | None = None
    monitoring_coverage: MonitoringCoverage | None = None
    noise_summary: list[NoiseCategorySummary] = Field(default_factory=list)
    signals: list[ExtractedSignal] = Field(default_factory=list)
    correlated_incidents: list[CorrelatedIncident] = Field(default_factory=list)
    timeline: list[TimelineEvent] = Field(default_factory=list)
    unresolved: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    reduction_stats: dict[str, Any] = Field(default_factory=dict)
