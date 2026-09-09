"""SEB Log AI Analysis prompt text.

This module contains the system prompt and user prompt builder for
interpreting reduced SEB log evidence. The deterministic layer has
already extracted, deduplicated, and categorized signals - the AI's
role is to interpret, correlate, and prioritize for human reviewers.
"""

from __future__ import annotations

from typing import Any

from .shared import DEFAULT_GEMINI_PRO_MODEL

SEB_LOG_ANALYSIS_PROMPT_VERSION = "seblog-analysis-v1"
SEB_LOG_ANALYSIS_MODEL = DEFAULT_GEMINI_PRO_MODEL

SEB_LOG_ANALYSIS_CAPS = {
    "signals": 60,
    "evidence_lines_per_signal": 3,
    "timeline": 80,
    "processes": 40,
    "foreground_spans": 40,
    "navigations": 20,
    "noise_rows": 12,
    "config_fields": 30,
    "correlated_incidents": 20,
}

JOURNEY_PHASES = [
    "launch",
    "configuration_load",
    "integrity_verification",
    "environment_checks",
    "lockdown_application",
    "kiosk_initialisation",
    "browser_initialisation",
    "exam_navigation",
    "exam_active",
    "session_stop",
    "shutdown",
]

FINDING_CATEGORIES = [
    "environment_integrity",
    "input_manipulation",
    "application_activity",
    "window_and_focus",
    "browser_activity",
    "configuration_and_policy",
    "session_control",
    "monitoring_coverage",
]

ENFORCEMENT_VALUES = {
    "blocked_by_tsb": [
        "blocked_keystroke",
        "blocked_mouse_action",
        "blocked_navigation",
        "blocked_download",
        "blocked_popup",
        "blacklisted_application_terminated",
        "quit_password_attempt_failed",
        # macOS signals with enforced_by_tsb=true
        "process_terminated_by_seb_macos",
        "session_terminated_processes_summary_macos",
        "correct_unlock_password_macos",
        "correct_quit_password_entered_macos",
        "browser_window_close_blocked_macos",
    ],
    "restriction_breached": [
        "blacklist_termination_failed",
    ],
}

SEB_LOG_ANALYSIS_SYSTEM_PROMPT = """You are a SEB/TSB log analysis expert interpreting exam session evidence.

=== ROLE AND DIVISION OF LABOUR ===

The deterministic layer has already:
- Parsed all log files and extracted timestamps
- Identified and deduplicated security signals with occurrence counts
- Categorized each signal with severity and confidence
- Correlated signals within 10-second windows into incidents
- Detected coverage gaps and monitoring status
- Aggregated bulk termination events
- Tracked processes and foreground window activity

Your role is to INTERPRET, CORRELATE, and PRIORITIZE what this means together for a human reviewer. You do NOT:
- Rediscover raw log patterns (signals are pre-extracted)
- Assign base severity (already set by deterministic rules)
- Decide session verdict (computed from your findings in code)
- Conclude that the candidate cheated (strictly forbidden)

=== INPUT DESCRIPTION ===

The reduced session evidence contains:
- signals[]: Pre-extracted security signals with category, severity, occurrenceCount, firstSeen, lastSeen, and evidence lines. Absence of a signal means the extractor did not emit it, NOT that the event did not occur.
- correlatedIncidents[]: Time-clustered signal groups (10-second window)
- timeline[]: Warning/error events (capped, severity reflects log level not significance)
- sessionPhases[]: Banner-marked phases only
- environment: Machine and OS info, VM indicators, remote session status
- effectiveConfig[]: Observed configuration with determinedBy provenance
- monitoringCoverage: Armed monitors, gaps, lockdown status
- processes[]: Started processes with signature info
- foregroundActivity[]: Window focus spans with allowed/hidden status
- browserActivity: Navigation, blocked requests, page errors
- bulkTerminations[]: Aggregated startup termination sweeps
- noiseSummary[]: Collapsed noise counts (pings, WebRTC errors, etc.)
- caveats[]: Known data quality issues

=== HARD RULES ===

1. NO INVENTED EVENTS - cite only signals, incidents, and evidence that exist in the input
2. CITATIONS VERBATIM - copy signal IDs, evidence references, and config field names exactly
3. EFFECTIVE CONFIG OUTRANKS DEFAULTS - if the effective config permits an action, it is not a violation
4. BLOCKED IS NOT BREACHED - enforced_by_tsb=true means the action was prevented, not successful
5. TECHNICAL IS NOT MALPRACTICE - network errors and crashes are technical problems, not suspicious
6. NEVER CONCLUDE CHEATING - you may identify suspicious patterns but never accuse
7. SECRETS STAY REDACTED - do not attempt to reconstruct redacted information
8. INTERPRET EVIDENCE TEXT - signal titles are extractor labels. If the cited evidence contradicts the title, follow the evidence. In particular: application-blacklist initialization listing vmware.exe/virtualbox.exe/qemu.exe/parallels.exe is NOT VM presence; VirtualMachineDetector stating the computer "appears not to be a virtual machine" is a negative result; "Detected N active displays, N are allowed" is compliance, not a display-policy breach.

=== JOURNEY PHASES ===

Structure the session into these phases: launch, configuration_load, integrity_verification, environment_checks, lockdown_application, kiosk_initialisation, browser_initialisation, exam_navigation, exam_active, session_stop, shutdown.

For each phase, report:
- status: "observed" (explicit banner or clear evidence), "inferred" (derived from timestamps/modules), or "not_observed"
- basis: the evidence placing the phase
- keyEvents[]: Only materially significant events; aggregate repeats

Example of proper aggregation: "37 blocked keystrokes between 10:42:00 and 10:44:30 including repeated Alt+Tab attempts" (not 37 separate findings)

=== FINDING CATEGORIES ===

Group findings into: environment_integrity, input_manipulation, application_activity, window_and_focus, browser_activity, configuration_and_policy, session_control, monitoring_coverage.

=== ENFORCEMENT DETERMINATION ===

Determine the enforcement field based on enforcedByTsb and evidence:
- "blocked_by_tsb": Action was attempted AND prevented. Signal IDs with enforced_by_tsb=true: blocked_keystroke, blocked_mouse_action, blocked_navigation, blocked_download, blocked_popup, blacklisted_application_terminated, quit_password_attempt_failed
- "restriction_breached": Action was attempted AND succeeded despite restrictions. Signal ID with enforced_by_tsb=false: blacklist_termination_failed
- "not_enforced": No enforcement mechanism was applicable
- "not_applicable": The finding is informational only

A blocked action MUST NOT be reported as a successful violation.

=== PROCESS AND WINDOW CONTEXT ===

processContext and windowContext are OPTIONAL structured objects on a
finding, used only for "application_activity" and "window_and_focus"
findings where the underlying process/window evidence is available.

- If not applicable, or the evidence does not name a specific
  process/window, set the field to `null`. NEVER omit it and NEVER return
  it as a plain string.
- If applicable, it MUST be a full JSON object with exactly these fields
  (all fields required unless marked optional; unknown values are `null`,
  never omitted):
  - processContext: {"name": string, "pid": integer, "path": string|null,
    "originalName": string|null, "signed": boolean|null}
  - windowContext: {"title": string, "handle": integer|null,
    "allowed": boolean|null, "actionTaken": string|null}

=== SEVERITY GUIDANCE ===

Calibration examples:
- CRITICAL: Injected keystrokes, established remote session, active blacklisted remote-control application
- HIGH: Developer console opened, display policy violation, cursor/accessibility tampering
- MEDIUM: Unknown foreground window (higher if TSB hid it), repeated blocked keystrokes with correlation
- LOW: Individual blocked keystroke, blocked popup
- INFO: Window focus change, process started

Timeline severity reflects LOG LEVEL not significance - treat timeline entries as supporting evidence, not primary signals.

Model severity may not exceed the deterministic severity of cited signals without explicit correlation justification.

=== CORRELATION ===

For correlated incidents, provide the chain as discrete fields:
- event: What happened
- context: Surrounding circumstances
- correlation: How events connect
- likelyInterpretation: What this pattern suggests
- classification: "suspicious", "technical", or "mixed"

Example (suspicious): Injected keystroke -> remote session active -> blacklisted app running = possible remote control attempt

Example (technical): Network disconnected -> request failed -> page errors -> resumed after reconnection = network interruption impact

=== CONFIGURATION AWARENESS ===

Check effectiveConfig before flagging policy violations:
- If window_guard is deactivated but config permits screen capture, this is not a violation
- If a service is ignored (service_ignored=true), this is a coverage gap, not misconduct
- Fields with determinedBy of "fork_default" or "unknown" have reduced certainty

You MUST NOT claim configuration differs from an expectation that was never stated. Only report deviations where the logs themselves show drift (e.g., OS lockdown drift signal, reconfiguration signal).

=== TECHNICAL VS SUSPICIOUS ===

Technical problems belong in technicalProblems[], not findings[]:
- Network errors, connection failures
- Browser crashes, page load failures
- Configuration download failures
- Service unavailability

When technical and suspicious events co-occur, explain the relationship rather than attribute intent:
"Network interruption occurred during suspected external consultation - unable to determine if related"

=== COVERAGE AND CONFIDENCE ===

Factors reducing analysis confidence:
- Missing log files (especially Service.log for lockdown verification)
- Coverage gaps > 2 minutes
- Low log level (DEBUG provides more evidence than ERROR-only)
- Monitors not armed
- Service ignored

conclusionQualifier MUST be one of:
- "no_suspicious_activity_observed": Coverage was sufficient AND no suspicious signals found
- "could_not_be_established": Coverage gaps, missing files, or other limitations prevent definitive assessment

=== REVIEWER SUMMARY ===

Provide exactly these elements:
- overallOutcome: Brief assessment (2-3 sentences)
- significantSecuritySignals[]: List of concerning signals with brief context
- significantTechnicalProblems[]: List of technical issues affecting the session
- configurationDeviations[]: Observed config drift
- coverageLimitations[]: What could not be verified
- eventsRequiringInvestigation[]: Specific items needing manual review
- summaryText: 250-300 words, neutral evidence-based prose. Cover (in order): session identity/environment and lockdown outcome; journey from launch through shutdown; significant security signals with when/what/enforcement; technical problems and coverage gaps; configuration deviations; what a reviewer should inspect next. Do not pad with speculation.

Use evidence-based language:
- "The logs indicate..." not "The candidate..."
- "Blocked keystroke patterns suggest..." not "The user tried to..."
- "Unable to verify due to..." not "The system failed to..."

=== OUTPUT FORMAT ===

Return valid JSON only:
{
  "sessionJourney": [
    {
      "phaseId": "launch",
      "phaseName": "Application Launch",
      "startedAt": "2024-01-15T10:00:00Z",
      "endedAt": "2024-01-15T10:00:05Z",
      "status": "observed|inferred|not_observed",
      "basis": "Session Start banner at line 15",
      "summary": "Application initialized normally",
      "keyEvents": ["Integrity verified", "Keyboard monitoring started"],
      "evidenceCited": ["runtime:15", "runtime:20"]
    }
  ],
  "findings": [
    {
      "findingId": "finding_001",
      "category": "input_manipulation",
      "title": "Repeated blocked keystroke attempts",
      "whatHappened": "37 Alt+Tab keystrokes blocked between 10:42 and 10:44",
      "whenText": "10:42:00 - 10:44:30",
      "whyItMatters": "Pattern suggests repeated attempts to switch applications",
      "severity": "medium",
      "confidence": "high",
      "occurrenceCount": 37,
      "firstSeen": "2024-01-15T10:42:00Z",
      "lastSeen": "2024-01-15T10:44:30Z",
      "enforcement": "blocked_by_tsb",
      "potentialImpact": "No impact - all attempts blocked",
      "benignExplanation": "User habit or frustration with restrictions",
      "requiresManualInvestigation": false,
      "signalsCited": ["blocked_keystroke"],
      "evidenceCited": ["client:100", "client:101"],
      "configCited": [],
      "processContext": null,
      "windowContext": null
    },
    {
      "findingId": "finding_002",
      "category": "application_activity",
      "title": "Unsigned process launched during exam",
      "whatHappened": "Process 'notes.exe' (pid 4821) started and its window was brought to foreground",
      "whenText": "10:45:12",
      "whyItMatters": "Unsigned third-party application running alongside the exam",
      "severity": "medium",
      "confidence": "medium",
      "occurrenceCount": 1,
      "firstSeen": "2024-01-15T10:45:12Z",
      "lastSeen": "2024-01-15T10:45:12Z",
      "enforcement": "not_enforced",
      "potentialImpact": "Could be used to view external material",
      "benignExplanation": "Common note-taking utility, may be unrelated to the exam",
      "requiresManualInvestigation": true,
      "signalsCited": ["unknown_foreground_window"],
      "evidenceCited": ["client:220"],
      "configCited": [],
      "processContext": {
        "name": "notes.exe",
        "pid": 4821,
        "path": "C:\\\\Users\\\\candidate\\\\AppData\\\\Local\\\\Notes\\\\notes.exe",
        "originalName": "notes.exe",
        "signed": false
      },
      "windowContext": {
        "title": "Untitled - Notes",
        "handle": 131072,
        "allowed": false,
        "actionTaken": "flagged"
      }
    }
  ],
  "correlatedIncidents": [
    {
      "incidentId": "incident_0001",
      "timeWindowStart": "2024-01-15T10:42:00Z",
      "timeWindowEnd": "2024-01-15T10:42:10Z",
      "classification": "suspicious|technical|mixed",
      "event": "What happened",
      "context": "Surrounding circumstances",
      "correlation": "How events connect",
      "likelyInterpretation": "What this suggests",
      "findingsCited": ["finding_001"],
      "evidenceCited": ["client:100"],
      "severity": "medium",
      "confidence": "high"
    }
  ],
  "technicalProblems": [
    {
      "problemType": "network_error",
      "whatHappened": "Connection lost for 30 seconds",
      "impactOnSession": "Page refresh required after reconnection",
      "relatedSuspiciousFindings": [],
      "relationshipExplanation": null
    }
  ],
  "configurationAssessment": {
    "appliedConfigSummary": "Standard exam lockdown with screen capture permitted",
    "deviations": [],
    "protectionsWeakened": [],
    "configExplainedSignals": ["window_guard_deactivated explained by screen capture permission"],
    "unknownOrDefaulted": ["field_x had unknown provenance"],
    "notes": []
  },
  "coverageAssessment": {
    "analysisConfidence": "high|medium|low",
    "missingLogFiles": [],
    "logGaps": ["2-minute gap in Runtime.log between 10:15 and 10:17"],
    "limitations": [],
    "conclusionQualifier": "no_suspicious_activity_observed|could_not_be_established"
  },
  "reviewerSummary": {
    "overallOutcome": "Session completed normally with standard blocked attempts",
    "significantSecuritySignals": [],
    "significantTechnicalProblems": [],
    "configurationDeviations": [],
    "coverageLimitations": [],
    "eventsRequiringInvestigation": [],
    "summaryText": "The session logs indicate a standard exam completion..."
  }
}

Note: verdict category and confidence are recomputed in code from surviving findings."""


def _cap_list(items: list[Any], cap: int, key: str = "") -> list[Any]:
    """Cap a list to maximum length, adding truncation note if needed."""
    if len(items) <= cap:
        return items
    return items[:cap]


def _serialize_signals(signals: list[dict[str, Any]], cap: int, evidence_cap: int) -> str:
    """Serialize signals for the user prompt."""
    if not signals:
        return "  (none)"

    lines = []
    for sig in _cap_list(signals, cap):
        sig_id = sig.get("signalId", "unknown")
        title = sig.get("title", "Unknown signal")
        category = sig.get("category", "unknown")
        severity = sig.get("severity", "unknown")
        count = sig.get("occurrenceCount", 1)
        enforced = sig.get("enforcedByTsb")
        enforced_str = f", enforcedByTsb={enforced}" if enforced is not None else ""
        first_seen = sig.get("firstSeen", "")
        last_seen = sig.get("lastSeen", "")

        lines.append(f"  [{sig_id}] {title}")
        lines.append(f"    category={category}, severity={severity}, count={count}{enforced_str}")
        lines.append(f"    firstSeen={first_seen}, lastSeen={last_seen}")

        evidence = sig.get("evidence", [])
        for ev in _cap_list(evidence, evidence_cap):
            file_type = ev.get("fileType", "?")
            line_num = ev.get("lineNumber", "?")
            text = ev.get("text", "")[:150]
            lines.append(f"    [{file_type}:{line_num}] {text}")

        captured = sig.get("capturedValues", {})
        if captured:
            for key, vals in captured.items():
                vals_str = ", ".join(vals[:5])
                if len(vals) > 5:
                    vals_str += f" (+{len(vals)-5} more)"
                lines.append(f"    captured.{key}=[{vals_str}]")

        lines.append("")

    return "\n".join(lines)


def _serialize_incidents(incidents: list[dict[str, Any]], cap: int) -> str:
    """Serialize correlated incidents for the user prompt."""
    if not incidents:
        return "  (none)"

    lines = []
    for inc in _cap_list(incidents, cap):
        inc_id = inc.get("incidentId", "unknown")
        start = inc.get("timeWindowStart", "")
        end = inc.get("timeWindowEnd", "")
        signals = inc.get("signals", [])
        desc = inc.get("description", "")

        lines.append(f"  [{inc_id}] {start} - {end}")
        lines.append(f"    signals: {', '.join(signals)}")
        lines.append(f"    description: {desc}")
        lines.append("")

    return "\n".join(lines)


def _serialize_timeline(timeline: list[dict[str, Any]], cap: int) -> str:
    """Serialize timeline events for the user prompt."""
    if not timeline:
        return "  (none)"

    lines = []
    for ev in _cap_list(timeline, cap):
        ts = ev.get("timestamp", "")
        source = ev.get("sourceFile", "?")
        severity = ev.get("severity", "?")
        event_type = ev.get("eventType", "unknown")
        desc = ev.get("description", "")[:150]

        lines.append(f"  [{ts}] [{source}:{severity}] {event_type}: {desc}")

    return "\n".join(lines)


def _serialize_phases(phases: list[dict[str, Any]]) -> str:
    """Serialize session phases for the user prompt."""
    if not phases:
        return "  (none observed via banners)"

    lines = []
    for phase in phases:
        name = phase.get("phaseName", "unknown")
        start = phase.get("startedAt", "")
        end = phase.get("endedAt", "")
        lines.append(f"  {name}: {start} - {end}")

    return "\n".join(lines)


def _serialize_environment(env: dict[str, Any] | None) -> str:
    """Serialize environment info for the user prompt."""
    if not env:
        return "  (not available)"

    lines = []
    if env.get("programVersion"):
        lines.append(f"  version: {env.get('programVersion')} build {env.get('programBuild', '?')}")
    if env.get("osName"):
        lines.append(f"  os: {env.get('osName')} {env.get('osVersion', '')}")
    if env.get("machineName"):
        lines.append(f"  machine: {env.get('machineName')} ({env.get('machineModel', '?')})")
    if env.get("architecture"):
        lines.append(f"  architecture: {env.get('architecture')}")

    vm_indicators = env.get("virtualMachineIndicators", [])
    if vm_indicators:
        lines.append(f"  VM indicators: {', '.join(vm_indicators)}")

    remote = env.get("remoteSession")
    if remote is not None:
        lines.append(f"  remoteSession: {remote}")

    return "\n".join(lines) if lines else "  (minimal info)"


def _serialize_config(config: list[dict[str, Any]], cap: int) -> str:
    """Serialize effective config for the user prompt."""
    if not config:
        return "  (not collected)"

    lines = []
    for field in _cap_list(config, cap):
        name = field.get("fieldName", "?")
        value = field.get("value", "?")
        determined = field.get("determinedBy", "unknown")
        lines.append(f"  {name} = {value} (via {determined})")

    return "\n".join(lines)


def _serialize_monitoring(monitoring: dict[str, Any] | None) -> str:
    """Serialize monitoring coverage for the user prompt."""
    if not monitoring:
        return "  (not available)"

    lines = []
    armed = monitoring.get("monitorsArmed", [])
    not_armed = monitoring.get("monitorsNotArmed", [])
    log_level = monitoring.get("logLevel")
    os_lockdowns = monitoring.get("osLockdownsApplied")
    service_ignored = monitoring.get("serviceIgnored")

    lines.append(f"  monitorsArmed: {', '.join(armed) if armed else '(none)'}")
    if not_armed:
        lines.append(f"  monitorsNotArmed: {', '.join(not_armed)}")
    if log_level:
        lines.append(f"  logLevel: {log_level}")
    if os_lockdowns is not None:
        lines.append(f"  osLockdownsApplied: {os_lockdowns}")
    if service_ignored:
        lines.append(f"  serviceIgnored: {service_ignored}")

    gaps = monitoring.get("coverageGaps", [])
    if gaps:
        lines.append(f"  coverageGaps: {len(gaps)} detected")
        for gap in gaps[:5]:
            file_type = gap.get("fileType", "?")
            duration = gap.get("durationSeconds", 0)
            lines.append(f"    {file_type}: {duration:.0f}s gap")

    return "\n".join(lines)


def _serialize_processes(processes: list[dict[str, Any]], cap: int) -> str:
    """Serialize process records for the user prompt."""
    if not processes:
        return "  (none observed)"

    lines = []
    for proc in _cap_list(processes, cap):
        name = proc.get("name", "?")
        pid = proc.get("pid", "?")
        path = proc.get("path", "")
        signed = proc.get("signed")
        terminated = proc.get("terminatedByTsb", False)

        sig_str = ""
        if signed is True:
            sig_str = " [signed]"
        elif signed is False:
            sig_str = " [unsigned]"

        term_str = " (terminated by TSB)" if terminated else ""
        lines.append(f"  [{pid}] {name}{sig_str}{term_str}")
        if path:
            lines.append(f"       path: {path[:100]}")

    return "\n".join(lines)


def _serialize_foreground(spans: list[dict[str, Any]], cap: int) -> str:
    """Serialize foreground window spans for the user prompt."""
    if not spans:
        return "  (none observed)"

    lines = []
    for span in _cap_list(spans, cap):
        title = span.get("windowTitle", "?")
        start = span.get("startTime", "")
        end = span.get("endTime", "")
        allowed = span.get("allowed")
        action = span.get("actionTaken", "none")

        allowed_str = ""
        if allowed is True:
            allowed_str = " [allowed]"
        elif allowed is False:
            allowed_str = " [not allowed]"

        action_str = f" action={action}" if action != "none" else ""
        lines.append(f"  {start} - {end}: '{title}'{allowed_str}{action_str}")

    return "\n".join(lines)


def _serialize_browser(browser: dict[str, Any] | None) -> str:
    """Serialize browser activity for the user prompt."""
    if not browser:
        return "  (not available)"

    lines = []
    lines.append(f"  windowsCreated: {browser.get('windowsCreated', 0)}")
    lines.append(f"  blockedRequests: {browser.get('blockedRequests', 0)}")
    lines.append(f"  blockedPopups: {browser.get('blockedPopups', 0)}")
    lines.append(f"  downloadsAttempted: {browser.get('downloadsAttempted', 0)}")

    navs = browser.get("navigations", [])
    if navs:
        lines.append(f"  navigations ({len(navs)}):")
        for nav in navs[:10]:
            lines.append(f"    {nav[:100]}")

    errors = browser.get("pageErrors", [])
    if errors:
        lines.append(f"  pageErrors ({len(errors)}):")
        for err in errors[:5]:
            lines.append(f"    {err[:100]}")

    return "\n".join(lines)


def _serialize_bulk_terminations(bulk: list[dict[str, Any]]) -> str:
    """Serialize bulk termination events for the user prompt."""
    if not bulk:
        return "  (none)"

    lines = []
    for bt in bulk:
        app = bt.get("applicationName", "?")
        count = bt.get("instanceCount", 0)
        force = bt.get("forceKillNeeded", False)
        all_term = bt.get("allTerminated", True)

        status = "all terminated" if all_term else "SOME NOT TERMINATED"
        force_str = " (force kill required)" if force else ""
        window = _format_window(bt.get("firstSeen"), bt.get("lastSeen"))
        lines.append(f"  {app}: {count} instances{window} - {status}{force_str}")

    return "\n".join(lines)


def _format_window(first_seen: Any, last_seen: Any) -> str:
    """Render a ' [first -> last]' span, collapsing an instantaneous one."""
    first = str(first_seen) if first_seen else ""
    last = str(last_seen) if last_seen else ""
    if not first and not last:
        return ""
    if not last or last == first:
        return f" [{first}]"
    return f" [{first} -> {last}]"


def _serialize_noise(noise: list[dict[str, Any]], cap: int) -> str:
    """Serialize noise summary for the user prompt."""
    if not noise:
        return "  (none collapsed)"

    lines = []
    for ns in _cap_list(noise, cap):
        category = ns.get("category", "?")
        count = ns.get("count", 0)
        lines.append(f"  {category}: {count} entries collapsed")

    return "\n".join(lines)


def _build_citation_inventory(evidence: dict[str, Any]) -> str:
    """Build the citation inventory for validation."""
    lines = ["=== CITATION INVENTORY ===", ""]

    lines.append("SIGNAL IDs:")
    for sig in evidence.get("signals", []):
        sig_id = sig.get("signalId", "")
        title = sig.get("title", "")
        lines.append(f"  {sig_id} ({title})")

    lines.append("")
    lines.append("INCIDENT IDs:")
    for inc in evidence.get("correlatedIncidents", []):
        inc_id = inc.get("incidentId", "")
        lines.append(f"  {inc_id}")

    lines.append("")
    lines.append("EVIDENCE REFERENCES (fileType:lineNumber):")
    evidence_refs = set()
    for sig in evidence.get("signals", []):
        for ev in sig.get("evidence", []):
            file_type = ev.get("fileType", "")
            line_num = ev.get("lineNumber", "")
            evidence_refs.add(f"{file_type}:{line_num}")
    for ref in sorted(evidence_refs):
        lines.append(f"  {ref}")

    lines.append("")
    lines.append("CONFIG FIELD NAMES:")
    for cfg in evidence.get("effectiveConfig", []):
        name = cfg.get("fieldName", "")
        lines.append(f"  {name}")

    lines.append("")
    lines.append("MONITOR NAMES:")
    monitoring = evidence.get("monitoringCoverage", {})
    for m in monitoring.get("monitorsArmed", []):
        lines.append(f"  {m} (armed)")
    for m in monitoring.get("monitorsNotArmed", []):
        lines.append(f"  {m} (not armed)")

    lines.append("")
    lines.append("PROCESS (name:pid):")
    for proc in evidence.get("processes", []):
        name = proc.get("name", "")
        pid = proc.get("pid", "")
        lines.append(f"  {name}:{pid}")

    lines.append("")
    lines.append("WINDOW TITLES:")
    seen_titles = set()
    for span in evidence.get("foregroundActivity", []):
        title = span.get("windowTitle", "")
        if title and title not in seen_titles:
            seen_titles.add(title)
            lines.append(f"  {title}")

    return "\n".join(lines)


def build_seb_log_analysis_user_prompt(
    *,
    reduced_evidence: dict[str, Any],
) -> str:
    """Build the user prompt from reduced session evidence.

    Args:
        reduced_evidence: The ReducedSessionEvidence as a dict (camelCase keys)

    Returns:
        The formatted user prompt string
    """
    caps = SEB_LOG_ANALYSIS_CAPS

    citation_inventory = _build_citation_inventory(reduced_evidence)

    signals_text = _serialize_signals(
        reduced_evidence.get("signals", []),
        caps["signals"],
        caps["evidence_lines_per_signal"],
    )

    incidents_text = _serialize_incidents(
        reduced_evidence.get("correlatedIncidents", []),
        caps["correlated_incidents"],
    )

    timeline_text = _serialize_timeline(
        reduced_evidence.get("timeline", []),
        caps["timeline"],
    )

    phases_text = _serialize_phases(
        reduced_evidence.get("sessionPhases", [])
    )

    env_text = _serialize_environment(
        reduced_evidence.get("environment")
    )

    config_text = _serialize_config(
        reduced_evidence.get("effectiveConfig", []),
        caps["config_fields"],
    )

    monitoring_text = _serialize_monitoring(
        reduced_evidence.get("monitoringCoverage")
    )

    processes_text = _serialize_processes(
        reduced_evidence.get("processes", []),
        caps["processes"],
    )

    foreground_text = _serialize_foreground(
        reduced_evidence.get("foregroundActivity", []),
        caps["foreground_spans"],
    )

    browser_text = _serialize_browser(
        reduced_evidence.get("browserActivity")
    )

    bulk_text = _serialize_bulk_terminations(
        reduced_evidence.get("bulkTerminations", [])
    )

    noise_text = _serialize_noise(
        reduced_evidence.get("noiseSummary", []),
        caps["noise_rows"],
    )

    caveats = reduced_evidence.get("caveats", [])
    caveats_text = "\n".join(f"  - {c}" for c in caveats) if caveats else "  (none)"

    unresolved = reduced_evidence.get("unresolved", [])
    unresolved_text = "\n".join(f"  - {u}" for u in unresolved) if unresolved else "  (none)"

    session_ref = reduced_evidence.get("sessionRef", {})
    session_id = session_ref.get("sessionFolder", "unknown")
    files_present = session_ref.get("filesPresent", [])
    files_missing = session_ref.get("filesMissing", [])
    platform = session_ref.get("platform", "windows")

    # Platform-specific note for macOS
    platform_note = ""
    if platform == "macos":
        platform_note = """
NOTE: This is a macOS session with a single unified log file.
On macOS, SEB produces one log file (org.safeexambrowser.SafeExamBrowser YYYY-MM-DD--HH-MM-SS-SSS.log)
instead of the four separate files on Windows (Runtime.log, Client.log, Browser.log, Service.log).
The absence of separate file types is expected and filesMissing being empty is normal for macOS."""

    return f"""{citation_inventory}

=== SESSION INFO ===
sessionId: {session_id}
platform: {platform}
filesPresent: {', '.join(files_present) if files_present else '(none)'}
filesMissing: {', '.join(files_missing) if files_missing else '(none)'}{platform_note}

=== EXTRACTED SIGNALS ===
{signals_text}

=== CORRELATED INCIDENTS ===
{incidents_text}

=== TIMELINE (warnings/errors, capped) ===
{timeline_text}

=== SESSION PHASES (banner-marked only) ===
{phases_text}

=== ENVIRONMENT ===
{env_text}

=== EFFECTIVE CONFIG ===
{config_text}

=== MONITORING COVERAGE ===
{monitoring_text}

=== PROCESSES ===
{processes_text}

=== FOREGROUND ACTIVITY ===
{foreground_text}

=== BROWSER ACTIVITY ===
{browser_text}

=== BULK TERMINATIONS ===
{bulk_text}

=== NOISE SUMMARY ===
{noise_text}

=== CAVEATS ===
{caveats_text}

=== UNRESOLVED ===
{unresolved_text}

Produce the JSON object described in the system instructions."""
