"""Validation rules for SEB Log AI Analysis responses.

This module provides deterministic post-LLM validation, similar to
the video pipeline's deliberation validation layer.
"""

from __future__ import annotations

from typing import Any


ENFORCEMENT_SIGNAL_MAPPING = {
    "blocked_by_tsb": {
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
    },
    "restriction_breached": {
        "blacklist_termination_failed",
    },
}

SEVERITY_ORDER = {
    "critical": 5,
    "high": 4,
    "medium": 3,
    "low": 2,
    "info": 1,
}


class ValidationResult:
    """Result of validating an AI analysis response."""

    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.rejected_findings: list[dict[str, Any]] = []

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def add_error(self, message: str) -> None:
        self.errors.append(message)

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)

    def reject_finding(self, finding_id: str, reason: str) -> None:
        self.rejected_findings.append({
            "finding_id": finding_id,
            "rejection_reason": reason,
        })


def build_citation_inventory(reduced_evidence: dict[str, Any]) -> dict[str, set[str]]:
    """Build a citation inventory from reduced evidence for validation."""
    inventory: dict[str, set[str]] = {
        "signal_ids": set(),
        "incident_ids": set(),
        "evidence_refs": set(),
        "config_fields": set(),
        "monitor_names": set(),
        "process_refs": set(),
        "window_titles": set(),
    }

    for sig in reduced_evidence.get("signals", []):
        sig_id = sig.get("signalId") or sig.get("signal_id")
        if sig_id:
            inventory["signal_ids"].add(sig_id)

        for ev in sig.get("evidence", []):
            file_type = ev.get("fileType") or ev.get("file_type", "")
            line_num = ev.get("lineNumber") or ev.get("line_number", "")
            if file_type and line_num:
                inventory["evidence_refs"].add(f"{file_type}:{line_num}")

    for inc in reduced_evidence.get("correlatedIncidents", []) or reduced_evidence.get("correlated_incidents", []):
        inc_id = inc.get("incidentId") or inc.get("incident_id")
        if inc_id:
            inventory["incident_ids"].add(inc_id)

    for cfg in reduced_evidence.get("effectiveConfig", []) or reduced_evidence.get("effective_config", []):
        field_name = cfg.get("fieldName") or cfg.get("field_name")
        if field_name:
            inventory["config_fields"].add(field_name)

    monitoring = reduced_evidence.get("monitoringCoverage") or reduced_evidence.get("monitoring_coverage", {})
    if monitoring:
        for m in monitoring.get("monitorsArmed", []) or monitoring.get("monitors_armed", []):
            inventory["monitor_names"].add(m)
        for m in monitoring.get("monitorsNotArmed", []) or monitoring.get("monitors_not_armed", []):
            inventory["monitor_names"].add(m)

    for proc in reduced_evidence.get("processes", []):
        name = proc.get("name", "")
        pid = proc.get("pid", "")
        if name and pid:
            inventory["process_refs"].add(f"{name}:{pid}")

    for span in reduced_evidence.get("foregroundActivity", []) or reduced_evidence.get("foreground_activity", []):
        title = span.get("windowTitle") or span.get("window_title")
        if title:
            inventory["window_titles"].add(title)

    return inventory


def get_signal_severity(reduced_evidence: dict[str, Any], signal_id: str) -> str | None:
    """Get the deterministic severity for a signal ID."""
    for sig in reduced_evidence.get("signals", []):
        sid = sig.get("signalId") or sig.get("signal_id")
        if sid == signal_id:
            return sig.get("severity", "info")
    return None


def get_signal_enforced_by_tsb(reduced_evidence: dict[str, Any], signal_id: str) -> bool | None:
    """Get the enforcedByTsb value for a signal ID."""
    for sig in reduced_evidence.get("signals", []):
        sid = sig.get("signalId") or sig.get("signal_id")
        if sid == signal_id:
            return sig.get("enforcedByTsb") or sig.get("enforced_by_tsb")
    return None


def validate_citations(
    analysis: dict[str, Any],
    inventory: dict[str, set[str]],
) -> ValidationResult:
    """Validate that all citations in the analysis exist in the inventory."""
    result = ValidationResult()

    for finding in analysis.get("findings", []):
        finding_id = finding.get("findingId") or finding.get("finding_id", "?")

        for sig_id in finding.get("signalsCited") or finding.get("signals_cited", []):
            if sig_id not in inventory["signal_ids"]:
                result.add_error(
                    f"Finding {finding_id} cites non-existent signal: {sig_id}"
                )

        for ev_ref in finding.get("evidenceCited") or finding.get("evidence_cited", []):
            if ev_ref not in inventory["evidence_refs"]:
                result.add_warning(
                    f"Finding {finding_id} cites evidence reference not in inventory: {ev_ref}"
                )

        for cfg_ref in finding.get("configCited") or finding.get("config_cited", []):
            if cfg_ref not in inventory["config_fields"]:
                result.add_warning(
                    f"Finding {finding_id} cites config field not in inventory: {cfg_ref}"
                )

    for incident in analysis.get("correlatedIncidents") or analysis.get("correlated_incidents", []):
        incident_id = incident.get("incidentId") or incident.get("incident_id", "?")

        for finding_id in incident.get("findingsCited") or incident.get("findings_cited", []):
            analysis_findings = {
                (f.get("findingId") or f.get("finding_id"))
                for f in analysis.get("findings", [])
            }
            if finding_id not in analysis_findings:
                result.add_warning(
                    f"Incident {incident_id} cites finding not in analysis: {finding_id}"
                )

    return result


def validate_severity(
    analysis: dict[str, Any],
    reduced_evidence: dict[str, Any],
) -> ValidationResult:
    """Validate that model severity does not exceed deterministic severity."""
    result = ValidationResult()

    for finding in analysis.get("findings", []):
        finding_id = finding.get("findingId") or finding.get("finding_id", "?")
        model_severity = finding.get("severity", "info").lower()
        signals_cited = finding.get("signalsCited") or finding.get("signals_cited", [])

        if not signals_cited:
            continue

        max_deterministic_severity = "info"
        for sig_id in signals_cited:
            det_severity = get_signal_severity(reduced_evidence, sig_id)
            if det_severity:
                if SEVERITY_ORDER.get(det_severity.lower(), 0) > SEVERITY_ORDER.get(max_deterministic_severity, 0):
                    max_deterministic_severity = det_severity.lower()

        if SEVERITY_ORDER.get(model_severity, 0) > SEVERITY_ORDER.get(max_deterministic_severity, 0):
            result.add_warning(
                f"Finding {finding_id} has severity '{model_severity}' exceeding "
                f"max deterministic severity '{max_deterministic_severity}' of cited signals"
            )

    return result


def validate_enforcement(
    analysis: dict[str, Any],
    reduced_evidence: dict[str, Any],
) -> ValidationResult:
    """Validate that enforcement status agrees with enforcedByTsb."""
    result = ValidationResult()

    for finding in analysis.get("findings", []):
        finding_id = finding.get("findingId") or finding.get("finding_id", "?")
        enforcement = finding.get("enforcement", "not_applicable")
        signals_cited = finding.get("signalsCited") or finding.get("signals_cited", [])

        for sig_id in signals_cited:
            enforced = get_signal_enforced_by_tsb(reduced_evidence, sig_id)

            if enforced is True and enforcement == "restriction_breached":
                result.add_error(
                    f"Finding {finding_id} claims restriction_breached but "
                    f"signal {sig_id} has enforcedByTsb=true (action was blocked)"
                )
                result.reject_finding(
                    finding_id,
                    f"Enforcement mismatch: {sig_id} was blocked, not breached"
                )

            if enforced is False and enforcement == "blocked_by_tsb":
                result.add_error(
                    f"Finding {finding_id} claims blocked_by_tsb but "
                    f"signal {sig_id} has enforcedByTsb=false (action succeeded)"
                )
                result.reject_finding(
                    finding_id,
                    f"Enforcement mismatch: {sig_id} was not blocked"
                )

    return result


def validate_coverage_conclusion(
    analysis: dict[str, Any],
    reduced_evidence: dict[str, Any],
) -> ValidationResult:
    """Validate that coverage conclusion is appropriate given gaps."""
    result = ValidationResult()

    coverage = analysis.get("coverageAssessment") or analysis.get("coverage_assessment", {})
    if not coverage:
        return result

    conclusion = coverage.get("conclusionQualifier") or coverage.get("conclusion_qualifier")
    missing_files = coverage.get("missingLogFiles") or coverage.get("missing_log_files", [])

    session_ref = reduced_evidence.get("sessionRef") or reduced_evidence.get("session_ref", {})
    files_missing = session_ref.get("filesMissing") or session_ref.get("files_missing", [])
    platform = session_ref.get("platform", "windows")

    if conclusion == "no_suspicious_activity_observed":
        # For macOS, empty filesMissing is normal since there's only one unified log
        if platform == "macos":
            pass
        elif files_missing or missing_files:
            result.add_warning(
                "conclusionQualifier is 'no_suspicious_activity_observed' but "
                f"files are missing: {files_missing or missing_files}"
            )

        monitoring = reduced_evidence.get("monitoringCoverage") or reduced_evidence.get("monitoring_coverage", {})
        gaps = monitoring.get("coverageGaps") or monitoring.get("coverage_gaps", [])
        significant_gaps = [g for g in gaps if (g.get("durationSeconds") or g.get("duration_seconds", 0)) > 120]

        if significant_gaps:
            result.add_warning(
                "conclusionQualifier is 'no_suspicious_activity_observed' but "
                f"{len(significant_gaps)} coverage gaps > 2 minutes exist"
            )

    return result


def validate_ai_response(
    analysis: dict[str, Any],
    reduced_evidence: dict[str, Any],
) -> ValidationResult:
    """Run all validation rules on an AI analysis response."""
    inventory = build_citation_inventory(reduced_evidence)

    combined = ValidationResult()

    citation_result = validate_citations(analysis, inventory)
    combined.errors.extend(citation_result.errors)
    combined.warnings.extend(citation_result.warnings)
    combined.rejected_findings.extend(citation_result.rejected_findings)

    severity_result = validate_severity(analysis, reduced_evidence)
    combined.errors.extend(severity_result.errors)
    combined.warnings.extend(severity_result.warnings)
    combined.rejected_findings.extend(severity_result.rejected_findings)

    enforcement_result = validate_enforcement(analysis, reduced_evidence)
    combined.errors.extend(enforcement_result.errors)
    combined.warnings.extend(enforcement_result.warnings)
    combined.rejected_findings.extend(enforcement_result.rejected_findings)

    coverage_result = validate_coverage_conclusion(analysis, reduced_evidence)
    combined.errors.extend(coverage_result.errors)
    combined.warnings.extend(coverage_result.warnings)
    combined.rejected_findings.extend(coverage_result.rejected_findings)

    return combined


def filter_rejected_findings(
    analysis: dict[str, Any],
    rejected: list[dict[str, Any]],
) -> dict[str, Any]:
    """Remove rejected findings from analysis and add rejection notes."""
    if not rejected:
        return analysis

    rejected_ids = {r["finding_id"] for r in rejected}

    filtered_findings = []
    for finding in analysis.get("findings", []):
        finding_id = finding.get("findingId") or finding.get("finding_id")
        if finding_id not in rejected_ids:
            filtered_findings.append(finding)

    analysis["findings"] = filtered_findings
    analysis["_rejectedFindings"] = rejected

    return analysis
