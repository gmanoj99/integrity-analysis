"""Manifest parsing utilities for SEB log sessions.

This module provides utilities for loading and validating manifest.json
files from SEB/TSB exam sessions.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from .contracts import SessionManifest


def parse_manifest(data: dict[str, Any]) -> SessionManifest:
    """Parse manifest.json content into a SessionManifest contract.

    Args:
        data: Raw JSON data from manifest.json

    Returns:
        SessionManifest with parsed and validated fields
    """
    start_time = None
    if start_str := data.get("startTime"):
        try:
            if start_str.endswith("Z"):
                start_time = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
            else:
                start_time = datetime.fromisoformat(start_str)
        except ValueError:
            pass

    return SessionManifest(
        launch_id=data.get("launchId", "unknown"),
        environment=data.get("environment"),
        org_assessment_id=data.get("orgAssessmentId", ""),
        user_id=data.get("userId", ""),
        attempt_id=data.get("attemptId"),
        app_version=data.get("appVersion"),
        build_version=data.get("buildVersion"),
        os_version=data.get("osVersion"),
        machine_name=data.get("machineName"),
        start_time=start_time,
        declared_files=data.get("files", []),
    )


def validate_manifest_against_discovered_files(
    manifest: SessionManifest,
    discovered_files: list[str],
) -> list[str]:
    """Validate manifest declared files against actually discovered files.

    Args:
        manifest: Parsed manifest
        discovered_files: List of actually discovered file names

    Returns:
        List of discrepancy notes (empty if no issues)
    """
    notes: list[str] = []

    declared_set = set(manifest.declared_files)
    discovered_set = set(discovered_files)

    missing_from_s3 = declared_set - discovered_set
    extra_in_s3 = discovered_set - declared_set

    if missing_from_s3:
        notes.append(
            f"Manifest declares files not found in S3: {', '.join(sorted(missing_from_s3))}"
        )

    if extra_in_s3:
        notes.append(
            f"S3 contains files not declared in manifest: {', '.join(sorted(extra_in_s3))}"
        )

    return notes
