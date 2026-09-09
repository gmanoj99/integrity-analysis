"""Discovery module for finding SEB log sessions in S3."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterator
from datetime import datetime
from typing import Any

from .contracts import (
    LogFileType,
    Platform,
    SebLogSessionRef,
    SessionManifest,
)
from .deps import SebLogPipelineDeps

# Windows folders include a launch suffix (`2026-08-30_15h03m58s_21d3ad67`);
# macOS folders from the bucket often stop at the timestamp (`2026-08-19_12h07m55s`).
SESSION_FOLDER_PATTERN = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})_(?P<time>\d{2}h\d{2}m\d{2}s)"
    r"(?:_(?P<id>[a-f0-9]+))?$"
)

MACOS_LOG_PATTERN = re.compile(
    r"^org\.safeexambrowser\.SafeExamBrowser "
    r"(?P<date>\d{4}-\d{2}-\d{2})--(?P<time>\d{2}-\d{2}-\d{2}-\d{3})\.log$"
)

FILE_TYPE_MAP = {
    "Runtime.log": LogFileType.RUNTIME,
    "Client.log": LogFileType.CLIENT,
    "Browser.log": LogFileType.BROWSER,
    "Service.log": LogFileType.SERVICE,
    "manifest.json": LogFileType.MANIFEST,
}

EXPECTED_LOG_FILES = {
    LogFileType.RUNTIME,
    LogFileType.CLIENT,
    LogFileType.BROWSER,
    LogFileType.SERVICE,
}

EXPECTED_LOG_FILES_BY_PLATFORM = {
    Platform.WINDOWS: {
        LogFileType.RUNTIME,
        LogFileType.CLIENT,
        LogFileType.BROWSER,
        LogFileType.SERVICE,
    },
    Platform.MACOS: {
        LogFileType.MACOS_UNIFIED,
    },
}


def parse_manifest(data: dict[str, Any]) -> SessionManifest:
    """Parse manifest.json content into a SessionManifest contract."""
    start_time = None
    if start_str := data.get("startTime"):
        try:
            # Manifests write UTC with a trailing ``Z``, which ``fromisoformat``
            # accepts directly and turns into an aware datetime.
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


def _is_apple_double_or_macos_meta(filename: str) -> bool:
    """Check if a filename is an AppleDouble or __MACOSX metadata entry."""
    return filename.startswith("._") or "__MACOSX" in filename


def _parse_macos_log_timestamp(match: re.Match[str]) -> datetime | None:
    """Parse timestamp from macOS log filename pattern match."""
    try:
        date_str = match.group("date")
        time_str = match.group("time")
        ts_str = f"{date_str} {time_str.replace('-', ':')}"
        return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S:%f")
    except (ValueError, AttributeError):
        return None


def discover_sessions(
    deps: SebLogPipelineDeps,
    bucket_prefix: str,
    org_assessment_id: str | None = None,
    user_id: str | None = None,
) -> Iterator[SebLogSessionRef]:
    """Discover all SEB log sessions under a given S3 prefix.

    The expected S3 structure is:
    Windows:
    s3://{bucket}/{prefix}/topin_prod/media/tsb_logs/{orgAssessmentId}/{userId}/{sessionFolder}/
        - Runtime.log
        - Client.log
        - Browser.log
        - Service.log
        - manifest.json

    macOS:
    s3://{bucket}/{prefix}/.../{sessionFolder}/
        - org.safeexambrowser.SafeExamBrowser YYYY-MM-DD--HH-MM-SS-SSS.log
        - (optional) manifest.json

    Args:
        deps: Pipeline dependencies (object store, logger)
        bucket_prefix: The S3 prefix to search under
        org_assessment_id: Optional filter for a specific org assessment
        user_id: Optional filter for a specific user

    Yields:
        SebLogSessionRef for each discovered session
    """
    deps.logger.info(
        "Starting session discovery",
        bucket_prefix=bucket_prefix,
        org_filter=org_assessment_id,
        user_filter=user_id,
    )

    search_prefix = bucket_prefix.rstrip("/") + "/"

    if org_assessment_id:
        search_prefix = f"{search_prefix}{org_assessment_id}/"
        if user_id:
            search_prefix = f"{search_prefix}{user_id}/"

    sessions: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"files": set(), "keys": {}, "macos_logs": [], "platform": Platform.WINDOWS}
    )

    for key in deps.object_store.list_keys(search_prefix):
        parts = key.split("/")

        # Skip AppleDouble and __MACOSX metadata entries
        if any(_is_apple_double_or_macos_meta(p) for p in parts):
            continue

        try:
            session_folder_idx = None
            for i, part in enumerate(parts):
                if SESSION_FOLDER_PATTERN.match(part):
                    session_folder_idx = i
                    break

            if session_folder_idx is None:
                continue

            if session_folder_idx < 2:
                continue

            org_id = parts[session_folder_idx - 2]
            uid = parts[session_folder_idx - 1]
            session_folder = parts[session_folder_idx]

            if org_assessment_id and org_id != org_assessment_id:
                continue
            if user_id and uid != user_id:
                continue

            session_key = f"{org_id}/{uid}/{session_folder}"

            if session_folder_idx + 1 < len(parts):
                filename = parts[session_folder_idx + 1]

                # Check for macOS log file
                macos_match = MACOS_LOG_PATTERN.match(filename)
                if macos_match:
                    sessions[session_key]["platform"] = Platform.MACOS
                    macos_ts = _parse_macos_log_timestamp(macos_match)
                    sessions[session_key]["macos_logs"].append({
                        "filename": filename,
                        "key": key,
                        "timestamp": macos_ts,
                    })
                elif filename in FILE_TYPE_MAP:
                    file_type = FILE_TYPE_MAP[filename]
                    sessions[session_key]["files"].add(file_type)
                    sessions[session_key]["keys"][file_type] = key

            sessions[session_key]["org_assessment_id"] = org_id
            sessions[session_key]["user_id"] = uid
            sessions[session_key]["session_folder"] = session_folder
            sessions[session_key]["s3_prefix"] = "/".join(
                parts[: session_folder_idx + 1]
            ) + "/"

        except (IndexError, ValueError) as e:
            deps.logger.warning(
                "Failed to parse S3 key path", key=key, error=str(e)
            )
            continue

    deps.logger.info(
        "Discovery found sessions", count=len(sessions)
    )

    for session_key, session_data in sessions.items():
        platform = session_data["platform"]
        caveats: list[str] = []

        # Determine files present and file_keys based on platform
        file_keys: dict[str, str] = {}

        if platform == Platform.MACOS:
            macos_logs = session_data["macos_logs"]
            if macos_logs:
                # Select the macOS log with the latest timestamp
                macos_logs_sorted = sorted(
                    macos_logs,
                    key=lambda x: x["timestamp"] or datetime.min,
                    reverse=True,
                )
                selected_log = macos_logs_sorted[0]
                files_present = [LogFileType.MACOS_UNIFIED]
                file_keys[LogFileType.MACOS_UNIFIED.value] = selected_log["key"]

                # Add caveat if multiple logs were present
                if len(macos_logs_sorted) > 1:
                    ignored_files = [log["filename"] for log in macos_logs_sorted[1:]]
                    caveats.append(
                        f"Multiple macOS logs found; using {selected_log['filename']}, "
                        f"ignored: {', '.join(ignored_files)}"
                    )
            else:
                files_present = []

            # Also include manifest if present
            if LogFileType.MANIFEST in session_data["files"]:
                files_present.append(LogFileType.MANIFEST)
                file_keys[LogFileType.MANIFEST.value] = session_data["keys"][LogFileType.MANIFEST]

            expected_files = EXPECTED_LOG_FILES_BY_PLATFORM[Platform.MACOS]
        else:
            files_present = list(session_data["files"])
            file_keys = {ft.value: key for ft, key in session_data["keys"].items()}
            expected_files = EXPECTED_LOG_FILES_BY_PLATFORM[Platform.WINDOWS]

        files_missing = [
            ft for ft in expected_files if ft not in files_present
        ]

        manifest = None
        if LogFileType.MANIFEST in session_data["files"]:
            try:
                manifest_key = session_data["keys"][LogFileType.MANIFEST]
                manifest_data = deps.object_store.get_json(manifest_key)
                manifest = parse_manifest(manifest_data)

                # Only validate manifest file list for Windows sessions
                if platform == Platform.WINDOWS:
                    declared_log_files = set()
                    for f in manifest.declared_files:
                        if f in FILE_TYPE_MAP:
                            declared_log_files.add(FILE_TYPE_MAP[f])

                    actual_logs = session_data["files"] - {LogFileType.MANIFEST}
                    if declared_log_files != actual_logs:
                        deps.logger.warning(
                            "Manifest file list mismatch",
                            session=session_key,
                            declared=list(manifest.declared_files),
                            found=[ft.value for ft in actual_logs],
                        )
            except Exception as e:
                deps.logger.warning(
                    "Failed to parse manifest",
                    session=session_key,
                    error=str(e),
                )

        session_ref = SebLogSessionRef(
            org_assessment_id=session_data.get("org_assessment_id", ""),
            user_id=session_data.get("user_id", ""),
            session_folder=session_data.get("session_folder", ""),
            s3_prefix=session_data.get("s3_prefix", ""),
            files_present=files_present,
            files_missing=files_missing,
            manifest=manifest,
            platform=platform,
            file_keys=file_keys,
        )

        # Log caveats for later use (they'll be added during reduction)
        if caveats:
            deps.logger.info(
                "Session has discovery caveats",
                session=session_key,
                caveats=caveats,
            )

        yield session_ref


def get_log_file_key(session_ref: SebLogSessionRef, file_type: LogFileType) -> str:
    """Get the S3 key for a specific log file in a session.

    For macOS sessions with dynamic filenames, reads from file_keys.
    For Windows sessions, falls back to static filename mapping.
    """
    # First check file_keys (required for macOS, optional for Windows)
    if session_ref.file_keys and file_type.value in session_ref.file_keys:
        return session_ref.file_keys[file_type.value]

    # Fallback to static mapping for Windows
    filename_map = {
        LogFileType.RUNTIME: "Runtime.log",
        LogFileType.CLIENT: "Client.log",
        LogFileType.BROWSER: "Browser.log",
        LogFileType.SERVICE: "Service.log",
        LogFileType.MANIFEST: "manifest.json",
    }
    filename = filename_map.get(file_type)
    if not filename:
        raise ValueError(f"Unknown file type: {file_type}")

    return f"{session_ref.s3_prefix}{filename}"
