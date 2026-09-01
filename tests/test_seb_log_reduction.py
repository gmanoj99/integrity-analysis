"""Tests for SEB log reduction pipeline."""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from integrity_review_pipeline.seb_logs.contracts import (
    LogFileType,
    NoiseCategory,
    ReducedSessionEvidence,
    SebLogSessionRef,
    SessionManifest,
    SignalCategory,
)
from integrity_review_pipeline.seb_logs.deps import SebLogPipelineDeps
from integrity_review_pipeline.seb_logs.discovery import parse_manifest
from integrity_review_pipeline.seb_logs.grammars.cef_browser import CefBrowserGrammar
from integrity_review_pipeline.seb_logs.grammars.windows_dotnet import WindowsDotNetGrammar
from integrity_review_pipeline.seb_logs.io import write_reduced_evidence
from integrity_review_pipeline.seb_logs.reduction.bulk_event_aggregator import (
    BulkEventAggregator,
)
from integrity_review_pipeline.seb_logs.reduction.noise_rules import NoiseDetector
from integrity_review_pipeline.seb_logs.reduction.session_reducer import SessionReducer
from integrity_review_pipeline.seb_logs.reduction.signal_rules import SignalDetector


FIXTURES_DIR = Path(__file__).parent / "fixtures" / "seb_logs"


class NoopLogger:
    """Logger that does nothing."""

    def debug(self, message: str, **fields: Any) -> None:
        pass

    def info(self, message: str, **fields: Any) -> None:
        pass

    def warning(self, message: str, **fields: Any) -> None:
        pass

    def error(self, message: str, **fields: Any) -> None:
        pass


class LocalTestStore:
    """Local file store for testing."""

    def __init__(self, base_path: Path) -> None:
        self._base_path = base_path

    def list_keys(self, prefix: str) -> Iterator[str]:
        search_path = self._base_path / prefix.lstrip("/")
        if not search_path.exists():
            return

        for path in search_path.rglob("*"):
            if path.is_file():
                yield str(path.relative_to(self._base_path))

    def iter_lines(self, key: str) -> Iterator[str]:
        full_path = self._base_path / key.lstrip("/")
        with open(full_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                yield line.rstrip("\n\r")

    def get_json(self, key: str) -> dict[str, Any]:
        full_path = self._base_path / key.lstrip("/")
        with open(full_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_object_size(self, key: str) -> int:
        full_path = self._base_path / key.lstrip("/")
        return full_path.stat().st_size


class TestWindowsDotNetGrammar:
    """Tests for Windows .NET log grammar."""

    def test_parse_info_line(self) -> None:
        grammar = WindowsDotNetGrammar()
        line = "2026-08-16 14:53:45.442 [12] - INFO: Initiating startup procedure..."
        entry = grammar.parse_line(line, 1)

        assert entry is not None
        assert entry.timestamp == datetime(2026, 8, 16, 14, 53, 45, 442000)
        assert entry.thread == "12"
        assert entry.severity == "info"
        assert entry.module is None
        assert "Initiating startup procedure" in entry.message

    def test_parse_line_with_module(self) -> None:
        grammar = WindowsDotNetGrammar()
        line = "2026-08-16 14:53:45.571 [12] - DEBUG: [Text] Data successfully loaded."
        entry = grammar.parse_line(line, 1)

        assert entry is not None
        assert entry.module == "Text"
        assert entry.severity == "debug"

    def test_parse_warning_line(self) -> None:
        grammar = WindowsDotNetGrammar()
        line = "2026-08-16 14:53:45.811 [12] - WARNING: [IntegrityModule] Integrity module is not available!"
        entry = grammar.parse_line(line, 1)

        assert entry is not None
        assert entry.severity == "warning"
        assert entry.module == "IntegrityModule"

    def test_skip_header_comment(self) -> None:
        grammar = WindowsDotNetGrammar()
        line = "/* Topin Secure Browser, Version 1.0.0.0, Build 1.0.0.0"
        entry = grammar.parse_line(line, 1)
        assert entry is None

    def test_skip_hash_comment(self) -> None:
        grammar = WindowsDotNetGrammar()
        line = "# Application started at 2026-08-16 14:53:43.512"
        entry = grammar.parse_line(line, 1)
        assert entry is None

    def test_parse_header(self) -> None:
        grammar = WindowsDotNetGrammar()
        header_lines = [
            "/* Topin Secure Browser, Version 1.0.0.0, Build 1.0.0.0",
            "/* Copyright © 2010-2025 ETH Zürich, IT Services. Modifications © 2026 NxtWave",
            "/* ",
            "# Application started at 2026-08-16 14:53:43.512",
            "# Running on Windows 11, Microsoft Windows NT 10.0.26200.0 (x64)",
            "# Computer 'DESKTOP-KV5VBHO' is a Inspiron Inspiron 15 3511 manufactured by Dell Inc.",
            "# Runtime-ID: 544d3913-e3ab-4e50-a024-8d4129c8da48",
        ]
        metadata = grammar.parse_header(header_lines)

        assert metadata["program_version"] == "1.0.0.0"
        assert metadata["program_build"] == "1.0.0.0"
        assert metadata["os_name"] == "Windows 11"
        assert metadata["machine_name"] == "DESKTOP-KV5VBHO"
        assert metadata["runtime_id"] == "544d3913-e3ab-4e50-a024-8d4129c8da48"


class TestCefBrowserGrammar:
    """Tests for CEF/Chromium browser log grammar."""

    def test_parse_error_line(self) -> None:
        grammar = CefBrowserGrammar(reference_year=2026)
        line = "[12708:16684:0816/145404.202:ERROR:chrome\\installer\\util\\google_update_settings.cc:265] Failed opening key"
        entry = grammar.parse_line(line, 1)

        assert entry is not None
        assert entry.timestamp == datetime(2026, 8, 16, 14, 54, 4, 202000)
        assert entry.thread == "12708:16684"
        assert entry.severity == "error"

    def test_parse_warning_line(self) -> None:
        grammar = CefBrowserGrammar(reference_year=2026)
        line = "[12708:11180:0816/145404.290:WARNING:chrome\\browser\\signin\\account_consistency_mode_manager.cc:74] Desktop Identity"
        entry = grammar.parse_line(line, 1)

        assert entry is not None
        assert entry.severity == "warning"

    def test_parse_console_log(self) -> None:
        grammar = CefBrowserGrammar(reference_year=2026)
        line = '[12708:11180:0816/145406.920:INFO:CONSOLE:45] "MathJax is loaded", source: https://example.com (45)'
        entry = grammar.parse_line(line, 1)

        assert entry is not None
        assert entry.module == "CONSOLE"


class TestNoiseDetector:
    """Tests for noise detection."""

    def test_detect_ping_noise(self) -> None:
        detector = NoiseDetector()
        grammar = WindowsDotNetGrammar()

        line = "2026-08-16 14:53:45.442 [12] - DEBUG: [RuntimeHost] Sending message 'Ping'..."
        entry = grammar.parse_line(line, 1)
        match = detector.check_entry(entry, LogFileType.RUNTIME)

        assert match is not None
        assert match.category == NoiseCategory.PING_KEEPALIVE

    def test_detect_config_monitor_noise(self) -> None:
        detector = NoiseDetector()
        grammar = WindowsDotNetGrammar()

        line = "2026-08-16 14:53:50.257 [15] - DEBUG: [FeatureConfigurationMonitor] Checking 13 configurations..."
        entry = grammar.parse_line(line, 1)
        match = detector.check_entry(entry, LogFileType.SERVICE)

        assert match is not None
        assert match.category == NoiseCategory.CONFIG_MONITOR_CHECK

    def test_accumulate_summaries(self) -> None:
        detector = NoiseDetector()
        grammar = WindowsDotNetGrammar()

        for i in range(10):
            line = f"2026-08-16 14:53:{45 + i}.000 [12] - DEBUG: [RuntimeHost] Sending message 'Ping'..."
            entry = grammar.parse_line(line, i + 1)
            detector.check_entry(entry, LogFileType.RUNTIME)

        summaries = detector.get_summaries()
        assert len(summaries) == 1
        assert summaries[0].category == NoiseCategory.PING_KEEPALIVE
        assert summaries[0].count == 10


class TestBulkEventAggregator:
    """Tests for bulk termination event aggregation."""

    def test_aggregate_termination_events(self) -> None:
        aggregator = BulkEventAggregator()
        grammar = WindowsDotNetGrammar()

        lines = [
            "2026-08-16 14:53:02.558 [01] - DEBUG: [ApplicationMonitor] Process 'chrome.exe' (23916) belongs to application 'chrome.exe' and needs to be terminated.",
            "2026-08-16 14:53:07.197 [01] - DEBUG: [Process 'chrome.exe' (23916)] Attempting to close process...",
            "2026-08-16 14:53:07.204 [01] - WARNING: [Process 'chrome.exe' (23916)] Failed to send close message to main window!",
            "2026-08-16 14:53:07.213 [01] - DEBUG: [Process 'chrome.exe' (23916)] Attempting to kill process...",
            "2026-08-16 14:53:07.238 [01] - INFO: [ApplicationMonitor] Successfully terminated process 'chrome.exe' (23916).",
        ]

        for i, line in enumerate(lines):
            entry = grammar.parse_line(line, i + 1)
            aggregator.check_entry(entry, LogFileType.CLIENT)

        events = aggregator.get_bulk_events()
        assert len(events) == 1
        assert events[0].application_name == "chrome.exe"
        assert events[0].instance_count == 1
        assert events[0].close_message_failures == 1
        assert events[0].force_kill_needed is True
        assert events[0].all_terminated is True


class TestSignalDetector:
    """Tests for signal detection."""

    def test_detect_integrity_warning(self) -> None:
        detector = SignalDetector()
        grammar = WindowsDotNetGrammar()

        line = "2026-08-16 14:53:45.811 [12] - WARNING: [IntegrityModule] Integrity module is not available!"
        entry = grammar.parse_line(line, 1)
        match = detector.check_entry(entry, LogFileType.RUNTIME)

        assert match is not None
        assert match.category == SignalCategory.COVERAGE_GAP
        assert match.title == "Integrity module unavailable"

    def test_detect_os_lockdown_drift(self) -> None:
        detector = SignalDetector()
        grammar = WindowsDotNetGrammar()

        line = "2026-08-16 14:54:35.561 [15] - WARNING: [FeatureConfigurationMonitor] WindowsUpdateConfiguration (b42cf635-76d1-44e4-8ec7-8442e25c843a) is enabled instead of disabled!"
        entry = grammar.parse_line(line, 1)
        match = detector.check_entry(entry, LogFileType.SERVICE)

        assert match is not None
        assert match.category == SignalCategory.CONFIG_DEVIATION
        assert match.title == "OS lockdown drift"

    def test_accumulate_signals(self) -> None:
        detector = SignalDetector()
        grammar = WindowsDotNetGrammar()

        for i in range(3):
            line = f"2026-08-16 14:54:{35 + i}.000 [15] - WARNING: [FeatureConfigurationMonitor] WindowsUpdateConfiguration is enabled instead of disabled!"
            entry = grammar.parse_line(line, i + 1)
            detector.check_entry(entry, LogFileType.SERVICE)

        signals = detector.get_signals()
        assert len(signals) == 1
        assert signals[0].occurrence_count == 3


class TestParseManifest:
    """Tests for manifest parsing."""

    def test_parse_manifest(self) -> None:
        data = {
            "launchId": "2026-08-16_14h53m43s_4a778719",
            "environment": "prod",
            "orgAssessmentId": "18b5742d-2926-4def-9db3-5560f457845b",
            "userId": "4a719bfa-15b3-4412-bf90-ff661c1f38f6",
            "attemptId": None,
            "appVersion": "1.0.0.0",
            "buildVersion": "1.0.0.0",
            "osVersion": "Microsoft Windows NT 10.0.26200.0",
            "machineName": "DESKTOP-KV5VBHO",
            "startTime": "2026-08-16T09:23:43.5123315Z",
            "files": ["Runtime.log", "Client.log", "Browser.log", "Service.log"],
        }

        manifest = parse_manifest(data)

        assert manifest.launch_id == "2026-08-16_14h53m43s_4a778719"
        assert manifest.environment == "prod"
        assert manifest.org_assessment_id == "18b5742d-2926-4def-9db3-5560f457845b"
        assert manifest.user_id == "4a719bfa-15b3-4412-bf90-ff661c1f38f6"
        assert manifest.start_time is not None
        assert len(manifest.declared_files) == 4


class TestSessionReducer:
    """Tests for session reducer."""

    def test_reduce_session_with_fixtures(self) -> None:
        if not (FIXTURES_DIR / "session1").exists():
            pytest.skip("Test fixtures not available")

        store = LocalTestStore(FIXTURES_DIR)
        deps = SebLogPipelineDeps(object_store=store, logger=NoopLogger())

        session_ref = SebLogSessionRef(
            org_assessment_id="18b5742d-2926-4def-9db3-5560f457845b",
            user_id="4a719bfa-15b3-4412-bf90-ff661c1f38f6",
            session_folder="session1",
            s3_prefix="session1/",
            files_present=[
                LogFileType.RUNTIME,
                LogFileType.SERVICE,
                LogFileType.BROWSER,
                LogFileType.CLIENT,
                LogFileType.MANIFEST,
            ],
            files_missing=[],
            manifest=SessionManifest(
                launch_id="2026-08-16_14h53m43s_4a778719",
                org_assessment_id="18b5742d-2926-4def-9db3-5560f457845b",
                user_id="4a719bfa-15b3-4412-bf90-ff661c1f38f6",
                start_time=datetime(2026, 8, 16, 9, 23, 43),
                declared_files=["Runtime.log", "Client.log", "Browser.log", "Service.log"],
            ),
        )

        reducer = SessionReducer(deps)
        evidence = reducer.reduce_session(session_ref)

        assert evidence is not None
        assert evidence.session_ref == session_ref
        assert len(evidence.analysed_files) == 4
        assert evidence.reduction_stats["raw_bytes"] > 0


class TestWriteReducedEvidence:
    """Tests for evidence writing."""

    def test_write_evidence_to_file(self, tmp_path: Path) -> None:
        evidence = ReducedSessionEvidence(
            session_ref=SebLogSessionRef(
                org_assessment_id="test-org",
                user_id="test-user",
                session_folder="2026-08-16_14h53m43s_test1234",
                s3_prefix="test/",
                files_present=[LogFileType.RUNTIME],
                files_missing=[LogFileType.CLIENT, LogFileType.BROWSER, LogFileType.SERVICE],
            ),
            analysed_files=[],
            signals=[],
            noise_summary=[],
        )

        output_path = write_reduced_evidence(evidence, tmp_path)

        assert output_path.exists()
        assert output_path.name == "2026-08-16_14h53m43s_test1234.json"

        with open(output_path, encoding="utf-8") as f:
            data = json.load(f)

        assert "sessionRef" in data
        assert data["sessionRef"]["orgAssessmentId"] == "test-org"
