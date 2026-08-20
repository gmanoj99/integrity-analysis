"""Build the canonical assessment timeline from activity logs."""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from typing import Any

from ..contracts.review import SectionInput
from ..contracts.timeline import ActivityAnchor, CanonicalTimeline, SessionSection

IST_OFFSET_MS = int(timedelta(hours=5, minutes=30).total_seconds() * 1000)
IST_OFFSET = timedelta(hours=5, minutes=30)
SECTION_END_TYPES = {
    "SECTION_TERMINATED",
    "SECTION_COMPLETED",
    "SECTION_QUIT_BY_USER",
    "SECTION_TIMEOUT",
}


def parse_activity_log_epoch_ms(creation_datetime: str) -> int:
    """Topin activity logs store IST-naive timestamps — match TS parseActivityLogEpochMs."""
    normalized = creation_datetime.strip().replace(" ", "T")
    if not normalized.endswith(("Z", "z")) and not re.search(
        r"[+-]\d{2}:?\d{2}$", normalized
    ):
        normalized += "Z"
    parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    return int(parsed.timestamp() * 1000) - IST_OFFSET_MS


def parse_submission_time_epoch_ms(submission_time: str) -> int:
    trimmed = submission_time.strip()
    if not trimmed:
        raise ValueError("submission time cannot be empty")
    has_tz = trimmed.endswith(("Z", "z")) or bool(
        re.search(r"[+-]\d{2}:?\d{2}$", trimmed)
    )
    if has_tz:
        normalized = trimmed.replace(" ", "T") if " " in trimmed else trimmed
        return int(datetime.fromisoformat(normalized.replace("Z", "+00:00")).timestamp() * 1000)
    normalized = re.sub(r"\.\d+$", "", trimmed.replace("T", " "))
    return parse_activity_log_epoch_ms(normalized)


def parse_activity_epoch_ms(value: str | int | float) -> int:
    if isinstance(value, (int, float)):
        return int(value)
    text = value.strip()
    if not text:
        raise ValueError("activity timestamp cannot be empty")
    if text.endswith(("Z", "z")) or re.search(r"[+-]\d{2}:?\d{2}$", text):
        normalized = text.replace(" ", "T")
        return int(datetime.fromisoformat(normalized.replace("Z", "+00:00")).timestamp() * 1000)
    return parse_activity_log_epoch_ms(text)


def _value(log: dict[str, Any], *names: str, default: Any = None) -> Any:
    return next((log[name] for name in names if name in log), default)


def _metadata(log: dict[str, Any]) -> dict[str, Any]:
    raw = _value(log, "metadata", "metadata_str", "metadataStr", default={})
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def build_canonical_timeline(
    activity_logs: list[dict[str, Any]], section_details: list[SectionInput]
) -> CanonicalTimeline:
    if not activity_logs:
        return CanonicalTimeline(
            T0=0,
            total_duration_ms=0,
            anchors=[],
            sections=[],
            source="fallback",
        )
    sorted_logs = sorted(
        activity_logs,
        key=lambda log: int(_value(log, "order", "sequence", default=0)),
    )

    def event_type(log: dict[str, Any]) -> str:
        return str(
            _value(log, "activity_type_enum", "activityTypeEnum", "eventType", "type")
        )

    def epoch(log: dict[str, Any]) -> int:
        timestamp = _value(
            log,
            "timestamp",
            "epochMs",
            "creation_datetime",
            "creationDatetime",
        )
        if timestamp is None:
            raise ValueError("activity event has no timestamp")
        return parse_activity_epoch_ms(timestamp)

    start = next(
        (log for log in sorted_logs if event_type(log) == "ASSESSMENT_STARTED"),
        sorted_logs[0],
    )
    t0 = epoch(start)
    anchors: list[ActivityAnchor] = []
    for index, log in enumerate(sorted_logs):
        metadata = _metadata(log)
        epoch_ms = epoch(log)
        anchors.append(
            ActivityAnchor(
                order=int(_value(log, "order", "sequence", default=index)),
                type=event_type(log),
                epoch_ms=epoch_ms,
                session_offset_ms=epoch_ms - t0,
                section_id=_value(
                    metadata, "section_id", "sectionId", default=None
                ),
                end_reason=_value(
                    metadata, "exam_end_reason", "examEndReason", default=None
                ),
            )
        )

    starts = [
        anchor
        for anchor in anchors
        if anchor.type == "SECTION_STARTED" and anchor.section_id
    ]
    ends = [
        anchor
        for anchor in anchors
        if anchor.type in SECTION_END_TYPES and anchor.section_id
    ]
    sections: list[SessionSection] = []
    for index, section_start in enumerate(starts):
        closing = next(
            (
                item
                for item in ends
                if item.section_id == section_start.section_id
                and item.epoch_ms >= section_start.epoch_ms
            ),
            None,
        )
        detail = section_details[index] if index < len(section_details) else None
        end_ms = (
            closing.session_offset_ms
            if closing is not None
            else section_start.session_offset_ms
        )
        sections.append(
            SessionSection(
                section_id=section_start.section_id or f"section-{index + 1}",
                label=(detail.title if detail and detail.title else f"Section {index + 1}"),
                section_type=detail.section_type if detail else None,
                exam_attempt_id=detail.exam_attempt_id if detail else None,
                exam_id=detail.exam_id if detail else None,
                start_ms=section_start.session_offset_ms,
                end_ms=end_ms,
            )
        )

    completion = next(
        (anchor for anchor in anchors if anchor.type == "ALL_SECTIONS_COMPLETED"),
        None,
    )
    duration = (
        completion.session_offset_ms
        if completion
        else sections[-1].end_ms
        if sections
        else anchors[-1].session_offset_ms
    )
    return CanonicalTimeline(
        T0=t0,
        total_duration_ms=max(0, duration),
        anchors=anchors,
        sections=sections,
        source="activity_logs",
    )
