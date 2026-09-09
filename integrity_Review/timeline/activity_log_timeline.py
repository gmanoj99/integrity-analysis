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


def _order_key(log: dict[str, Any]) -> int:
    """Sort by the backend's order, tolerating missing or non-numeric values."""

    try:
        return int(_value(log, "order", "sequence", default=0) or 0)
    except (TypeError, ValueError):
        return 0


def _anchor_order(log: dict[str, Any], fallback: int) -> int:
    """``order`` can be absent or explicitly null on a backend log row."""

    try:
        raw = _value(log, "order", "sequence", default=None)
        return fallback if raw is None else int(raw)
    except (TypeError, ValueError):
        return fallback


def _safe_epoch_ms(value: str | int | float | None) -> int | None:
    """Parse a timestamp, or return ``None`` when it is absent or malformed."""

    if value is None:
        return None
    try:
        return parse_activity_epoch_ms(value)
    except (ValueError, TypeError, OverflowError):
        return None


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


def _sections_from_spec(
    section_details: list[SectionInput], t0: int, anchors: list[ActivityAnchor]
) -> list[SessionSection]:
    """Build sections straight from the backend's ``sections[]`` datetimes.

    The backend does not always emit ``SECTION_STARTED``/``SECTION_*`` anchors in
    the activity log, so this is the primary source when it is available.
    """

    dated = [detail for detail in section_details if detail.start_datetime]
    if not dated:
        return []
    ordered = sorted(dated, key=lambda item: item.order if item.order is not None else 0)
    last_anchor_offset = anchors[-1].session_offset_ms if anchors else 0

    sections: list[SessionSection] = []
    for index, detail in enumerate(ordered):
        start_epoch_ms = _safe_epoch_ms(detail.start_datetime)
        if start_epoch_ms is None:
            # One section with an unusable timestamp must not cost the whole
            # review; it simply contributes no span to the timeline.
            continue
        start_ms = max(0, start_epoch_ms - t0)
        end_epoch_ms = _safe_epoch_ms(detail.end_datetime)
        next_start_epoch_ms = (
            _safe_epoch_ms(ordered[index + 1].start_datetime)
            if index + 1 < len(ordered)
            else None
        )
        if end_epoch_ms is not None:
            end_ms = max(start_ms, end_epoch_ms - t0)
        elif next_start_epoch_ms is not None:
            end_ms = max(start_ms, next_start_epoch_ms - t0)
        else:
            end_ms = max(start_ms, last_anchor_offset)
        sections.append(
            SessionSection(
                section_id=detail.section_id,
                label=detail.title or detail.section_type or detail.section_id,
                section_type=detail.section_type,
                exam_attempt_id=detail.exam_attempt_id,
                exam_id=detail.exam_id,
                start_ms=start_ms,
                end_ms=end_ms,
            )
        )
    return sections


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
    sorted_logs = sorted(activity_logs, key=_order_key)

    def event_type(log: dict[str, Any]) -> str:
        return str(
            _value(
                log,
                "activity_type_enum",
                "activityTypeEnum",
                "activity_type",
                "eventType",
                "type",
            )
        )

    def epoch(log: dict[str, Any]) -> int | None:
        return _safe_epoch_ms(
            _value(
                log,
                "timestamp",
                "epochMs",
                "creation_datetime",
                "creationDatetime",
            )
        )

    # A log row the backend could not timestamp cannot be placed on the
    # timeline, but the rest of the session is still perfectly analysable.
    timed_logs = [log for log in sorted_logs if epoch(log) is not None]
    if not timed_logs:
        return CanonicalTimeline(
            T0=0,
            total_duration_ms=0,
            anchors=[],
            sections=[],
            source="fallback",
        )

    start = next(
        (log for log in timed_logs if event_type(log) == "ASSESSMENT_STARTED"),
        timed_logs[0],
    )
    t0 = epoch(start) or 0
    anchors: list[ActivityAnchor] = []
    for index, log in enumerate(timed_logs):
        metadata = _metadata(log)
        epoch_ms = epoch(log)
        if epoch_ms is None:  # pragma: no cover - filtered above
            continue
        anchors.append(
            ActivityAnchor(
                order=_anchor_order(log, index),
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

    sections = _sections_from_spec(section_details, t0, anchors)
    if not sections:
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
