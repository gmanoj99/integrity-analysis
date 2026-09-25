from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from .contracts import MachineFact
from .kinds import MachineFactKind
from .rrweb_normalizer import normalize_rrweb_event_kind

ACTIVITY_GAP_THRESHOLD_MS = 10_000
CHUNK_WINDOW_MS = 15_000

RRWEB_TYPE_INCREMENTAL = 3
RRWEB_TYPE_CUSTOM = 5
RRWEB_SOURCE_MUTATION = 0
RRWEB_SOURCE_INPUT = 5
RRWEB_SOURCE_STYLESHEET_RULE = 8
RRWEB_SOURCE_MOUSE_INTERACT = 2
MOUSE_CONTEXT_MENU = 3


def event_timestamp_ms(event: dict[str, Any]) -> int | None:
    try:
        return int(event["timestamp"])
    except (KeyError, TypeError, ValueError):
        return None


@dataclass(frozen=True, slots=True)
class ActivityEvent:
    offset_ms: int
    kind: str
    target_id: str
    value: str


def reconstruct_activity_events(
    events: list[dict[str, Any]],
) -> tuple[list[ActivityEvent], int]:
    activity: list[ActivityEvent] = []
    filtered_noise = 0
    for event in events:
        if event.get("type") != RRWEB_TYPE_INCREMENTAL:
            continue
        data = event.get("data") or {}
        source = data.get("source")
        if source == RRWEB_SOURCE_STYLESHEET_RULE:
            filtered_noise += 1
            continue
        timestamp = event_timestamp_ms(event)
        if timestamp is None:
            continue
        if source == RRWEB_SOURCE_MUTATION:
            for text in data.get("texts") or []:
                activity.append(
                    ActivityEvent(
                        offset_ms=timestamp,
                        kind="mutation_text",
                        target_id=str(text.get("id", "")),
                        value=str(text.get("value", "")),
                    )
                )
        elif source == RRWEB_SOURCE_INPUT:
            element_id = data.get("id")
            if element_id is not None:
                activity.append(
                    ActivityEvent(
                        offset_ms=timestamp,
                        kind="input",
                        target_id=str(element_id),
                        value=str(data.get("text", "")),
                    )
                )
    activity.sort(key=lambda item: item.offset_ms)
    return activity, filtered_noise


def _extract_custom_events(events: list[dict[str, Any]]) -> list[MachineFact]:
    facts: list[MachineFact] = []
    for event in events:
        if event.get("type") != RRWEB_TYPE_CUSTOM:
            continue
        timestamp = event_timestamp_ms(event)
        if timestamp is None:
            continue
        data = event.get("data") or {}
        payload = data.get("payload") or {}
        inner = payload.get("data") if isinstance(payload, dict) else {}
        raw_event_tag = (
            inner.get("data")
            if isinstance(inner, dict) and inner.get("data")
            else data.get("tag", "unknown_custom_event")
        )
        message = inner.get("message") if isinstance(inner, dict) else None
        snapshot = inner.get("image") if isinstance(inner, dict) else None
        kind = normalize_rrweb_event_kind(str(raw_event_tag))
        detail: dict[str, Any] = {"rawEventTag": raw_event_tag}
        if message is not None:
            detail["message"] = message
        if snapshot is not None:
            detail["snapshot"] = snapshot
        facts.append(
            MachineFact(
                id=str(uuid.uuid4()),
                start_offset_ms=0,
                end_offset_ms=0,
                raw_timestamp_ms=timestamp,
                kind=kind.value,
                source="rrweb_deterministic",
                evidence_source="keystroke",
                attribution="unclear",
                confidence=1.0,
                detail=detail,
            )
        )
    return facts


def analyze_keystroke_chunk(
    events: list[dict[str, Any]], *, sequence: int = 0
) -> list[MachineFact]:
    del sequence
    facts = _extract_custom_events(events)
    activity, _ = reconstruct_activity_events(events)
    for index in range(1, len(activity)):
        prev, nxt = activity[index - 1], activity[index]
        gap_ms = nxt.offset_ms - prev.offset_ms
        if gap_ms < ACTIVITY_GAP_THRESHOLD_MS:
            continue
        facts.append(
            MachineFact(
                id=str(uuid.uuid4()),
                start_offset_ms=0,
                end_offset_ms=0,
                raw_timestamp_ms=nxt.offset_ms,
                kind=MachineFactKind.ACTIVITY_GAP.value,
                source="rrweb_deterministic",
                evidence_source="keystroke",
                attribution="unclear",
                confidence=1.0,
                detail={
                    "gapMs": gap_ms,
                    "before": {"kind": prev.kind, "targetId": prev.target_id},
                    "after": {"kind": nxt.kind, "targetId": nxt.target_id},
                },
            )
        )
    return sorted(facts, key=lambda fact: fact.raw_timestamp_ms or 0)
