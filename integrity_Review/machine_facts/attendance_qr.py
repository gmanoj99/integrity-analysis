from __future__ import annotations

import json
import uuid
from typing import Any

from ..contracts.timeline import SessionSection
from .chunk_analysis import RRWEB_SOURCE_MUTATION, RRWEB_TYPE_INCREMENTAL, event_timestamp_ms
from .contracts import MachineFact
from .kinds import MachineFactKind
from .typing_extraction import resolve_section_id

RRWEB_TYPE_FULL_SNAPSHOT = 2
QR_ATTENDANCE_MARKERS = ("QR Attendance", "Show this QR to your invigilator")


def _mentions_qr(node: Any) -> bool:
    text = json.dumps(node, ensure_ascii=False)
    return any(marker in text for marker in QR_ATTENDANCE_MARKERS)


def _find_qr_node(node: dict[str, Any]) -> int | None:
    text = node.get("textContent")
    if isinstance(text, str) and any(marker in text for marker in QR_ATTENDANCE_MARKERS):
        node_id = node.get("id")
        return node_id if isinstance(node_id, int) else None
    for child in node.get("childNodes") or []:
        found = _find_qr_node(child)
        if found is not None:
            return found
    return None


def _register(node: dict[str, Any], parent_id: int | None, parents: dict[int, int | None]) -> None:
    node_id = node.get("id")
    if isinstance(node_id, int):
        parents[node_id] = parent_id
    for child in node.get("childNodes") or []:
        _register(child, node_id if isinstance(node_id, int) else None, parents)


def _ancestors(node_id: int, parents: dict[int, int | None]) -> set[int]:
    chain: set[int] = set()
    current: int | None = node_id
    while current is not None and current not in chain:
        chain.add(current)
        current = parents.get(current)
    return chain


def extract_qr_attendance_facts(
    *,
    chunks: list[tuple[str, int, list[dict[str, Any]]]],
    session_start_ms: int,
    sections: list[SessionSection],
) -> list[MachineFact]:

    events = sorted(
        (
            event
            for _, _, batch in sorted(chunks, key=lambda item: item[1])
            for event in batch
            if event_timestamp_ms(event) is not None
        ),
        key=lambda event: event_timestamp_ms(event) or 0,
    )
    parents: dict[int, int | None] = {}
    intervals: list[tuple[int, int]] = []
    open_at: int | None = None
    qr_node: int | None = None
    last_ms: int | None = None

    for event in events:
        ts = event_timestamp_ms(event) or 0
        last_ms = ts
        if event.get("type") == RRWEB_TYPE_FULL_SNAPSHOT:
            node = (event.get("data") or {}).get("node") or {}
            parents.clear()
            _register(node, None, parents)
            present = _mentions_qr(node)
            if present:
                open_at = ts if open_at is None else open_at
                qr_node = _find_qr_node(node)
            elif not present and open_at is not None:
                intervals.append((open_at, ts))
                open_at, qr_node = None, None
            continue
        data = event.get("data") or {}
        if event.get("type") != RRWEB_TYPE_INCREMENTAL or data.get("source") != RRWEB_SOURCE_MUTATION:
            continue
        for add in data.get("adds") or []:
            node = add.get("node") or {}
            _register(node, add.get("parentId"), parents)
            if open_at is None and _mentions_qr(node):
                open_at, qr_node = ts, node.get("id")
        if open_at is None or qr_node is None:
            continue
        chain = _ancestors(qr_node, parents)
        if any(remove.get("id") in chain for remove in data.get("removes") or []):
            intervals.append((open_at, ts))
            open_at, qr_node = None, None

    if open_at is not None and last_ms is not None:
        intervals.append((open_at, last_ms))

    facts: list[MachineFact] = []
    for start_raw, end_raw in intervals:
        start = max(0, start_raw - session_start_ms)
        end = max(start, end_raw - session_start_ms)
        detail: dict[str, Any] = {"durationMs": end - start}
        section_id = resolve_section_id(start, sections)
        if section_id:
            detail["sectionId"] = section_id
        facts.append(
            MachineFact(
                id=str(uuid.uuid4()),
                start_offset_ms=start,
                end_offset_ms=end,
                raw_timestamp_ms=start_raw,
                kind=MachineFactKind.QR_ATTENDANCE_SHOWN.value,
                source="rrweb_deterministic",
                evidence_source="keystroke",
                attribution="candidate",
                confidence=1.0,
                detail=detail,
            )
        )
    return facts
