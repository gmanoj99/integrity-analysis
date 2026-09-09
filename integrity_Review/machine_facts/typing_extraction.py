"""Extract typing / paste MachineFacts from decoded rrweb Input events."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Any

from ..contracts.timeline import SessionSection
from .chunk_analysis import (
    MOUSE_CONTEXT_MENU,
    RRWEB_SOURCE_INPUT,
    RRWEB_SOURCE_MOUSE_INTERACT,
    RRWEB_SOURCE_MUTATION,
    RRWEB_TYPE_INCREMENTAL,
    event_timestamp_ms,
)
from .contracts import MachineFact
from .kinds import MachineFactKind
from .paste_utils import compute_paste_excerpt

LARGE_PASTE_THRESHOLD = 50
CORRECTION_THRESHOLD = 10
AUTOFORMAT_WINDOW_MS = 500
MCQ_OPTION_UUID = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
EDITOR_CHUNK_REGEX = re.compile(
    r"^keystrokeData-(?:ide_based_coding|primitive_coding|html_coding|sql_coding|coding)-"
)
UI_CONSTANT_STRINGS = {"NO INPUT REQUIRED FOR THIS QUESTION"}
TEST_RESULT_FULL = re.compile(r"(\d+)\s*/\s*(\d+)\s+test\s+cases?\s+passed", re.IGNORECASE)
TEST_RESULT_ANCHOR = re.compile(r"test\s+cases?\s+(?:passed|failed|run)", re.IGNORECASE)
TEST_RESULT_COUNT_ONLY = re.compile(r"^(\d+)\s*/\s*(\d+)$")
TEST_RESULT_CORRELATION_MS = 3_000
MCQ_SWITCH_WINDOW_MS = 100


def resolve_section_id(
    session_offset_ms: int, sections: list[SessionSection]
) -> str | None:
    if not sections:
        return None
    for section in sections:
        if section.start_ms <= session_offset_ms <= section.end_ms:
            return section.section_id
    nearest = min(
        sections,
        key=lambda section: min(
            abs(session_offset_ms - section.start_ms),
            abs(session_offset_ms - section.end_ms),
        ),
    )
    return nearest.section_id


def _classify_input(text: str, is_checked: bool | None, section_type: str | None) -> str:
    if text == "":
        return "editor_snapshot"
    if is_checked is not None and MCQ_OPTION_UUID.match(text):
        return "mcq_radio"
    if text.strip() in UI_CONSTANT_STRINGS:
        return "ui_label"
    if section_type and (
        re.search(r"coding", section_type, re.IGNORECASE)
        or re.fullmatch(r"(?:SQL|SQL_QUERY|HTML|HTML_CSS)", section_type.strip(), re.IGNORECASE)
    ):
        if re.fullmatch(r"[\s\d\[\]{},.:'\"()\-]+", text):
            return "test_fixture"
    return "real_typing"


def _is_autoformat(prev: str, nxt: str) -> bool:
    skeleton = lambda value: re.sub(r"[\s{}()[\]<>;,'\"]", "", value)
    return skeleton(prev) == skeleton(nxt)


@dataclass
class _TypingState:
    first_timestamp: int
    first_offset_ms: int
    last_timestamp: int
    last_offset_ms: int
    peak_char_count: int
    final_char_count: int
    total_inserted_chars: int
    insertion_events: int
    deletion_events: int
    section_id: str | None
    section_type: str | None


@dataclass
class _McqState:
    current_option_id: str | None
    change_count: int
    first_offset_ms: int
    last_offset_ms: int
    section_id: str | None


@dataclass
class _PendingTestResult:
    section_id: str | None
    start_offset_ms: int
    raw_timestamp: int
    anchor_text: str


def _collect_rrweb_text_nodes(node: Any, output: list[str]) -> None:
    if not isinstance(node, dict):
        return
    if node.get("type") == 3 and isinstance(node.get("textContent"), str):
        text = node["textContent"].strip()
        if text:
            output.append(text)
    for child in node.get("childNodes") or []:
        _collect_rrweb_text_nodes(child, output)


def _insertion_locality(prev_text: str, new_text: str) -> str:
    return "append_at_end" if new_text.startswith(prev_text) else "mid_document_replace"


def extract_rrweb_typing_facts(
    *,
    chunks: list[tuple[str, int, list[dict[str, Any]]]],
    session_start_ms: int,
    sections: list[SessionSection],
) -> tuple[list[MachineFact], bool]:
    facts: list[MachineFact] = []
    had_monaco = any(EDITOR_CHUNK_REGEX.match(chunk_id) for chunk_id, _, _ in chunks)
    element_last_text: dict[str, str] = {}
    element_state: dict[str, _TypingState] = {}
    element_last_edit: dict[str, int] = {}
    mcq_state: dict[str, _McqState] = {}
    mcq_element_to_group: dict[str, str] = {}
    recently_deselected: dict[str, tuple[int, str | None]] = {}
    recently_selected: dict[str, tuple[int, str | None]] = {}
    pending_test_results: dict[str, _PendingTestResult] = {}

    # A duplicate sequence means the same chunk was listed twice; replaying it
    # would double-count keystrokes, so the repeat is skipped rather than
    # failing a review over a manifest artefact.
    ordered = []
    seen_sequences: set[int] = set()
    for chunk in sorted(chunks, key=lambda item: item[1]):
        if chunk[1] in seen_sequences:
            continue
        seen_sequences.add(chunk[1])
        ordered.append(chunk)

    for _, _, events in ordered:
        for event in events:
            if event.get("type") != RRWEB_TYPE_INCREMENTAL:
                continue
            data = event.get("data") or {}
            source = data.get("source")
            # Client-produced events are not guaranteed well-formed; one event
            # without a usable timestamp cannot be placed on the timeline.
            timestamp = event_timestamp_ms(event)
            if timestamp is None:
                continue
            session_offset_ms = max(0, timestamp - session_start_ms)
            section_id = resolve_section_id(session_offset_ms, sections)
            section_type = next(
                (section.section_type for section in sections if section.section_id == section_id),
                None,
            )

            if source == RRWEB_SOURCE_MOUSE_INTERACT and data.get("type") == MOUSE_CONTEXT_MENU:
                facts.append(
                    MachineFact(
                        id=str(uuid.uuid4()),
                        start_offset_ms=session_offset_ms,
                        end_offset_ms=session_offset_ms,
                        raw_timestamp_ms=timestamp,
                        kind=MachineFactKind.RIGHT_CLICK.value,
                        source="rrweb_deterministic",
                        evidence_source="keystroke",
                        attribution="candidate",
                        confidence=1.0,
                        detail={
                            "x": data.get("x"),
                            "y": data.get("y"),
                            **({"sectionId": section_id} if section_id else {}),
                        },
                    )
                )
                continue

            if source == RRWEB_SOURCE_MUTATION:
                if not section_type or not re.search(
                    r"coding", section_type, re.IGNORECASE
                ):
                    continue
                mutation_texts = [
                    str(item.get("value", "")).strip()
                    for item in data.get("texts") or []
                    if str(item.get("value", "")).strip()
                ]
                for addition in data.get("adds") or []:
                    _collect_rrweb_text_nodes(addition.get("node"), mutation_texts)
                if not mutation_texts:
                    continue
                section_key = section_id or "unknown"
                full = next(
                    (
                        (match, text)
                        for text in mutation_texts
                        if (match := TEST_RESULT_FULL.search(text))
                    ),
                    None,
                )
                if full:
                    match, raw_text = full
                    facts.append(
                        MachineFact(
                            id=str(uuid.uuid4()),
                            start_offset_ms=session_offset_ms,
                            end_offset_ms=session_offset_ms,
                            raw_timestamp_ms=timestamp,
                            kind=MachineFactKind.TEST_RESULT_OBSERVED.value,
                            source="rrweb_deterministic",
                            evidence_source="keystroke",
                            attribution="candidate",
                            confidence=1.0,
                            detail={
                                "passedCount": int(match.group(1)),
                                "totalCount": int(match.group(2)),
                                "rawText": raw_text,
                                **({"sectionId": section_id} if section_id else {}),
                            },
                        )
                    )
                    pending_test_results.pop(section_key, None)
                    continue
                anchor = next(
                    (text for text in mutation_texts if TEST_RESULT_ANCHOR.search(text)),
                    None,
                )
                count = next(
                    (
                        (match, text)
                        for text in mutation_texts
                        if (match := TEST_RESULT_COUNT_ONLY.fullmatch(text))
                    ),
                    None,
                )
                if anchor and count:
                    match, raw_count = count
                    facts.append(
                        MachineFact(
                            id=str(uuid.uuid4()),
                            start_offset_ms=session_offset_ms,
                            end_offset_ms=session_offset_ms,
                            raw_timestamp_ms=timestamp,
                            kind=MachineFactKind.TEST_RESULT_OBSERVED.value,
                            source="rrweb_deterministic",
                            evidence_source="keystroke",
                            attribution="candidate",
                            confidence=1.0,
                            detail={
                                "passedCount": int(match.group(1)),
                                "totalCount": int(match.group(2)),
                                "rawText": f"{anchor} [count:{raw_count}]",
                                **({"sectionId": section_id} if section_id else {}),
                            },
                        )
                    )
                    pending_test_results.pop(section_key, None)
                elif anchor:
                    pending_test_results[section_key] = _PendingTestResult(
                        section_id, session_offset_ms, timestamp, anchor
                    )
                elif count:
                    pending = pending_test_results.get(section_key)
                    if (
                        pending
                        and timestamp - pending.raw_timestamp
                        <= TEST_RESULT_CORRELATION_MS
                    ):
                        match, raw_count = count
                        facts.append(
                            MachineFact(
                                id=str(uuid.uuid4()),
                                start_offset_ms=pending.start_offset_ms,
                                end_offset_ms=session_offset_ms,
                                raw_timestamp_ms=timestamp,
                                kind=MachineFactKind.TEST_RESULT_OBSERVED.value,
                                source="rrweb_deterministic",
                                evidence_source="keystroke",
                                attribution="candidate",
                                confidence=1.0,
                                detail={
                                    "passedCount": int(match.group(1)),
                                    "totalCount": int(match.group(2)),
                                    "rawText": (
                                        f"{pending.anchor_text} [count:{raw_count}]"
                                    ),
                                    **(
                                        {"sectionId": pending.section_id}
                                        if pending.section_id
                                        else {}
                                    ),
                                },
                            )
                        )
                        pending_test_results.pop(section_key, None)
                continue

            if source != RRWEB_SOURCE_INPUT:
                continue

            element_id = str(data.get("id", ""))
            if not element_id:
                continue
            text = str(data.get("text", ""))
            is_checked = data.get("isChecked")
            prev_text = element_last_text.get(element_id)
            if prev_text is None and text == "":
                continue

            role = _classify_input(text, is_checked, section_type)
            if role in {"editor_snapshot", "test_fixture", "ui_label"}:
                element_last_text[element_id] = text
                continue

            if role == "mcq_radio":
                if is_checked is False:
                    own_group = mcq_element_to_group.get(element_id)
                    for selected_id, (selected_offset, selected_section) in list(
                        recently_selected.items()
                    ):
                        if (
                            selected_section == section_id
                            and abs(session_offset_ms - selected_offset)
                            <= MCQ_SWITCH_WINDOW_MS
                        ):
                            selected_group = mcq_element_to_group.get(
                                selected_id, selected_id
                            )
                            own_group_key = own_group or element_id
                            if selected_group != own_group_key:
                                mcq_element_to_group[element_id] = selected_group
                                group = mcq_state.get(selected_group)
                                if group and group.current_option_id != text:
                                    group.change_count += 1
                            recently_selected.pop(selected_id, None)
                            break
                    recently_deselected[element_id] = (
                        session_offset_ms,
                        section_id,
                    )
                    element_last_text[element_id] = text
                    continue
                if is_checked is True:
                    recently_selected[element_id] = (session_offset_ms, section_id)
                    group_key = mcq_element_to_group.get(element_id)
                    if group_key is None:
                        linked_group: str | None = None
                        for deselected_id, (
                            deselected_offset,
                            deselected_section,
                        ) in list(recently_deselected.items()):
                            if (
                                deselected_section == section_id
                                and session_offset_ms - deselected_offset
                                <= MCQ_SWITCH_WINDOW_MS
                            ):
                                linked_group = mcq_element_to_group.get(
                                    deselected_id, deselected_id
                                )
                                recently_deselected.pop(deselected_id, None)
                                break
                        group_key = linked_group or element_id
                        mcq_element_to_group[element_id] = group_key
                    existing = mcq_state.get(group_key)
                    if existing is None:
                        mcq_state[group_key] = _McqState(
                            text, 0, session_offset_ms, session_offset_ms, section_id
                        )
                    else:
                        existing.last_offset_ms = session_offset_ms
                        if text != existing.current_option_id:
                            existing.current_option_id = text
                            existing.change_count += 1
                    facts.append(
                        MachineFact(
                            id=str(uuid.uuid4()),
                            start_offset_ms=session_offset_ms,
                            end_offset_ms=session_offset_ms,
                            raw_timestamp_ms=timestamp,
                            kind=MachineFactKind.MCQ_ANSWER_SELECTED.value,
                            source="rrweb_deterministic",
                            evidence_source="keystroke",
                            attribution="candidate",
                            confidence=1.0,
                            detail={
                                "elementId": element_id,
                                "optionId": text,
                                **({"sectionId": section_id} if section_id else {}),
                            },
                        )
                    )
                element_last_text[element_id] = text
                continue

            if prev_text is None and text:
                initial_typed = len(text) if len(text) < LARGE_PASTE_THRESHOLD else 0
                element_state[element_id] = _TypingState(
                    first_timestamp=timestamp,
                    first_offset_ms=session_offset_ms,
                    last_timestamp=timestamp,
                    last_offset_ms=session_offset_ms,
                    peak_char_count=len(text),
                    final_char_count=len(text),
                    total_inserted_chars=initial_typed,
                    insertion_events=1,
                    deletion_events=0,
                    section_id=section_id,
                    section_type=section_type,
                )
                element_last_edit[element_id] = timestamp
                facts.append(
                    MachineFact(
                        id=str(uuid.uuid4()),
                        start_offset_ms=session_offset_ms,
                        end_offset_ms=session_offset_ms,
                        raw_timestamp_ms=timestamp,
                        kind=MachineFactKind.TYPING_STARTED.value,
                        source="rrweb_deterministic",
                        evidence_source="keystroke",
                        attribution="candidate",
                        confidence=1.0,
                        detail={
                            "elementId": element_id,
                            "charCount": len(text),
                            **({"sectionId": section_id} if section_id else {}),
                        },
                    )
                )
            elif prev_text is not None:
                state = element_state.get(element_id)
                delta = len(text) - len(prev_text)
                if state and delta != 0:
                    ms_since = timestamp - element_last_edit.get(element_id, timestamp)
                    if (
                        abs(delta) < LARGE_PASTE_THRESHOLD
                        and ms_since < AUTOFORMAT_WINDOW_MS
                        and _is_autoformat(prev_text, text)
                    ):
                        facts.append(
                            MachineFact(
                                id=str(uuid.uuid4()),
                                start_offset_ms=session_offset_ms,
                                end_offset_ms=session_offset_ms,
                                raw_timestamp_ms=timestamp,
                                kind=MachineFactKind.AUTOFORMAT.value,
                                source="rrweb_deterministic",
                                evidence_source="keystroke",
                                attribution="candidate",
                                confidence=1.0,
                                detail={
                                    "elementId": element_id,
                                    "deltaChars": delta,
                                    "msSincePriorEdit": ms_since,
                                    **({"sectionId": section_id} if section_id else {}),
                                },
                            )
                        )
                        element_last_text[element_id] = text
                        element_last_edit[element_id] = timestamp
                        continue
                    if timestamp - session_start_ms <= 4 * 60 * 60 * 1000:
                        state.last_timestamp = timestamp
                        state.last_offset_ms = session_offset_ms
                    state.final_char_count = len(text)
                    state.peak_char_count = max(state.peak_char_count, len(text))
                    if delta > 0:
                        state.insertion_events += 1
                        if delta < LARGE_PASTE_THRESHOLD:
                            state.total_inserted_chars += delta
                    if delta < 0:
                        state.deletion_events += 1

                if delta >= LARGE_PASTE_THRESHOLD:
                    excerpt = compute_paste_excerpt(prev_text, text)
                    paste_origin = (
                        "internal"
                        if excerpt
                        and len(excerpt) >= 20
                        and excerpt in "\n".join(element_last_text.values())
                        else "external"
                    )
                    facts.append(
                        MachineFact(
                            id=str(uuid.uuid4()),
                            start_offset_ms=session_offset_ms,
                            end_offset_ms=session_offset_ms,
                            raw_timestamp_ms=timestamp,
                            kind=MachineFactKind.LARGE_PASTE.value,
                            source="rrweb_deterministic",
                            evidence_source="keystroke",
                            attribution="candidate",
                            confidence=1.0,
                            detail={
                                "elementId": element_id,
                                "charsAdded": delta,
                                "totalChars": len(text),
                                "prevChars": len(prev_text),
                                "insertionLocality": _insertion_locality(prev_text, text),
                                "pasteOrigin": paste_origin,
                                **({"pastedExcerpt": excerpt} if excerpt else {}),
                                **({"sectionId": section_id} if section_id else {}),
                                **({"sectionType": section_type} if section_type else {}),
                                "elementRole": role,
                            },
                        )
                    )
                if -delta >= CORRECTION_THRESHOLD:
                    facts.append(
                        MachineFact(
                            id=str(uuid.uuid4()),
                            start_offset_ms=session_offset_ms,
                            end_offset_ms=session_offset_ms,
                            raw_timestamp_ms=timestamp,
                            kind=MachineFactKind.TEXT_CORRECTION.value,
                            source="rrweb_deterministic",
                            evidence_source="keystroke",
                            attribution="candidate",
                            confidence=1.0,
                            detail={
                                "elementId": element_id,
                                "charsDeleted": -delta,
                                "charsAfter": len(text),
                                "charsBefore": len(prev_text),
                                **({"sectionId": section_id} if section_id else {}),
                                **({"sectionType": section_type} if section_type else {}),
                                "elementRole": role,
                            },
                        )
                    )
                element_last_edit[element_id] = timestamp
            element_last_text[element_id] = text

    for pending in pending_test_results.values():
        facts.append(
            MachineFact(
                id=str(uuid.uuid4()),
                start_offset_ms=pending.start_offset_ms,
                end_offset_ms=pending.start_offset_ms,
                raw_timestamp_ms=session_start_ms + pending.start_offset_ms,
                kind=MachineFactKind.TEST_RESULT_OBSERVED.value,
                source="rrweb_deterministic",
                evidence_source="keystroke",
                attribution="candidate",
                confidence=1.0,
                detail={
                    "passedCount": "UNKNOWN",
                    "totalCount": "UNKNOWN",
                    "rawText": pending.anchor_text,
                    **({"sectionId": pending.section_id} if pending.section_id else {}),
                },
            )
        )

    for element_id, state in mcq_state.items():
        if state.change_count < 1:
            continue
        facts.append(
            MachineFact(
                id=str(uuid.uuid4()),
                start_offset_ms=state.last_offset_ms,
                end_offset_ms=state.last_offset_ms,
                raw_timestamp_ms=session_start_ms + state.last_offset_ms,
                kind=MachineFactKind.MCQ_ANSWER_CHANGED.value,
                source="rrweb_deterministic",
                evidence_source="keystroke",
                attribution="candidate",
                confidence=1.0,
                detail={
                    "elementId": element_id,
                    "selectionChangeCount": state.change_count,
                    **({"sectionId": state.section_id} if state.section_id else {}),
                },
            )
        )

    for element_id, state in element_state.items():
        duration_ms = state.last_timestamp - state.first_timestamp
        avg_cps = (
            round((state.total_inserted_chars / duration_ms) * 1000, 1)
            if duration_ms > 0
            else 0.0
        )
        facts.append(
            MachineFact(
                id=str(uuid.uuid4()),
                start_offset_ms=state.last_offset_ms,
                end_offset_ms=state.last_offset_ms,
                raw_timestamp_ms=state.last_timestamp,
                kind=MachineFactKind.TYPING_STOPPED.value,
                source="rrweb_deterministic",
                evidence_source="keystroke",
                attribution="candidate",
                confidence=1.0,
                detail={
                    "elementId": element_id,
                    "durationMs": duration_ms,
                    "typingStartOffsetMs": state.first_offset_ms,
                    "finalCharCount": state.final_char_count,
                    "peakCharCount": state.peak_char_count,
                    "totalInsertedChars": state.total_inserted_chars,
                    "avgCharsPerSecond": avg_cps,
                    "insertionEvents": state.insertion_events,
                    "deletionEvents": state.deletion_events,
                    **({"sectionId": state.section_id} if state.section_id else {}),
                    **({"sectionType": state.section_type} if state.section_type else {}),
                },
            )
        )
    return sorted(facts, key=lambda fact: fact.start_offset_ms), had_monaco
