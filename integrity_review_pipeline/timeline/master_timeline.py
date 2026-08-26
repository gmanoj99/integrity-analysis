"""Session mapper — canonical timeline synchronization (TS masterTimeline.ts parity)."""

from __future__ import annotations

import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

from ..contracts.evidence import EvidenceType, ManifestSet
from ..contracts.review import ReviewRequest
from ..contracts.timeline import (
    ArtifactRecord,
    GapSegment,
    MasterTimeline,
    MergedSegment,
    RrwebChunkSpan,
    SessionSection,
    SyncReport,
    TimelineEvent,
    TimelineEventRecord,
    TimelineLayer,
    UnknownSegment,
    VideoChunkSpan,
)
from .activity_log_timeline import build_canonical_timeline

MASTER_TIMELINE_LOGIC_VERSION = "mt-screen-v1"
GAP_THRESHOLD_MS = 2_000
MERGE_TOLERANCE_MS = 5_000
SECTION_GAP_THRESHOLD_MS = 60_000
DEFAULT_CHUNK_DURATION_MS = 30_000
KEYSTROKE_FALLBACK_DURATION_MS = 90_000
MAX_SESSION_DURATION_MS = 8 * 60 * 60 * 1_000

_SECTION_LABELS = {
    "coding": "Coding",
    "html_coding": "HTML Coding",
    "primitive_coding": "Algorithmic Coding",
    "sql_coding": "SQL Coding",
    "mcq": "MCQ",
}


@dataclass(frozen=True, slots=True)
class ParsedChunkId:
    start_epoch_ms: int
    end_epoch_ms: int
    duration_ms: int | None


@dataclass(frozen=True, slots=True)
class ReceivedChunk:
    sequence: int
    chunk_id: str
    duration_ms: int | None = None
    section_id: str | None = None


def parse_chunk_id(chunk_id: str) -> ParsedChunkId | None:
    match = re.search(
        r"-(\d{13})(?:__(\d+))?(?:__(?:events|metadata))?(?:$)", chunk_id
    )
    if not match:
        return None
    upload_epoch_ms = int(match.group(1))
    duration_ms = int(match.group(2)) if match.group(2) is not None else None
    if duration_ms is not None:
        return ParsedChunkId(
            start_epoch_ms=upload_epoch_ms - duration_ms,
            end_epoch_ms=upload_epoch_ms,
            duration_ms=duration_ms,
        )
    return ParsedChunkId(
        start_epoch_ms=upload_epoch_ms,
        end_epoch_ms=upload_epoch_ms,
        duration_ms=None,
    )


def to_offset(raw_ms: int, session_start_ms: int) -> int:
    return raw_ms - session_start_ms


def to_raw(offset_ms: int, session_start_ms: int) -> int:
    return offset_ms + session_start_ms


def parse_section_from_chunk_id(chunk_id: str) -> str | None:
    match = re.match(r"^(?:video|keystrokeData|screenRecording)-([a-z_]+)-\d{13}", chunk_id)
    return match.group(1) if match else None


def _median(values: list[int]) -> int:
    if not values:
        return 0
    return int(statistics.median(values))


def _section_label(section_id: str) -> str:
    return _SECTION_LABELS.get(section_id, section_id)


def _chunks_for_type(manifests: ManifestSet, evidence_type: EvidenceType) -> list[ReceivedChunk]:
    manifest = manifests.by_type(evidence_type)
    return [
        ReceivedChunk(
            sequence=chunk.sequence,
            chunk_id=chunk.chunk_id,
            duration_ms=chunk.duration_ms,
            section_id=chunk.section_id,
        )
        for chunk in manifest.chunks
        if chunk.sidecar_role is None
    ]


def _events_by_sequence(
    timeline_events: list[TimelineEventRecord],
    evidence_types: set[str],
) -> dict[int, list[int]]:
    grouped: dict[int, list[int]] = defaultdict(list)
    for event in timeline_events:
        if event.evidence_type not in evidence_types or event.sequence is None:
            continue
        grouped[event.sequence].append(event.timestamp_ms)
    for timestamps in grouped.values():
        timestamps.sort()
    return grouped


def _merge_unknown_segments(segments: list[UnknownSegment]) -> list[UnknownSegment]:
    if not segments:
        return []
    ordered = sorted(segments, key=lambda item: item.start_offset_ms)
    merged: list[UnknownSegment] = []
    current = ordered[0].model_copy(deep=True)
    current.sequences = list(current.sequences or [])

    priority = {"gap": 2, "missing_sequence": 1, "fetch_error": 0}

    for nxt in ordered[1:]:
        if nxt.start_offset_ms < current.end_offset_ms:
            current.end_offset_ms = max(current.end_offset_ms, nxt.end_offset_ms)
            if nxt.sequences:
                current.sequences.extend(nxt.sequences)
            if priority[nxt.reason] > priority[current.reason]:
                current.reason = nxt.reason
        else:
            merged.append(current)
            current = nxt.model_copy(deep=True)
            current.sequences = list(current.sequences or [])
    merged.append(current)
    return merged


def _emit_gap_segment(
    first_seq: int,
    run_seqs: list[int],
    reason: Literal["missing_sequence", "fetch_error"],
    spans: list[VideoChunkSpan],
    estimated_chunk_duration_ms: int,
    out: list[UnknownSegment],
) -> None:
    last_seq = run_seqs[-1]
    prev_span = next((span for span in reversed(spans) if span.sequence < first_seq), None)
    next_span = next((span for span in spans if span.sequence > last_seq), None)
    start_off = prev_span.end_offset_ms if prev_span else 0
    natural_end = start_off + estimated_chunk_duration_ms * len(run_seqs)
    end_off = (
        min(natural_end, next_span.start_offset_ms)
        if next_span
        else natural_end
    )
    out.append(
        UnknownSegment(
            start_offset_ms=start_off,
            end_offset_ms=end_off,
            reason=reason,
            sequences=run_seqs,
        )
    )


def build_video_spans(
    received_chunks: list[ReceivedChunk],
    missing_sequences: list[int],
    total_chunks: int,
    events_by_video_seq: dict[int, list[int]],
    session_start_ms: int,
    estimated_chunk_duration_ms: int,
) -> tuple[list[VideoChunkSpan], list[UnknownSegment]]:
    sorted_received = sorted(received_chunks, key=lambda item: item.sequence)
    received_seq_set = {chunk.sequence for chunk in sorted_received}

    anchored: list[tuple[ReceivedChunk, int]] = []
    unanchored: list[ReceivedChunk] = []

    for chunk in sorted_received:
        parsed = parse_chunk_id(chunk.chunk_id)
        if parsed:
            duration = parsed.duration_ms if parsed.duration_ms is not None else chunk.duration_ms
            anchored.append((chunk, parsed.start_epoch_ms))
        else:
            timestamps = events_by_video_seq.get(chunk.sequence)
            if timestamps:
                anchored.append((chunk, timestamps[0]))
            else:
                unanchored.append(chunk)

    anchored.sort(key=lambda item: item[1])

    combined: list[tuple[int, str, int, str | None, int]] = []

    if anchored:
        first_chunk, first_anchor = anchored[0]
        first_seq = first_chunk.sequence
        before = sorted(
            (chunk for chunk in unanchored if chunk.sequence < first_seq),
            key=lambda item: item.sequence,
        )
        back_cursor = first_anchor
        for chunk in reversed(before):
            duration = chunk.duration_ms or estimated_chunk_duration_ms
            back_cursor -= duration
            combined.insert(
                0,
                (
                    chunk.sequence,
                    chunk.chunk_id,
                    duration,
                    chunk.section_id,
                    back_cursor,
                ),
            )

        for index, (chunk, start_raw_ms) in enumerate(anchored):
            duration = chunk.duration_ms or estimated_chunk_duration_ms
            combined.append(
                (chunk.sequence, chunk.chunk_id, duration, chunk.section_id, start_raw_ms)
            )
            next_anchor_seq = (
                anchored[index + 1][0].sequence if index + 1 < len(anchored) else 10**9
            )
            between = sorted(
                (
                    item
                    for item in unanchored
                    if chunk.sequence < item.sequence < next_anchor_seq
                ),
                key=lambda item: item.sequence,
            )
            after_cursor = start_raw_ms + duration
            for item in between:
                item_duration = item.duration_ms or estimated_chunk_duration_ms
                combined.append(
                    (
                        item.sequence,
                        item.chunk_id,
                        item_duration,
                        item.section_id,
                        after_cursor,
                    )
                )
                after_cursor += item_duration

        last_chunk, last_anchor = anchored[-1]
        last_seq = last_chunk.sequence
        after = sorted(
            (chunk for chunk in unanchored if chunk.sequence > last_seq),
            key=lambda item: item.sequence,
        )
        cursor = last_anchor + (
            last_chunk.duration_ms or estimated_chunk_duration_ms
        )
        for chunk in after:
            duration = chunk.duration_ms or estimated_chunk_duration_ms
            combined.append(
                (chunk.sequence, chunk.chunk_id, duration, chunk.section_id, cursor)
            )
            cursor += duration
    else:
        cursor = session_start_ms
        for chunk in sorted_received:
            duration = chunk.duration_ms or estimated_chunk_duration_ms
            combined.append(
                (chunk.sequence, chunk.chunk_id, duration, chunk.section_id, cursor)
            )
            cursor += duration

    combined.sort(key=lambda item: item[4])

    spans: list[VideoChunkSpan] = []
    unknowns: list[UnknownSegment] = []

    for sequence, chunk_id, duration, section_id, start_raw_ms in combined:
        start_offset_ms = start_raw_ms - session_start_ms
        end_offset_ms = start_offset_ms + duration
        spans.append(
            VideoChunkSpan(
                sequence=sequence,
                chunk_id=chunk_id,
                start_offset_ms=start_offset_ms,
                end_offset_ms=end_offset_ms,
                duration_ms=duration,
                section_id=section_id,
            )
        )

    for index in range(1, len(spans)):
        prev = spans[index - 1]
        curr = spans[index]
        gap_ms = curr.start_offset_ms - prev.end_offset_ms
        if gap_ms > GAP_THRESHOLD_MS:
            unknowns.append(
                UnknownSegment(
                    start_offset_ms=prev.end_offset_ms,
                    end_offset_ms=curr.start_offset_ms,
                    reason="gap",
                    sequences=[prev.sequence, curr.sequence],
                )
            )

    max_seq = max(
        total_chunks,
        sorted_received[-1].sequence if sorted_received else 0,
    )
    min_seq = sorted_received[0].sequence if sorted_received else 1
    missing_set = set(missing_sequences)
    pending: list[tuple[int, Literal["missing_sequence", "fetch_error"]]] = []
    for seq in range(min_seq, max_seq + 1):
        if seq in received_seq_set:
            continue
        pending.append(
            (
                seq,
                "missing_sequence" if seq in missing_set else "fetch_error",
            )
        )

    if pending:
        run_start, run_reason = pending[0]
        run_seqs = [run_start]
        for prev, curr in zip(pending, pending[1:]):
            if curr[0] == prev[0] + 1 and curr[1] == prev[1]:
                run_seqs.append(curr[0])
            else:
                _emit_gap_segment(
                    run_start,
                    run_seqs,
                    run_reason,
                    spans,
                    estimated_chunk_duration_ms,
                    unknowns,
                )
                run_start, run_reason = curr
                run_seqs = [run_start]
        _emit_gap_segment(
            run_start,
            run_seqs,
            run_reason,
            spans,
            estimated_chunk_duration_ms,
            unknowns,
        )

    return spans, _merge_unknown_segments(unknowns)


def _chunk_sort_epoch(chunk: ReceivedChunk) -> tuple[int, int]:
    parsed = parse_chunk_id(chunk.chunk_id)
    if parsed is None:
        return (0, chunk.sequence)
    return (parsed.end_epoch_ms or parsed.start_epoch_ms, chunk.sequence)


def _chunk_epoch_end(chunk: ReceivedChunk) -> int:
    parsed = parse_chunk_id(chunk.chunk_id)
    if parsed is None:
        return 0
    return parsed.end_epoch_ms or parsed.start_epoch_ms


def build_rrweb_spans(
    received_chunks: list[ReceivedChunk],
    events_by_keystroke_seq: dict[int, list[int]],
    session_start_ms: int,
    estimated_keystroke_span_ms: int,
) -> list[RrwebChunkSpan]:
    sorted_keystroke = sorted(received_chunks, key=_chunk_sort_epoch)

    spans: list[RrwebChunkSpan] = []
    prev_end_raw_ms = 0

    for chunk in sorted_keystroke:
        parsed = parse_chunk_id(chunk.chunk_id)
        epoch_end = 0
        if parsed is not None:
            epoch_end = parsed.end_epoch_ms or parsed.start_epoch_ms
        db_ts = events_by_keystroke_seq.get(chunk.sequence, [])

        if len(db_ts) >= 2:
            start_raw_ms = db_ts[0]
            end_raw_ms = max(db_ts[-1], epoch_end or db_ts[-1])
        elif len(db_ts) == 1:
            only = db_ts[0]
            end_raw_ms = max(epoch_end, only) if epoch_end > 0 else only
            start_raw_ms = (
                only
                if only < end_raw_ms
                else prev_end_raw_ms
                if prev_end_raw_ms > 0
                else end_raw_ms - estimated_keystroke_span_ms
            )
        elif epoch_end > 0:
            end_raw_ms = epoch_end
            start_raw_ms = (
                prev_end_raw_ms
                if prev_end_raw_ms > 0
                and end_raw_ms - prev_end_raw_ms <= estimated_keystroke_span_ms * 2
                else end_raw_ms - estimated_keystroke_span_ms
            )
        else:
            start_raw_ms = prev_end_raw_ms if prev_end_raw_ms > 0 else session_start_ms
            end_raw_ms = start_raw_ms + estimated_keystroke_span_ms

        if end_raw_ms < start_raw_ms:
            start_raw_ms, end_raw_ms = end_raw_ms, start_raw_ms

        start_offset_ms = max(0, start_raw_ms - session_start_ms)
        end_offset_ms = max(start_offset_ms, end_raw_ms - session_start_ms)
        spans.append(
            RrwebChunkSpan(
                sequence=chunk.sequence,
                chunk_id=chunk.chunk_id,
                start_offset_ms=start_offset_ms,
                end_offset_ms=end_offset_ms,
                section_id=chunk.section_id,
            )
        )
        prev_end_raw_ms = max(prev_end_raw_ms, end_raw_ms)

    spans.sort(key=lambda item: item.start_offset_ms)
    return spans


def build_merged_segments(
    spans: list[VideoChunkSpan] | list[RrwebChunkSpan],
    *,
    tolerance_ms: int = MERGE_TOLERANCE_MS,
) -> list[MergedSegment]:
    if not spans:
        return []

    def start(item: VideoChunkSpan | RrwebChunkSpan) -> int:
        return item.start_offset_ms

    def end(item: VideoChunkSpan | RrwebChunkSpan) -> int:
        return item.end_offset_ms

    ordered = sorted(spans, key=start)
    first = ordered[0]
    section_id = (
        first.section_id
        if isinstance(first, VideoChunkSpan)
        else parse_section_from_chunk_id(first.chunk_id) or first.section_id
    )
    segments = [
        MergedSegment(
            start_ms=start(first),
            end_ms=end(first),
            duration_ms=end(first) - start(first),
            chunk_ids=[first.chunk_id],
            sequences=[first.sequence],
            sections=[section_id] if section_id else [],
            chunk_count=1,
        )
    ]

    for span in ordered[1:]:
        current = segments[-1]
        gap_ms = start(span) - current.end_ms
        section = (
            span.section_id
            if isinstance(span, VideoChunkSpan)
            else parse_section_from_chunk_id(span.chunk_id) or span.section_id
        )
        if gap_ms <= tolerance_ms:
            new_end = max(current.end_ms, end(span))
            sections = list(current.sections)
            if section and section not in sections:
                sections.append(section)
            segments[-1] = MergedSegment(
                start_ms=current.start_ms,
                end_ms=new_end,
                duration_ms=new_end - current.start_ms,
                chunk_ids=[*current.chunk_ids, span.chunk_id],
                sequences=[*current.sequences, span.sequence],
                sections=sections,
                chunk_count=current.chunk_count + 1,
            )
        else:
            segments.append(
                MergedSegment(
                    start_ms=start(span),
                    end_ms=end(span),
                    duration_ms=end(span) - start(span),
                    chunk_ids=[span.chunk_id],
                    sequences=[span.sequence],
                    sections=[section] if section else [],
                    chunk_count=1,
                )
            )
    return segments


def build_gaps_from_merged_segments(segments: list[MergedSegment]) -> list[GapSegment]:
    if not segments:
        return []

    gaps: list[GapSegment] = []
    first = segments[0]
    if first.start_ms > GAP_THRESHOLD_MS:
        gaps.append(
            GapSegment(
                start_ms=0,
                end_ms=first.start_ms,
                duration_ms=first.start_ms,
                reason="before_recording",
                adjacent_artifact_ids=(None, first.chunk_ids[0] if first.chunk_ids else None),
            )
        )

    for prev, curr in zip(segments, segments[1:]):
        gap_ms = curr.start_ms - prev.end_ms
        if gap_ms <= GAP_THRESHOLD_MS:
            continue
        prev_sections = set(prev.sections)
        sections_differ = any(section not in prev_sections for section in curr.sections)
        if sections_differ:
            reason = "section_transition"
        elif gap_ms < SECTION_GAP_THRESHOLD_MS:
            reason = "recording_gap"
        else:
            reason = "unknown"
        gaps.append(
            GapSegment(
                start_ms=prev.end_ms,
                end_ms=curr.start_ms,
                duration_ms=gap_ms,
                reason=reason,
                adjacent_artifact_ids=(
                    prev.chunk_ids[-1] if prev.chunk_ids else None,
                    curr.chunk_ids[0] if curr.chunk_ids else None,
                ),
            )
        )
    return gaps


def build_artifact_registry(
    video_spans: list[VideoChunkSpan],
    rrweb_spans: list[RrwebChunkSpan],
    screen_spans: list[VideoChunkSpan],
    session_start_ms: int,
) -> list[ArtifactRecord]:
    records: list[ArtifactRecord] = []

    for span in video_spans:
        parsed = parse_chunk_id(span.chunk_id)
        epoch_start = (
            parsed.start_epoch_ms
            if parsed is not None
            else session_start_ms + span.start_offset_ms
        )
        section_id = parse_section_from_chunk_id(span.chunk_id) or span.section_id
        records.append(
            ArtifactRecord(
                artifact_id=span.chunk_id,
                artifact_type="video",
                section_id=section_id,
                sequence=span.sequence,
                session_start_ms=span.start_offset_ms,
                session_end_ms=span.end_offset_ms,
                local_end_ms=span.duration_ms,
                epoch_start_ms=epoch_start,
                epoch_end_ms=epoch_start + span.duration_ms,
                quality="ok" if parsed is not None else "estimated",
            )
        )

    for span in rrweb_spans:
        parsed = parse_chunk_id(span.chunk_id)
        duration_ms = span.end_offset_ms - span.start_offset_ms
        epoch_start = (
            parsed.start_epoch_ms
            if parsed is not None
            else session_start_ms + span.start_offset_ms
        )
        records.append(
            ArtifactRecord(
                artifact_id=span.chunk_id,
                artifact_type="rrweb",
                section_id=parse_section_from_chunk_id(span.chunk_id) or span.section_id,
                sequence=span.sequence,
                session_start_ms=span.start_offset_ms,
                session_end_ms=span.end_offset_ms,
                local_end_ms=duration_ms,
                epoch_start_ms=epoch_start,
                epoch_end_ms=epoch_start + duration_ms,
                quality="ok" if parsed is not None else "estimated",
            )
        )

    for span in screen_spans:
        parsed = parse_chunk_id(span.chunk_id)
        epoch_start = (
            parsed.start_epoch_ms
            if parsed is not None
            else session_start_ms + span.start_offset_ms
        )
        section_id = parse_section_from_chunk_id(span.chunk_id) or span.section_id
        records.append(
            ArtifactRecord(
                artifact_id=span.chunk_id,
                artifact_type="screenRecording",
                section_id=section_id,
                sequence=span.sequence,
                session_start_ms=span.start_offset_ms,
                session_end_ms=span.end_offset_ms,
                local_end_ms=span.duration_ms,
                epoch_start_ms=epoch_start,
                epoch_end_ms=epoch_start + span.duration_ms,
                quality="ok" if parsed is not None else "estimated",
            )
        )

    return sorted(
        records,
        key=lambda item: (item.session_start_ms, item.artifact_type),
    )


def _build_sections_from_artifacts(artifacts: list[ArtifactRecord]) -> list[SessionSection]:
    section_map: dict[str, tuple[int, int]] = {}
    for artifact in artifacts:
        if not artifact.section_id:
            continue
        existing = section_map.get(artifact.section_id)
        if existing is None:
            section_map[artifact.section_id] = (
                artifact.session_start_ms,
                artifact.session_end_ms,
            )
        else:
            section_map[artifact.section_id] = (
                min(existing[0], artifact.session_start_ms),
                max(existing[1], artifact.session_end_ms),
            )

    return [
        SessionSection(
            section_id=section_id,
            label=_section_label(section_id),
            start_ms=bounds[0],
            end_ms=bounds[1],
        )
        for section_id, bounds in sorted(section_map.items(), key=lambda item: item[1][0])
    ]


def _attribute_events_to_artifacts(
    events: list[TimelineEvent],
    video_artifacts: list[ArtifactRecord],
) -> list[TimelineEvent]:
    enriched: list[TimelineEvent] = []
    for event in events:
        artifact = next(
            (
                item
                for item in video_artifacts
                if item.session_start_ms <= event.session_offset_ms <= item.session_end_ms
            ),
            None,
        )
        if artifact is None:
            enriched.append(event)
            continue
        enriched.append(
            event.model_copy(
                update={
                    "local_offset_ms": max(
                        0, event.session_offset_ms - artifact.session_start_ms
                    ),
                    "chunk_artifact_id": artifact.artifact_id,
                }
            )
        )
    return enriched


def _activity_log_events(canonical) -> list[TimelineEvent]:
    events: list[TimelineEvent] = []
    for index, anchor in enumerate(canonical.anchors):
        detail: dict[str, str] = {}
        if anchor.section_id:
            detail["sectionId"] = anchor.section_id
        if anchor.end_reason:
            detail["endReason"] = anchor.end_reason
        events.append(
            TimelineEvent(
                id=f"anchor-{anchor.order if anchor.order is not None else index}",
                source="activityLog",
                kind=anchor.type,
                raw_timestamp_ms=anchor.epoch_ms,
                session_offset_ms=max(0, anchor.session_offset_ms),
                detail=detail,
            )
        )
    return events


def _timeline_events_from_records(
    records: list[TimelineEventRecord],
    session_start_ms: int,
) -> list[TimelineEvent]:
    return [
        TimelineEvent(
            id=record.event_id or f"evt-{record.timestamp_ms}-{index}",
            source=record.evidence_type,
            kind=record.kind,
            raw_timestamp_ms=record.timestamp_ms,
            session_offset_ms=(
                0 if session_start_ms == 0 else to_offset(record.timestamp_ms, session_start_ms)
            ),
            signal_source=record.signal_source,
            detail=dict(record.detail),
        )
        for index, record in enumerate(records)
        if (
            session_start_ms == 0
            or to_offset(record.timestamp_ms, session_start_ms) <= MAX_SESSION_DURATION_MS
        )
    ]


def build_master_timeline(
    request: ReviewRequest,
    *,
    timeline_events: list[TimelineEventRecord] | None = None,
    missing_sequences: dict[EvidenceType, list[int]] | None = None,
) -> MasterTimeline:
    timeline_events = timeline_events or []
    missing_sequences = missing_sequences or {}

    canonical = build_canonical_timeline(
        request.activity_timeline,
        request.sections,
    )

    video_chunks = _chunks_for_type(request.evidence, EvidenceType.VIDEO)
    keystroke_chunks = _chunks_for_type(request.evidence, EvidenceType.KEYSTROKE_DATA)
    screen_chunks = _chunks_for_type(request.evidence, EvidenceType.SCREEN_RECORDING)

    events_by_video_seq = _events_by_sequence(timeline_events, {"video"})
    events_by_keystroke_seq = _events_by_sequence(
        timeline_events, {"keystrokeData", "screenRecording"}
    )

    all_event_timestamps = [event.timestamp_ms for event in timeline_events]
    t0_source: SyncReport.__annotations__["t0_source"] = "none"
    if canonical.source == "activity_logs" and canonical.t0 > 0:
        session_start_ms = canonical.t0
        t0_source = "activity_logs"
    elif all_event_timestamps:
        session_start_ms = min(all_event_timestamps)
        t0_source = "fallback_events"
    else:
        chunk_epochs = [
            parsed.start_epoch_ms
            for chunk in video_chunks
            if (parsed := parse_chunk_id(chunk.chunk_id)) is not None and parsed.start_epoch_ms > 0
        ]
        session_start_ms = min(chunk_epochs) if chunk_epochs else 0
        t0_source = "fallback_chunk_epoch" if session_start_ms > 0 else "none"

    known_video_durations = [
        chunk.duration_ms or (parse_chunk_id(chunk.chunk_id).duration_ms if parse_chunk_id(chunk.chunk_id) else 0)
        for chunk in video_chunks
    ]
    known_video_durations = [value for value in known_video_durations if value and value > 0]
    estimated_video_duration_ms = _median(known_video_durations) or DEFAULT_CHUNK_DURATION_MS

    video_manifest = request.evidence.by_type(EvidenceType.VIDEO)
    video_spans, unknown_segments = build_video_spans(
        video_chunks,
        missing_sequences.get(EvidenceType.VIDEO, []),
        video_manifest.total_chunks or len(video_chunks),
        events_by_video_seq,
        session_start_ms,
        estimated_video_duration_ms,
    )

    known_screen_durations = [
        chunk.duration_ms or (parse_chunk_id(chunk.chunk_id).duration_ms if parse_chunk_id(chunk.chunk_id) else 0)
        for chunk in screen_chunks
    ]
    known_screen_durations = [value for value in known_screen_durations if value and value > 0]
    estimated_screen_duration_ms = (
        _median(known_screen_durations) or estimated_video_duration_ms
    )
    screen_spans, _ = build_video_spans(
        screen_chunks,
        missing_sequences.get(EvidenceType.SCREEN_RECORDING, []),
        len(screen_chunks),
        {},
        session_start_ms,
        estimated_screen_duration_ms,
    )

    sorted_keystroke = sorted(keystroke_chunks, key=_chunk_sort_epoch)
    sorted_epochs = [
        epoch
        for chunk in sorted_keystroke
        if (epoch := _chunk_epoch_end(chunk)) > 0
    ]
    inter_pair_gaps = [
        sorted_epochs[index] - sorted_epochs[index - 1]
        for index in range(1, len(sorted_epochs))
        if sorted_epochs[index] - sorted_epochs[index - 1] > MERGE_TOLERANCE_MS
    ]
    known_keystroke_durations = [
        chunk.duration_ms for chunk in keystroke_chunks if chunk.duration_ms and chunk.duration_ms > 0
    ]
    estimated_keystroke_span_ms = (
        _median(inter_pair_gaps)
        or _median(known_keystroke_durations)
        or KEYSTROKE_FALLBACK_DURATION_MS
    )

    rrweb_spans = build_rrweb_spans(
        keystroke_chunks,
        events_by_keystroke_seq,
        session_start_ms,
        estimated_keystroke_span_ms,
    )

    db_events = _timeline_events_from_records(timeline_events, session_start_ms)
    activity_events = _activity_log_events(canonical) if canonical.source == "activity_logs" else []
    raw_events = sorted(
        [*db_events, *activity_events],
        key=lambda item: item.session_offset_ms,
    )[:500]

    last_video_offset = video_spans[-1].end_offset_ms if video_spans else 0
    last_screen_offset = screen_spans[-1].end_offset_ms if screen_spans else 0
    in_session_rrweb = [
        span for span in rrweb_spans if span.start_offset_ms <= MAX_SESSION_DURATION_MS
    ]
    last_rrweb_offset = in_session_rrweb[-1].end_offset_ms if in_session_rrweb else 0
    in_session_events = [
        event for event in raw_events if event.session_offset_ms <= MAX_SESSION_DURATION_MS
    ]
    last_event_offset = (
        in_session_events[-1].session_offset_ms if in_session_events else 0
    )

    canonical_duration_ms = canonical.total_duration_ms if canonical.total_duration_ms > 0 else 0
    duration_source: SyncReport.__annotations__["duration_source"] = "none"
    if canonical_duration_ms > 0:
        duration_ms = max(canonical_duration_ms, last_video_offset, last_screen_offset)
        duration_source = "activity_logs"
    elif last_video_offset > 0:
        duration_ms = max(last_video_offset, last_screen_offset)
        duration_source = "video_chunks"
    elif last_screen_offset > 0:
        duration_ms = last_screen_offset
        duration_source = "screen_chunks"
    else:
        duration_ms = max(
            last_rrweb_offset,
            last_event_offset,
            video_manifest.total_duration_ms or 0,
        )
        duration_source = "rrweb_chunks" if last_rrweb_offset > 0 else "none"

    artifact_registry = build_artifact_registry(
        video_spans, rrweb_spans, screen_spans, session_start_ms
    )
    merged_video_segments = build_merged_segments(video_spans)
    merged_keystroke_segments = build_merged_segments(rrweb_spans)
    merged_screen_segments = build_merged_segments(screen_spans)
    gaps = build_gaps_from_merged_segments(merged_video_segments)

    video_artifacts = [item for item in artifact_registry if item.artifact_type == "video"]
    events = _attribute_events_to_artifacts(raw_events, video_artifacts)

    sections = (
        list(canonical.sections)
        if canonical.sections
        else _build_sections_from_artifacts(artifact_registry)
    )

    layers = [
        TimelineLayer(type="video", artifacts=[item for item in artifact_registry if item.artifact_type == "video"]),
        TimelineLayer(type="rrweb", artifacts=[item for item in artifact_registry if item.artifact_type == "rrweb"]),
        TimelineLayer(
            type="screenRecording",
            artifacts=[item for item in artifact_registry if item.artifact_type == "screenRecording"],
        ),
    ]
    layers = [layer for layer in layers if layer.artifacts]

    unresolved_issues: list[str] = []
    if t0_source == "none":
        unresolved_issues.append(
            "No T0 source: activity logs, behavior events, and chunk epochs are all absent"
        )
    if t0_source == "fallback_events":
        unresolved_issues.append(
            "T0 is min(behavior_event.timestamp) — activity logs not yet fetched for this bundle"
        )
    if t0_source == "fallback_chunk_epoch":
        unresolved_issues.append(
            "T0 derived from earliest video chunk epoch — activity logs unavailable"
        )
    if not canonical.sections and not sections:
        unresolved_issues.append("No sections resolved from any source")

    sorted_video_spans = sorted(video_spans, key=lambda item: item.start_offset_ms)
    sorted_rrweb_spans = sorted(rrweb_spans, key=lambda item: item.start_offset_ms)
    sorted_screen_spans = sorted(screen_spans, key=lambda item: item.start_offset_ms)
    sorted_events = sorted(events, key=lambda item: item.session_offset_ms)

    sync_report = SyncReport(
        t0_epoch_ms=session_start_ms,
        t0_source=t0_source,
        canonical_duration_ms=duration_ms,
        duration_source=duration_source,
        video_alignment={
            "chunkCount": len(video_spans),
            "firstChunkSessionOffsetMs": (
                sorted_video_spans[0].start_offset_ms if sorted_video_spans else None
            ),
            "lastChunkEndSessionOffsetMs": (
                sorted_video_spans[-1].end_offset_ms if sorted_video_spans else None
            ),
        },
        rrweb_alignment={
            "chunkCount": len(rrweb_spans),
            "firstChunkSessionOffsetMs": (
                sorted_rrweb_spans[0].start_offset_ms if sorted_rrweb_spans else None
            ),
        },
        screen_alignment={
            "chunkCount": len(screen_spans),
            "firstChunkSessionOffsetMs": (
                sorted_screen_spans[0].start_offset_ms if sorted_screen_spans else None
            ),
            "lastChunkEndSessionOffsetMs": (
                sorted_screen_spans[-1].end_offset_ms if sorted_screen_spans else None
            ),
        },
        machine_fact_alignment={
            "eventCount": len(events),
            "firstEventSessionOffsetMs": (
                sorted_events[0].session_offset_ms if sorted_events else None
            ),
            "lastEventSessionOffsetMs": (
                sorted_events[-1].session_offset_ms if sorted_events else None
            ),
        },
        activity_log_anchor_count=len(canonical.anchors),
        canonical_section_count=len(canonical.sections),
        unresolved_issues=unresolved_issues,
    )

    return MasterTimeline(
        candidate_id=request.candidate_id,
        assessment_id=request.assessment_id,
        session_start_ms=session_start_ms,
        duration_ms=duration_ms,
        canonical_timeline=canonical,
        sync_report=sync_report,
        artifact_registry=artifact_registry,
        layers=layers,
        gaps=gaps,
        sections=sections,
        merged_video_segments=merged_video_segments,
        merged_keystroke_segments=merged_keystroke_segments,
        merged_screen_segments=merged_screen_segments,
        video_chunk_spans=video_spans,
        rrweb_chunk_spans=rrweb_spans,
        screen_chunk_spans=screen_spans,
        unknown_segments=unknown_segments,
        events=events,
    )
