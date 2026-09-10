"""Shared video derivation core — config, episodes, helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import log10

from ..contracts.perception import PerceptionBundle, PerceptionObservation, PerceptionWindow
from ..lib.interval_utils import union_interval_length_ms
from .contracts import (
    ClearedItem,
    ContextualEvent,
    CriticalMoment,
    EvidenceFinding,
    SpeechProof,
    UnknownItem,
    VideoDerivationResult,
    VideoProof,
    VideoProofAnchor,
)

DERIVATION_CONFIG = {
    "MIN_USABLE_CONFIDENCE": 0.3,
    "EPISODE_MAX_GAP_SEC": 22,
    "face": {"IGNORE_BELOW_SEC": 3, "SUSTAINED_SEC": 30, "PROLONGED_SEC": 120},
    "secondPerson": {"MIN_SEC": 3, "ADJACENT_SUSTAINED_SEC": 30},
    # ANCHOR_MIN_EPISODES retired with trackb-v10: gaze is adjudicated by the
    # model, not gated on an episode count here.
    "gaze": {"SEGMENT_MERGE_GAP_SEC": 45},
    "phone": {"MIN_SEC": 3},
    "audioWearable": {"MIN_SEC": 3},
    "notes": {"MIN_SEC": 6},
    "seatDeparture": {"MIN_SEC": 5, "BODY_PRESENT_SUSTAINED_SEC": 30},
    "speech": {"MIN_SEC": 2, "SUMMARY_MAX_CHARS": 400},
    "critical": {"TOP_K": 8, "MAX_TOTAL_SEC": 180, "CLIP_BUFFER_SEC": 5},
}

INTEGRITY_SPEECH_CLASSES = {
    "asking_for_answer",
    "receiving_dictation",
    "discussing_solution",
    "reciting_answer_choices",
}
CONTEXT_SPEECH_CLASSES = {
    "invigilator_or_admin",
    "technical_exam_help",
    "self_talk_or_thinking",
    "casual_non_exam",
    "reading_question",
}
ADMIN_SPEECH_CLASSES = {"invigilator_or_admin", "technical_exam_help"}
OFF_ZONES = {"left", "right", "down", "up", "away"}
SEV_WEIGHT = {"high": 3, "medium": 2, "low": 1}

_id_seq = 0


@dataclass
class Episode:
    observations: list[PerceptionObservation]
    window_ids: list[str]
    t0: int
    t1: int
    duration_ms: int
    avg_confidence: float


def _sec(ms: int) -> int:
    return round(ms / 1000)


def build_episodes(
    observations: list[PerceptionObservation],
    predicate,
) -> list[Episode]:
    max_gap_ms = DERIVATION_CONFIG["EPISODE_MAX_GAP_SEC"] * 1000
    episodes: list[Episode] = []
    cur: list[PerceptionObservation] = []

    def flush() -> None:
        nonlocal cur
        if not cur:
            return
        duration_ms = union_interval_length_ms([(o.start_ms, o.end_ms) for o in cur])
        episodes.append(
            Episode(
                observations=list(cur),
                # Several events in one chunk share a window id; a repeated id
                # would double up in evidence refs and proof anchors.
                window_ids=list(dict.fromkeys(o.window_id for o in cur)),
                t0=min(o.start_ms for o in cur),
                t1=max(o.end_ms for o in cur),
                duration_ms=duration_ms,
                avg_confidence=sum(o.confidence for o in cur) / len(cur),
            )
        )
        cur = []

    for obs in observations:
        if not predicate(obs):
            continue
        if cur and obs.start_ms - cur[-1].end_ms > max_gap_ms:
            flush()
        cur.append(obs)
    flush()
    return episodes


def pick_proof_anchors(
    window_ids: list[str],
    span_start_ms: int,
    bundle: PerceptionBundle,
) -> list[VideoProofAnchor]:
    win_by_id = {w.window_id: w for w in bundle.windows}
    obs_by_id = {o.window_id: o for o in bundle.observations}
    peak_wid = None
    peak_conf = -1.0
    for wid in window_ids:
        w = win_by_id.get(wid)
        o = obs_by_id.get(wid)
        if not w or not w.overlapping_chunks or not o or not o.video_available:
            continue
        if o.confidence > peak_conf:
            peak_conf = o.confidence
            peak_wid = wid
    anchors: list[VideoProofAnchor] = []

    def pick(wid: str | None, t_ms: int) -> None:
        if not wid:
            return
        chunk = (win_by_id.get(wid) or PerceptionWindow.model_construct()).overlapping_chunks
        if not chunk:
            return
        c0 = chunk[0]
        if any(a.chunk_id == c0.chunk_id and a.approx_frame_timestamp_ms == t_ms for a in anchors):
            return
        anchors.append(
            VideoProofAnchor(
                chunk_id=c0.chunk_id,
                chunk_sequence=c0.sequence,
                clip_label=wid,
                approx_frame_timestamp_ms=t_ms,
            )
        )

    if peak_wid:
        pick(peak_wid, win_by_id[peak_wid].start_ms)
    if window_ids:
        pick(window_ids[0], span_start_ms)
        if len(window_ids) > 1:
            pick(window_ids[-1], win_by_id.get(window_ids[-1], win_by_id.get(window_ids[0])).start_ms)
    return anchors[:3]


def _is_usable(o: PerceptionObservation) -> bool:
    return (
        o.video_available
        and o.capture_quality_tier != "NONE"
        and o.confidence >= DERIVATION_CONFIG["MIN_USABLE_CONFIDENCE"]
    )


def _capture_issue(o: PerceptionObservation) -> bool:
    return (
        not o.video_available
        or o.capture_quality_tier in {"NONE", "LOW"}
        or o.environment.framing == "obscured"
    )


def has_integrity_speech(obs: list[PerceptionObservation]) -> bool:
    return any(
        o.audio.speech_content_class in INTEGRITY_SPEECH_CLASSES for o in obs if o.audio.speech_content_class
    )


def has_admin_speech(obs: list[PerceptionObservation]) -> bool:
    return any(
        o.audio.speech_content_class in ADMIN_SPEECH_CLASSES for o in obs if o.audio.speech_content_class
    )


def has_only_cleared_speech(obs: list[PerceptionObservation]) -> bool:
    classes = [
        o.audio.speech_content_class
        for o in obs
        if o.audio.speech_content_class not in {None, "UNKNOWN", "silence"}
    ]
    if not classes:
        return False
    if any(c in INTEGRITY_SPEECH_CLASSES for c in classes):
        return False
    return all(c in CONTEXT_SPEECH_CLASSES or c in ADMIN_SPEECH_CLASSES for c in classes)


def _mk_finding(
    event_type: str,
    ep: Episode,
    bundle: PerceptionBundle,
    *,
    attribution: str,
    severity: str,
    evidence_strength: str,
    verdict: str,
    reasoning: str,
    data_gaps: str | None = None,
    gap_ms_to_next: int | None = None,
    speech_proof: SpeechProof | None = None,
    candidate_gaze_diverted_toward_person: bool | None = None,
) -> EvidenceFinding:
    global _id_seq
    win_map = {w.window_id: w for w in bundle.windows}
    obs_map = {o.window_id: o for o in bundle.observations}
    best_wid = ep.window_ids[0]
    best_conf = -1.0
    chunk = None
    for wid in ep.window_ids:
        chunks = win_map.get(wid, PerceptionWindow.model_construct()).overlapping_chunks
        if not chunks:
            continue
        conf = obs_map.get(wid, PerceptionObservation.model_construct()).confidence
        if conf > best_conf:
            best_conf = conf
            best_wid = wid
            chunk = chunks[0]
    peak_win = win_map.get(best_wid)
    _id_seq += 1
    return EvidenceFinding(
        id=f"perc_{_id_seq}",
        source="video",
        event_type=event_type,
        timestamp_window_ms=(ep.t0, ep.t1),
        attribution=attribution,  # type: ignore[arg-type]
        phase="live_exam",
        severity=severity,  # type: ignore[arg-type]
        evidence_ref=f"perception:w={'+'.join(ep.window_ids)}",
        verdict=verdict,  # type: ignore[arg-type]
        reasoning=reasoning,
        evidence_strength=evidence_strength,  # type: ignore[arg-type]
        candidate_gaze_diverted_toward_person=candidate_gaze_diverted_toward_person,
        data_gaps=data_gaps,
        gap_ms_to_next=gap_ms_to_next,
        speech_proof=speech_proof,
        video_proof=(
            VideoProof(
                chunk_id=chunk.chunk_id,
                chunk_sequence=chunk.sequence,
                clip_label=best_wid,
                approx_frame_timestamp_ms=peak_win.start_ms if peak_win else ep.t0,
            )
            if chunk
            else None
        ),
        video_proof_anchors=pick_proof_anchors(ep.window_ids, ep.t0, bundle),
        occurrence_count=len(ep.observations),
    )
