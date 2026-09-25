from __future__ import annotations

from ..contracts.perception import PerceptionBundle, PerceptionWindow
from .contracts import (
    ClearedItem,
    EvidenceFinding,
    VideoDerivationResult,
    VideoProof,
)
from .video_derivation_classifiers import (
    classify_audio_wearables,
    classify_face,
    classify_gaze,
    classify_phone,
    classify_second_person,
    classify_speech,
    coalesce_cleared_items,
    rank_critical_moments,
)
from .video_derivation_core import (
    _id_seq,
    _is_usable,
    _sec,
    pick_proof_anchors,
)

__all__ = ["build_episodes", "derive_video_findings_from_perception", "DERIVATION_CONFIG"]

from .video_derivation_core import DERIVATION_CONFIG, build_episodes  # noqa: E402


def derive_video_findings_from_perception(
    bundle: PerceptionBundle,
    section_labels: dict[str, dict[str, str]] | None = None,
) -> VideoDerivationResult:
    from . import video_derivation_core as core

    core._id_seq = 0
    win_by_id = {w.window_id: w for w in bundle.windows}
    live = sorted(
        (
            o
            for o in bundle.observations
            if win_by_id.get(o.window_id) and win_by_id[o.window_id].phase == "live_exam"
        ),
        key=lambda o: o.start_ms,
    )
    usable = [o for o in live if _is_usable(o)]

    face = classify_face(usable, bundle)
    people = classify_second_person(usable, bundle)
    gaze = classify_gaze(usable, bundle)
    phone = classify_phone(usable, bundle)
    wearables = classify_audio_wearables(usable, bundle)
    speech = classify_speech(usable, bundle)

    flagged_raw = [
        *face["findings"],
        *people["findings"],
        *gaze["findings"],
        *phone["findings"],
        *wearables["findings"],
        *speech["findings"],
    ]
    cleared_ledger: list[ClearedItem] = [
        *face["cleared"],
        *people["cleared"],
        *gaze["cleared"],
        *phone["cleared"],
        *wearables["cleared"],
        *speech["cleared"],
    ]

    admin_spans = [
        e.timestamp_window_ms
        for e in speech["contextual"]
        if e.event_type in {"invigilator_interaction", "technical_exam_help"}
    ]
    integrity_spans = [f.timestamp_window_ms for f in speech["findings"]]

    def overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
        return a[0] < b[1] and b[0] < a[1]

    flagged: list[EvidenceFinding] = []
    for finding in flagged_raw:
        if finding.event_type not in {"external_help", "multiple_faces"}:
            flagged.append(finding)
            continue
        if not any(overlaps(finding.timestamp_window_ms, s) for s in admin_spans):
            flagged.append(finding)
            continue
        if any(overlaps(finding.timestamp_window_ms, s) for s in integrity_spans):
            flagged.append(finding)
            continue
        cleared_ledger.append(
            ClearedItem(
                event_type=finding.event_type,
                reason="cross_modal_invigilator_audio",
                note=(
                    f"Cleared: {finding.event_type} overlapped invigilator/admin speech "
                    "with no answer-discussion audio."
                ),
                window_ids=[],
                timestamp_window_ms=finding.timestamp_window_ms,
                duration_sec=max(1, _sec(finding.timestamp_window_ms[1] - finding.timestamp_window_ms[0])),
                speech_proof=finding.speech_proof,
            )
        )

    unknown_ledger = [*face["unknowns"], *people["unknowns"], *speech["unknowns"]]
    contextual_events = speech["contextual"]

    cleared_findings: list[EvidenceFinding] = []
    for cleared in coalesce_cleared_items(cleared_ledger):
        chunk = next(
            (
                ch
                for wid in cleared.window_ids
                for ch in (win_by_id.get(wid) or PerceptionWindow.model_construct()).overlapping_chunks
            ),
            None,
        )
        core._id_seq += 1
        cleared_findings.append(
            EvidenceFinding(
                id=f"perc_clr_{core._id_seq}",
                source="video",
                event_type=cleared.event_type,
                timestamp_window_ms=cleared.timestamp_window_ms,
                attribution="environment",
                phase="live_exam",
                severity="low",
                evidence_ref=f"perception:w={'+'.join(cleared.window_ids)}",
                verdict="cleared",
                reasoning=cleared.note,
                evidence_strength="thin",
                data_gaps=cleared.reason,
                occurrence_count=cleared.occurrence_count,
                speech_proof=cleared.speech_proof,
                video_proof=(
                    VideoProof(
                        chunk_id=chunk.chunk_id,
                        chunk_sequence=chunk.sequence,
                        clip_label=cleared.window_ids[0] if cleared.window_ids else None,
                        approx_frame_timestamp_ms=cleared.timestamp_window_ms[0],
                    )
                    if chunk
                    else None
                ),
                video_proof_anchors=pick_proof_anchors(cleared.window_ids, cleared.timestamp_window_ms[0], bundle),
            )
        )

    critical_moments = rank_critical_moments(flagged, bundle, section_labels)
    parts: list[str] = []

    def count(event_type: str) -> int:
        return sum(1 for f in flagged if f.event_type == event_type)

    if count("no_candidate"):
        parts.append(f"Face-absence episodes: {count('no_candidate')}.")
    if count("phone_usage"):
        parts.append(f"Phone handling: {count('phone_usage')}.")
    if count("suspicious_eye_movement"):
        parts.append(f"Recurring off-screen gaze: {count('suspicious_eye_movement')}.")
    if contextual_events:
        parts.append(f"Contextual (non-scoring) speech/admin events: {len(contextual_events)}.")
    if cleared_findings:
        parts.append(f"Cleared as normal/environment: {len(cleared_findings)} episode(s).")
    if unknown_ledger:
        parts.append(f"Routed to reviewer (ambiguous): {len(unknown_ledger)}.")
    if not parts:
        parts.append("No notable visual observations." if usable else "Video analysis produced no usable windows.")

    unique_chunks = {c.chunk_id for w in bundle.windows for c in w.overlapping_chunks}
    return VideoDerivationResult(
        findings=[*flagged, *cleared_findings],
        critical_moments=critical_moments,
        cleared_ledger=cleared_ledger,
        unknown_ledger=unknown_ledger,
        contextual_events=contextual_events,
        video_observation=" ".join(parts),
        video_chunks_sampled=len(unique_chunks),
        total_video_chunks=bundle.total_video_chunks,
    )
