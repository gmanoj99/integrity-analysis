    )


# Classifiers are continued in video_derivation_classifiers.py and merged at import time.
from .video_derivation_classifiers import (  # noqa: E402
    classify_audio_wearables,
    classify_face,
    classify_gaze,
    classify_phone,
    classify_second_person,
    classify_speech,
    coalesce_cleared_items,
    rank_critical_moments,
)


def derive_video_findings_from_perception(
    bundle: PerceptionBundle,
    section_labels: dict[str, dict[str, str]] | None = None,
) -> VideoDerivationResult:
    global _id_seq
    _id_seq = 0
    win_by_id = {w.window_id: w for w in bundle.windows}
    live = sorted(
        (o for o in bundle.observations if win_by_id.get(o.window_id) and win_by_id[o.window_id].phase == "live_exam"),
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
    for f in flagged_raw:
        if f.event_type not in {"external_help", "multiple_faces"}:
            flagged.append(f)
            continue
        hits_admin = any(overlaps(f.timestamp_window_ms, s) for s in admin_spans)
        if not hits_admin:
            flagged.append(f)
            continue
        if any(overlaps(f.timestamp_window_ms, s) for s in integrity_spans):
            flagged.append(f)
            continue
        cleared_ledger.append(
            ClearedItem(
                event_type=f.event_type,
                reason="cross_modal_invigilator_audio",
                note=(
                    f"Cleared: {f.event_type} overlapped invigilator/admin speech with no answer-discussion audio."
                ),
                window_ids=[],
                timestamp_window_ms=f.timestamp_window_ms,
                duration_sec=max(1, _sec(f.timestamp_window_ms[1] - f.timestamp_window_ms[0])),
                speech_proof=f.speech_proof,
            )
        )

    unknown_ledger = [*face["unknowns"], *people["unknowns"], *speech["unknowns"]]
    contextual_events = speech["contextual"]

    cleared_findings: list[EvidenceFinding] = []
    for c in coalesce_cleared_items(cleared_ledger):
        chunk = next(
            (
                ch
                for wid in c.window_ids
                for ch in (win_by_id.get(wid) or PerceptionWindow.model_construct()).overlapping_chunks
            ),
            None,
        )
        _id_seq += 1
        cleared_findings.append(
            EvidenceFinding(
                id=f"perc_clr_{_id_seq}",
                source="video",
                event_type=c.event_type,
                timestamp_window_ms=c.timestamp_window_ms,
                attribution="environment",
                phase="live_exam",
                severity="low",
                evidence_ref=f"perception:w={'+'.join(c.window_ids)}",
                verdict="cleared",
                reasoning=c.note,
                evidence_strength="thin",
                data_gaps=c.reason,
                occurrence_count=c.occurrence_count,
                speech_proof=c.speech_proof,
                video_proof=(
                    VideoProof(
                        chunk_id=chunk.chunk_id,
                        chunk_sequence=chunk.sequence,
                        clip_label=c.window_ids[0] if c.window_ids else None,
                        approx_frame_timestamp_ms=c.timestamp_window_ms[0],
                    )
                    if chunk
                    else None
                ),
                video_proof_anchors=pick_proof_anchors(c.window_ids, c.timestamp_window_ms[0], bundle),
            )
        )

    critical_moments = rank_critical_moments(flagged, bundle, section_labels)
    parts: list[str] = []
    counts = lambda t: sum(1 for f in flagged if f.event_type == t)
    if counts("no_candidate"):
        parts.append(f"Face-absence episodes: {counts('no_candidate')}.")
    if counts("phone_usage"):
        parts.append(f"Phone handling: {counts('phone_usage')}.")
    if counts("suspicious_eye_movement"):
        parts.append(f"Recurring off-screen gaze: {counts('suspicious_eye_movement')}.")
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
