"""Video finding classifiers — port of videoFindingsDerivation.ts classifiers."""

from __future__ import annotations

from math import log10

from ..contracts.perception import PerceptionBundle, PerceptionObservation
from .contracts import ClearedItem, ContextualEvent, CriticalMoment, EvidenceFinding, SpeechProof, UnknownItem
from .video_derivation_core import (
    ADMIN_SPEECH_CLASSES,
    CONTEXT_SPEECH_CLASSES,
    DERIVATION_CONFIG,
    Episode,
    INTEGRITY_SPEECH_CLASSES,
    OFF_ZONES,
    SEV_WEIGHT,
    _capture_issue,
    _mk_finding,
    _sec,
    build_episodes,
    has_admin_speech,
    has_integrity_speech,
    has_only_cleared_speech,
    pick_proof_anchors,
)


def coalesce_cleared_items(items: list[ClearedItem]) -> list[ClearedItem]:
    if not items:
        return []
    max_gap_ms = DERIVATION_CONFIG["EPISODE_MAX_GAP_SEC"] * 1000
    sorted_items = sorted(items, key=lambda i: i.timestamp_window_ms[0])
    out: list[dict] = []
    for item in sorted_items:
        prev = out[-1] if out else None
        if (
            prev
            and prev["event_type"] == item.event_type
            and prev["reason"] == item.reason
            and item.timestamp_window_ms[0] - prev["timestamp_window_ms"][1] <= max_gap_ms
        ):
            t0 = min(prev["timestamp_window_ms"][0], item.timestamp_window_ms[0])
            t1 = max(prev["timestamp_window_ms"][1], item.timestamp_window_ms[1])
            prev["window_ids"] = list(dict.fromkeys([*prev["window_ids"], *item.window_ids]))
            prev["timestamp_window_ms"] = (t0, t1)
            prev["duration_sum_sec"] += item.duration_sec
            prev["duration_sec"] = max(1, round(prev["duration_sum_sec"]))
            prev["occurrence_count"] += 1
        else:
            out.append(
                {
                    **item.model_dump(),
                    "occurrence_count": 1,
                    "duration_sum_sec": item.duration_sec,
                }
            )
    return [ClearedItem(**{k: v for k, v in d.items() if k != "duration_sum_sec"}) for d in out]


def classify_face(
    live: list[PerceptionObservation], bundle: PerceptionBundle
) -> dict:
    findings: list[EvidenceFinding] = []
    cleared: list[ClearedItem] = []
    unknowns: list[UnknownItem] = []
    episodes = build_episodes(live, lambda o: o.identity.face_present == "no")
    for i, ep in enumerate(episodes):
        next_ep = episodes[i + 1] if i + 1 < len(episodes) else None
        gap_ms_to_next = next_ep.t0 - ep.t1 if next_ep else None
        dur_sec = _sec(ep.duration_ms)
        span = (ep.t0, ep.t1)
        mostly_capture = sum(1 for o in ep.observations if _capture_issue(o)) >= len(ep.observations) / 2
        explicit_seat = any(o.body.left_seat == "yes" for o in ep.observations)
        if mostly_capture and not explicit_seat:
            cleared.append(
                ClearedItem(
                    event_type="no_candidate",
                    reason="capture_quality_issue",
                    note=f"Face not detected for ~{dur_sec}s during low/blocked capture — video-quality gap.",
                    window_ids=ep.window_ids,
                    timestamp_window_ms=span,
                    duration_sec=dur_sec,
                )
            )
            continue
        if dur_sec < DERIVATION_CONFIG["face"]["IGNORE_BELOW_SEC"]:
            cleared.append(
                ClearedItem(
                    event_type="no_candidate",
                    reason="brief_look_away",
                    note=f"Brief ~{dur_sec}s face drop-out — below reporting threshold.",
                    window_ids=ep.window_ids,
                    timestamp_window_ms=span,
                    duration_sec=dur_sec,
                )
            )
            continue
        left_seat = any(o.body.left_seat == "yes" for o in ep.observations)
        if left_seat:
            if dur_sec < DERIVATION_CONFIG["seatDeparture"]["MIN_SEC"]:
                cleared.append(
                    ClearedItem(
                        event_type="no_candidate",
                        reason="brief_seat_movement",
                        note=f"Brief seat movement ~{dur_sec}s.",
                        window_ids=ep.window_ids,
                        timestamp_window_ms=span,
                        duration_sec=dur_sec,
                    )
                )
                continue
            body_still = any(
                (isinstance(o.people.people_in_frame, int) and o.people.people_in_frame > 0)
                or o.environment.framing in {"partial", "off_center"}
                for o in ep.observations
            )
            fully_left = not body_still or all(
                o.people.people_in_frame == 0 or o.environment.framing == "obscured"
                for o in ep.observations
            )
            if fully_left:
                findings.append(
                    _mk_finding(
                        "left_examination_area",
                        ep,
                        bundle,
                        attribution="candidate",
                        severity="high",
                        evidence_strength="strong" if dur_sec >= DERIVATION_CONFIG["face"]["SUSTAINED_SEC"] else "moderate",
                        verdict="flagged",
                        reasoning=f"Candidate left examination area entirely for ~{dur_sec}s.",
                        gap_ms_to_next=gap_ms_to_next,
                    )
                )
            else:
                sev = "high" if dur_sec >= DERIVATION_CONFIG["seatDeparture"]["BODY_PRESENT_SUSTAINED_SEC"] else "medium"
                findings.append(
                    _mk_finding(
                        "left_seat_body_present",
                        ep,
                        bundle,
                        attribution="candidate",
                        severity=sev,
                        evidence_strength="moderate" if sev == "high" else "thin",
                        verdict="flagged",
                        reasoning=f"Candidate left seat ~{dur_sec}s with body still in frame.",
                        gap_ms_to_next=gap_ms_to_next,
                    )
                )
            continue
        severity = "low"
        strength = "thin"
        if dur_sec >= DERIVATION_CONFIG["face"]["PROLONGED_SEC"]:
            severity, strength = "high", "strong"
        elif dur_sec >= DERIVATION_CONFIG["face"]["SUSTAINED_SEC"]:
            severity, strength = "medium", "moderate"
        findings.append(
            _mk_finding(
                "no_candidate",
                ep,
                bundle,
                attribution="candidate",
                severity=severity,
                evidence_strength=strength,
                verdict="flagged",
                reasoning=f"Face absent ~{dur_sec}s; seat not vacated.",
                gap_ms_to_next=gap_ms_to_next,
            )
        )
    return {"findings": findings, "cleared": cleared, "unknowns": unknowns}


def classify_second_person(live: list[PerceptionObservation], bundle: PerceptionBundle) -> dict:
    findings: list[EvidenceFinding] = []
    cleared: list[ClearedItem] = []
    unknowns: list[UnknownItem] = []
    episodes = build_episodes(live, lambda o: o.people.second_person_visible == "yes")
    for i, ep in enumerate(episodes):
        next_ep = episodes[i + 1] if i + 1 < len(episodes) else None
        gap_ms_to_next = next_ep.t0 - ep.t1 if next_ep else None
        dur_sec = _sec(ep.duration_ms)
        span = (ep.t0, ep.t1)
        if dur_sec < DERIVATION_CONFIG["secondPerson"]["MIN_SEC"]:
            cleared.append(
                ClearedItem(
                    event_type="multiple_faces",
                    reason="transient_second_person",
                    note=f"Brief second person ~{dur_sec}s.",
                    window_ids=ep.window_ids,
                    timestamp_window_ms=span,
                    duration_sec=dur_sec,
                )
            )
            continue
        interacting = any(
            o.people.second_person_interacting == "yes" or o.people.second_person_activity == "speaking_to_candidate"
            for o in ep.observations
        )
        candidate_responded = any(o.people.candidate_responding_to_second_person == "yes" for o in ep.observations)
        just_passing = all(
            o.people.second_person_activity in {"passing_by", "idle", "UNKNOWN"} for o in ep.observations
        )
        looks_like = {o.people.second_person_looks_like for o in ep.observations}
        if looks_like <= {"reflection", "poster_or_photo", "on_screen_video"}:
            cleared.append(
                ClearedItem(
                    event_type="multiple_faces",
                    reason="not_a_live_person",
                    note="Second face classified as reflection/poster — not a person.",
                    window_ids=ep.window_ids,
                    timestamp_window_ms=span,
                    duration_sec=dur_sec,
                )
            )
            continue
        invigilator = any(o.people.second_person_role_cue == "invigilator_or_staff" for o in ep.observations)
        if (invigilator or has_admin_speech(ep.observations)) and not has_integrity_speech(ep.observations):
            cleared.append(
                ClearedItem(
                    event_type="multiple_faces",
                    reason="invigilator_or_staff_presence" if invigilator else "invigilator_audio_presence",
                    note=f"Second person ~{dur_sec}s with admin/non-cheating audio.",
                    window_ids=ep.window_ids,
                    timestamp_window_ms=span,
                    duration_sec=dur_sec,
                )
            )
            continue
        if interacting:
            gaze_diverted = any(
                o.people.candidate_responding_to_second_person == "yes"
                or o.interaction.reaching_outside_frame == "yes"
                or (
                    o.people.second_person_interacting == "yes"
                    and o.attention.gaze_direction not in {"screen", "down", "UNKNOWN"}
                )
                for o in ep.observations
            ) or has_integrity_speech(ep.observations)
            if not gaze_diverted:
                cleared.append(
                    ClearedItem(
                        event_type="multiple_faces",
                        reason="second_person_interacting_candidate_gaze_on_work",
                        note="Second person active but candidate gaze on work.",
                        window_ids=ep.window_ids,
                        timestamp_window_ms=span,
                        duration_sec=dur_sec,
                    )
                )
                continue
            findings.append(
                _mk_finding(
                    "external_help",
                    ep,
                    bundle,
                    attribution="other_person",
                    severity="high",
                    evidence_strength="strong" if candidate_responded else "moderate",
                    verdict="flagged",
                    reasoning=f"Second person interacting ~{dur_sec}s.",
                    candidate_gaze_diverted_toward_person=True,
                    gap_ms_to_next=gap_ms_to_next,
                )
            )
            continue
        if just_passing:
            cleared.append(
                ClearedItem(
                    event_type="multiple_faces",
                    reason="background_person_passing_by",
                    note=f"Background person passing ~{dur_sec}s.",
                    window_ids=ep.window_ids,
                    timestamp_window_ms=span,
                    duration_sec=dur_sec,
                )
            )
            continue
        adjacent = any(o.people.second_person_position == "adjacent" for o in ep.observations)
        if adjacent and dur_sec >= DERIVATION_CONFIG["secondPerson"]["ADJACENT_SUSTAINED_SEC"]:
            findings.append(
                _mk_finding(
                    "multiple_faces",
                    ep,
                    bundle,
                    attribution="environment",
                    severity="medium",
                    evidence_strength="thin",
                    verdict="flagged",
                    reasoning=f"Second person adjacent ~{dur_sec}s without interaction.",
                    gap_ms_to_next=gap_ms_to_next,
                )
            )
    return {"findings": findings, "cleared": cleared, "unknowns": unknowns}


def _cluster_gaze_segments(eps: list[Episode], merge_gap_ms: int) -> list[Episode]:
    if not eps:
        return []
    sorted_eps = sorted(eps, key=lambda e: (e.t0, e.t1))
    clusters: list[list[Episode]] = [[sorted_eps[0]]]
    for ep in sorted_eps[1:]:
        if ep.t0 - clusters[-1][-1].t1 <= merge_gap_ms:
            clusters[-1].append(ep)
        else:
            clusters.append([ep])
    out: list[Episode] = []
    for group in clusters:
        t0 = min(e.t0 for e in group)
        t1 = max(e.t1 for e in group)
        total_gaze = sum(e.duration_ms for e in group)
        out.append(
            Episode(
                observations=[o for e in group for o in e.observations],
                window_ids=[wid for e in group for wid in e.window_ids],
                t0=t0,
                t1=t1,
                duration_ms=min(total_gaze, t1 - t0),
                avg_confidence=sum(e.avg_confidence for e in group) / len(group),
            )
        )
    return out


def classify_gaze(live: list[PerceptionObservation], bundle: PerceptionBundle) -> dict:
    findings: list[EvidenceFinding] = []
    cleared: list[ClearedItem] = []
    episodes = build_episodes(
        live,
        lambda o: o.attention.gaze_direction in OFF_ZONES and o.attention.gaze_target != "secondary_monitor",
    )
    by_dir: dict[str, list[Episode]] = {}
    for ep in episodes:
        dirs = [o.attention.gaze_direction for o in ep.observations]
        dominant = max(set(dirs), key=dirs.count)
        by_dir.setdefault(dominant, []).append(ep)
    model_recurring = any(
        o.attention.repeated_gaze_pattern not in {"none", "UNKNOWN"} for o in live
    )
    merge_gap_ms = DERIVATION_CONFIG["gaze"]["SEGMENT_MERGE_GAP_SEC"] * 1000
    for direction, eps in by_dir.items():
        # Every gaze segment goes to the model as "provisional". Clearing sparse
        # glances here on an episode count decided the question before anyone
        # looked at how long the candidate actually looked away, or at what.
        # The counts and durations below are reported so the model can weigh
        # them; they no longer gate emission.
        segments = _cluster_gaze_segments(eps, merge_gap_ms)
        session_cumulative_sec = _sec(sum(e.duration_ms for e in eps))
        for segment in segments:
            segment_sec = _sec(segment.duration_ms)
            gaze_severity = (
                "high" if len(eps) >= 8 or session_cumulative_sec >= 600
                else "medium" if len(eps) >= 5 or session_cumulative_sec >= 300
                else "low"
            )
            gaze_evidence = (
                "strong" if len(eps) >= 8 or session_cumulative_sec >= 600
                else "moderate" if len(eps) >= 5 or session_cumulative_sec >= 300
                else "thin"
            )
            findings.append(
                _mk_finding(
                    "suspicious_eye_movement",
                    segment,
                    bundle,
                    attribution="candidate",
                    severity=gaze_severity,
                    evidence_strength=gaze_evidence,
                    verdict="provisional",
                    reasoning=(
                        f"Off-screen gaze toward '{direction}' ~{segment_sec}s; "
                        f"session {len(eps)} episode(s), ~{session_cumulative_sec}s cumulative"
                        + (", recurring pattern reported" if model_recurring else "")
                        + "."
                    ),
                )
            )
    return {"findings": findings, "cleared": cleared}


def classify_phone(live: list[PerceptionObservation], bundle: PerceptionBundle) -> dict:
    findings: list[EvidenceFinding] = []
    cleared: list[ClearedItem] = []
    episodes = build_episodes(live, lambda o: o.objects.phone_visible == "yes")
    for i, ep in enumerate(episodes):
        next_ep = episodes[i + 1] if i + 1 < len(episodes) else None
        gap_ms_to_next = next_ep.t0 - ep.t1 if next_ep else None
        dur_sec = _sec(ep.duration_ms)
        span = (ep.t0, ep.t1)
        if dur_sec < DERIVATION_CONFIG["phone"]["MIN_SEC"]:
            continue
        in_hand = any(
            o.hands.object_in_hand == "phone" or o.hands.hand_location == "phone" for o in ep.observations
        )
        if not in_hand:
            cleared.append(
                ClearedItem(
                    event_type="phone_usage",
                    reason="phone_visible_not_handled",
                    note=f"Phone visible ~{dur_sec}s but not held by candidate.",
                    window_ids=ep.window_ids,
                    timestamp_window_ms=span,
                    duration_sec=dur_sec,
                )
            )
            continue
        findings.append(
            _mk_finding(
                "phone_usage",
                ep,
                bundle,
                attribution="candidate",
                severity="high" if dur_sec >= DERIVATION_CONFIG["face"]["SUSTAINED_SEC"] else "medium",
                evidence_strength="moderate",
                verdict="flagged",
                reasoning=f"Phone held ~{dur_sec}s.",
                gap_ms_to_next=gap_ms_to_next,
            )
        )
    return {"findings": findings, "cleared": cleared}


def _describe_audio_wearable(obs: list[PerceptionObservation]) -> str:
    parts: list[str] = []
    if any(o.objects.headphones_visible == "yes" for o in obs):
        parts.append("headphones")
    if any(o.objects.earphone_visible == "yes" for o in obs):
        parts.append("earphones")
    return " + ".join(parts) if parts else "audio wearable"


def _has_audio_wearable_contacting(obs: list[PerceptionObservation]) -> bool:
    if has_integrity_speech(obs):
        return True
    speech_windows = [o for o in obs if o.audio.speech_present == "yes"]
    if not speech_windows:
        return False
    lips_mismatch = any(
        o.audio.speech_overlap_with_lips == "no" or o.interaction.lip_movement == "none" for o in speech_windows
    )
    off_camera = any(o.audio.speech_source in {"off_camera", "device_playback"} for o in speech_windows)
    device_call = any(o.audio.device_sounds == "call_ringtone" for o in obs)
    return lips_mismatch or off_camera or device_call


def classify_audio_wearables(live: list[PerceptionObservation], bundle: PerceptionBundle) -> dict:
    findings: list[EvidenceFinding] = []
    cleared: list[ClearedItem] = []
    episodes = build_episodes(
        live, lambda o: o.objects.headphones_visible == "yes" or o.objects.earphone_visible == "yes"
    )
    for i, ep in enumerate(episodes):
        next_ep = episodes[i + 1] if i + 1 < len(episodes) else None
        gap_ms_to_next = next_ep.t0 - ep.t1 if next_ep else None
        dur_sec = _sec(ep.duration_ms)
        span = (ep.t0, ep.t1)
        if dur_sec < DERIVATION_CONFIG["audioWearable"]["MIN_SEC"]:
            continue
        device_desc = _describe_audio_wearable(ep.observations)
        contacting = _has_audio_wearable_contacting(ep.observations)
        if not contacting:
            cleared.append(
                ClearedItem(
                    event_type="audio_wearable_assisted_comms",
                    reason="audio_wearable_visible_not_communicating",
                    note=f"{device_desc} visible ~{dur_sec}s — presence is not contacting someone.",
                    window_ids=ep.window_ids,
                    timestamp_window_ms=span,
                    duration_sec=dur_sec,
                )
            )
            continue
        findings.append(
            _mk_finding(
                "audio_wearable_assisted_comms",
                ep,
                bundle,
                attribution="candidate",
                severity="high" if dur_sec >= DERIVATION_CONFIG["face"]["SUSTAINED_SEC"] else "medium",
                evidence_strength="moderate",
                verdict="flagged",
                reasoning=f"{device_desc} visible ~{dur_sec}s with contacting/remote-audio evidence.",
                gap_ms_to_next=gap_ms_to_next,
            )
        )
    return {"findings": findings, "cleared": cleared}


def _build_speech_proof(ep: Episode) -> SpeechProof | None:
    summaries = []
    seen = set()
    for o in ep.observations:
        s = (o.audio.conversation_summary_en or "").strip()
        if s and s not in seen:
            seen.add(s)
            summaries.append(s)
    joined = " ".join(summaries)
    if not joined:
        return None
    max_chars = DERIVATION_CONFIG["speech"]["SUMMARY_MAX_CHARS"]
    if len(joined) > max_chars:
        joined = joined[: max_chars - 1] + "…"
    langs = [o.audio.speech_language for o in ep.observations if o.audio.speech_language != "UNKNOWN"]
    lang = max(set(langs), key=langs.count) if langs else "UNKNOWN"
    classes = [o.audio.speech_content_class for o in ep.observations if o.audio.speech_content_class not in {"UNKNOWN", "silence"}]
    content_class = max(set(classes), key=classes.count) if classes else "unclear"
    return SpeechProof(
        conversation_summary_en=joined,
        speech_language=lang,
        code_mixing=any(o.audio.code_mixing == "yes" for o in ep.observations),
        speech_content_class=content_class,
        clip_start_ms=ep.t0,
        clip_end_ms=ep.t1,
    )


def classify_speech(live: list[PerceptionObservation], bundle: PerceptionBundle) -> dict:
    findings: list[EvidenceFinding] = []
    cleared: list[ClearedItem] = []
    unknowns: list[UnknownItem] = []
    contextual: list[ContextualEvent] = []
    episodes = build_episodes(live, lambda o: o.audio.speech_present == "yes")
    buffer_ms = DERIVATION_CONFIG["critical"]["CLIP_BUFFER_SEC"] * 1000
    for i, ep in enumerate(episodes):
        next_ep = episodes[i + 1] if i + 1 < len(episodes) else None
        gap_ms_to_next = next_ep.t0 - ep.t1 if next_ep else None
        dur_sec = _sec(ep.duration_ms)
        span = (ep.t0, ep.t1)
        speech_proof = _build_speech_proof(ep)
        content_class = speech_proof.speech_content_class if speech_proof else "unclear"
        if dur_sec < DERIVATION_CONFIG["speech"]["MIN_SEC"]:
            continue
        if not speech_proof:
            unknowns.append(
                UnknownItem(
                    event_type="speech_episode",
                    reason="speech_without_summary",
                    note=f"Speech ~{dur_sec}s without summary.",
                    window_ids=ep.window_ids,
                    timestamp_window_ms=span,
                )
            )
            continue
        if content_class in INTEGRITY_SPEECH_CLASSES:
            findings.append(
                _mk_finding(
                    "discussing_solution_audio",
                    ep,
                    bundle,
                    attribution="candidate" if any(o.audio.speech_source == "candidate" for o in ep.observations) else "other_person",
                    severity="high",
                    evidence_strength="strong" if content_class in {"receiving_dictation", "reciting_answer_choices"} else "moderate",
                    verdict="flagged",
                    reasoning=f"Speech classified as {content_class} ~{dur_sec}s.",
                    speech_proof=speech_proof,
                    gap_ms_to_next=gap_ms_to_next,
                )
            )
            continue
        if content_class in CONTEXT_SPEECH_CLASSES:
            ctx_type = {
                "invigilator_or_admin": "invigilator_interaction",
                "technical_exam_help": "technical_exam_help",
                "self_talk_or_thinking": "self_talk",
                "reading_question": "self_talk",
                "casual_non_exam": "casual_non_exam",
            }.get(content_class, "exam_hall_ambient")
            contextual.append(
                ContextualEvent(
                    event_type=ctx_type,
                    timestamp_window_ms=span,
                    one_line_why=f"{content_class}: {speech_proof.conversation_summary_en}",
                    clip_start_ms=max(0, ep.t0 - buffer_ms),
                    clip_end_ms=ep.t1 + buffer_ms,
                    speech_proof=speech_proof,
                )
            )
            cleared.append(
                ClearedItem(
                    event_type=ctx_type,
                    reason=f"speech_{content_class}",
                    note=f"Cleared as {content_class} ~{dur_sec}s.",
                    window_ids=ep.window_ids,
                    timestamp_window_ms=span,
                    duration_sec=dur_sec,
                    speech_proof=speech_proof,
                )
            )
    return {"findings": findings, "cleared": cleared, "unknowns": unknowns, "contextual": contextual}


def rank_critical_moments(
    flagged: list[EvidenceFinding],
    bundle: PerceptionBundle,
    section_labels: dict[str, dict[str, str]] | None = None,
) -> list[CriticalMoment]:
    win_by_id = {w.window_id: w for w in bundle.windows}
    scored = []
    for f in flagged:
        t0, t1 = f.timestamp_window_ms
        dur_sec = max(1, _sec(t1 - t0))
        sev = SEV_WEIGHT.get(f.severity or "low", 1)
        strength_bonus = {"strong": 1.5, "moderate": 1.2}.get(f.evidence_strength or "thin", 1.0)
        interaction_bonus = 1.5 if f.event_type == "external_help" else 1.0
        score = sev * strength_bonus * interaction_bonus * (1 + log10(dur_sec))
        scored.append((f, score, dur_sec))
    scored.sort(key=lambda x: x[1], reverse=True)
    out: list[CriticalMoment] = []
    total_sec = 0
    buffer_ms = DERIVATION_CONFIG["critical"]["CLIP_BUFFER_SEC"] * 1000
    for f, score, dur_sec in scored:
        if len(out) >= DERIVATION_CONFIG["critical"]["TOP_K"]:
            break
        if total_sec + dur_sec > DERIVATION_CONFIG["critical"]["MAX_TOTAL_SEC"]:
            continue
        t0, t1 = f.timestamp_window_ms
        out.append(
            CriticalMoment(
                rank=len(out) + 1,
                event_type=f.event_type,
                timestamp_window_ms=(t0, t1),
                clip_start_ms=max(0, t0 - buffer_ms),
                clip_end_ms=min(bundle.duration_ms, t1 + buffer_ms),
                duration_sec=dur_sec,
                section_label=None,
                score=round(score, 2),
                one_line_why=f.reasoning,
            )
        )
        total_sec += dur_sec
    return out
