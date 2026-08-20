"""Video findings entry points."""

from __future__ import annotations

from ..contracts.perception import PerceptionBundle
from .contracts import EvidenceFindingsResult, VideoObservationWindow
from .video_derivation import DERIVATION_CONFIG, build_episodes, derive_video_findings_from_perception


def derive_video_findings(
    windows_or_bundle: list[VideoObservationWindow] | PerceptionBundle,
) -> EvidenceFindingsResult:
    """Derive video findings from a full perception bundle (preferred) or legacy windows."""
    if isinstance(windows_or_bundle, PerceptionBundle):
        result = derive_video_findings_from_perception(windows_or_bundle)
        flagged = [f for f in result.findings if f.verdict == "flagged"]
        return EvidenceFindingsResult(
            findings=result.findings,
            video_observation=result.video_observation,
            video_available=result.total_video_chunks > 0,
        )

    windows = windows_or_bundle
    usable = [w for w in windows if w.video_available and w.confidence >= DERIVATION_CONFIG["MIN_USABLE_CONFIDENCE"]]
    from .contracts import EvidenceFinding, VideoProof

    findings: list[EvidenceFinding] = []
    index = 0
    for event_type, predicate in (
        ("multiple_faces", lambda w: w.second_person_visible == "yes"),
        ("phone_usage", lambda w: w.phone_visible == "yes"),
        ("suspicious_eye_movement", lambda w: w.gaze_direction == "off_screen"),
        ("no_candidate", lambda w: w.face_present == "no"),
    ):
        hits = sorted((w for w in usable if predicate(w)), key=lambda w: w.start_ms)
        episodes: list[tuple[VideoObservationWindow, VideoObservationWindow]] = []
        for window in hits:
            if episodes and window.start_ms - episodes[-1][1].end_ms <= DERIVATION_CONFIG["EPISODE_MAX_GAP_SEC"] * 1000:
                episodes[-1] = (episodes[-1][0], window)
            else:
                episodes.append((window, window))
        for start, end in episodes:
            duration = end.end_ms - start.start_ms
            if duration < 3000 and event_type != "no_candidate":
                continue
            findings.append(
                EvidenceFinding(
                    id=f"vid_{index}",
                    source="video",
                    event_type=event_type,
                    timestamp_window_ms=(start.start_ms, end.end_ms),
                    attribution="candidate",
                    severity="high" if event_type in {"multiple_faces", "phone_usage"} else "medium",
                    evidence_ref=f"video:{start.window_id}",
                    verdict="flagged",
                    evidence_strength="moderate",
                    reasoning=f"Legacy window episode: {event_type.replace('_', ' ')} for {duration // 1000}s.",
                    video_proof=VideoProof(clip_label=start.window_id, approx_frame_timestamp_ms=start.start_ms),
                )
            )
            index += 1
    observation = (
        f"{len(findings)} flagged video episode(s) from {len(usable)} usable windows."
        if findings
        else "No flagged video activity in usable camera windows."
    )
    return EvidenceFindingsResult(
        findings=findings,
        video_observation=observation,
        video_available=bool(usable),
    )
