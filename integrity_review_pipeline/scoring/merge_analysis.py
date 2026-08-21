"""Track B merge and composite scoring (TS mergeAnalysis.ts parity; cohort nudge excluded)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal

from ..duck_helpers import attr

CorroborationStatus = Literal["corroborated", "track_b_only", "track_a_only"]
MergedVerdict = Literal["confirmed", "ai_detected", "unverified_signal", "overturned", "cleared"]
RiskBand = Literal["Clear", "Needs Review", "Likely Violation"]

TIME_TOLERANCE_MS = 90_000
SEVERITY_POINTS = {"high": 32, "medium": 18, "low": 9}

EVENT_TYPE_TO_RULE_FAMILY: dict[str, list[str]] = {
    "phone_usage": ["video.prohibited_object"],
    "reading_notes": ["video.prohibited_object"],
    "external_help": ["video.multiple_faces"],
    "multiple_faces": ["video.multiple_faces"],
    "suspicious_eye_movement": ["video.suspicious_eye_movement"],
    "no_candidate": ["video.multiple_faces", "video.suspicious_eye_movement"],
    "left_examination_area": ["video.multiple_faces", "video.suspicious_eye_movement"],
    "left_seat_body_present": ["video.multiple_faces", "video.suspicious_eye_movement"],
    "prolonged_absence": ["video.suspicious_eye_movement"],
    "external_paste": [],
    "mass_paste": [],
    "screen_external_paste": [],
    "external_resource_open": [],
    "ai_assistant_ui_visible": [],
    "secondary_workspace_visible": [],
    "fullscreen_exam_lost": [],
    "answer_appeared": [],
    "typing_restart": [],
    "rapid_typing": [],
    "bulk_delete": [],
}


@dataclass(slots=True)
class MergedFinding:
    id: str
    corroboration_status: CorroborationStatus
    merged_verdict: MergedVerdict
    event_type: str
    attribution: Literal["candidate", "environment", "unclear"]
    phase: Literal["setup_verification", "live_exam", "unknown"]
    severity: Literal["high", "medium", "low"]
    reasoning: str
    score_weight: float
    rule_id: str | None = None
    track_a_finding: Any | None = None
    track_b_finding: Any | None = None
    proctoring_corroboration: dict[str, object] | None = None


@dataclass(slots=True)
class ScoreBreakdown:
    integrity_score: int
    cohort_nudge: int
    total: int
    band: RiskBand
    confidence: Literal["high", "medium", "low"]
    driving_finding_count: int
    data_quality_capped: bool
    data_quality_note: str | None = None


@dataclass(slots=True)
class ReconcileResult:
    merged_findings: list[MergedFinding]
    score: ScoreBreakdown


def _finding_attr(finding: Any, *names: str, default: Any = None) -> Any:
    return attr(finding, *names, default=default)


def _finding_window(finding: Any) -> tuple[int, int]:
    window = _finding_attr(finding, "timestamp_window_ms", "timestampWindowMs", default=(0, 0))
    return int(window[0]), int(window[1])


def _centre(window: tuple[int, int]) -> float:
    return (window[0] + window[1]) / 2


def _track_a_centre_ms(finding: Any) -> float:
    proof = _finding_attr(finding, "proof_clip", "proofClip")
    if proof is not None:
        start = _finding_attr(proof, "window_start_ms", "windowStartMs")
        end = _finding_attr(proof, "window_end_ms", "windowEndMs")
        if start is not None and end is not None:
            return (int(start) + int(end)) / 2
    return 0.0


def _track_b_centre_ms(finding: Any) -> float:
    window = _finding_window(finding)
    return _centre(window)


def _rule_family(event_type: str) -> list[str]:
    return EVENT_TYPE_TO_RULE_FAMILY.get(event_type, [])


def _is_compatible(rule_id: str, event_type: str) -> bool:
    families = _rule_family(event_type)
    if not families:
        return False
    return any(rule_id == family or rule_id.startswith(family) for family in families)


def _is_time_match(track_a: Any, track_b: Any) -> bool:
    a_centre = _track_a_centre_ms(track_a)
    b_centre = _track_b_centre_ms(track_b)
    if a_centre == 0 or b_centre == 0:
        return True
    return abs(a_centre - b_centre) <= TIME_TOLERANCE_MS


def _reconcile_attribution(track_a: Any, track_b: Any) -> Literal["candidate", "environment", "unclear"]:
    b_raw = str(_finding_attr(track_b, "attribution", default="unclear"))
    b: Literal["candidate", "environment", "unclear"] = (
        "environment" if b_raw == "other_person" else b_raw  # type: ignore[assignment]
    )
    if b == "unclear":
        a_raw = str(_finding_attr(track_a, "attribution", default="unclear"))
        if a_raw in {"candidate", "environment", "unclear"}:
            return a_raw  # type: ignore[return-value]
    return b if b in {"candidate", "environment", "unclear"} else "unclear"


def _verdict_from_corroborated(
    track_a: Any,
    attribution: Literal["candidate", "environment", "unclear"],
    phase: str,
    track_b: Any | None,
) -> MergedVerdict:
    if attribution == "environment" or phase == "setup_verification":
        return "cleared"
    verdict = str(_finding_attr(track_a, "verdict", default=""))
    if verdict == "sustain":
        return "confirmed"
    if verdict == "needs_review" and _finding_attr(track_b, "evidence_strength", "evidenceStrength") == "strong":
        return "confirmed"
    if verdict == "needs_review":
        return "ai_detected"
    return "cleared"


def _verdict_from_track_b_only(
    finding: Any,
    attribution: Literal["candidate", "environment", "unclear"],
) -> MergedVerdict:
    event_type = str(_finding_attr(finding, "event_type", "eventType", default=""))
    phase = str(_finding_attr(finding, "phase", default="unknown"))
    if event_type == "external_help":
        return "ai_detected"
    if attribution == "environment" or phase == "setup_verification":
        return "cleared"
    strength = _finding_attr(finding, "evidence_strength", "evidenceStrength")
    severity = str(_finding_attr(finding, "severity", default="low"))
    if (
        strength == "strong"
        and severity == "high"
        and attribution == "candidate"
        and phase == "live_exam"
    ):
        return "confirmed"
    return "ai_detected"


def _compute_finding_weight(finding: MergedFinding) -> float:
    if finding.merged_verdict in {"cleared", "overturned"}:
        return 0.0
    if finding.attribution == "environment" or finding.phase == "setup_verification":
        return 0.0
    if finding.corroboration_status == "track_a_only":
        return 0.0
    if finding.merged_verdict == "unverified_signal":
        return 0.0
    if finding.corroboration_status == "corroborated" and finding.merged_verdict == "confirmed":
        return 1.0
    if finding.corroboration_status == "corroborated" and finding.merged_verdict == "ai_detected":
        return 0.6
    if finding.corroboration_status == "track_b_only" and finding.merged_verdict == "confirmed":
        return 1.0
    if finding.corroboration_status == "track_b_only" and finding.merged_verdict == "ai_detected":
        return 0.85
    return 0.0


def _build_merged_finding(**kwargs: Any) -> MergedFinding:
    finding = MergedFinding(score_weight=0.0, **kwargs)
    finding.score_weight = _compute_finding_weight(finding)
    return finding


def compute_composite_score(
    merged_findings: list[MergedFinding],
    *,
    video_available: bool = True,
    keystroke_available: bool = True,
) -> ScoreBreakdown:
    integrity_score = 0.0
    driving = 0
    for finding in merged_findings:
        if finding.score_weight <= 0:
            continue
        integrity_score += SEVERITY_POINTS.get(finding.severity, 9) * finding.score_weight
        driving += 1
    integrity_score = min(round(integrity_score), 90)
    cohort_nudge = 0
    total = max(0, min(100, integrity_score + cohort_nudge))
    band: RiskBand = (
        "Likely Violation" if total >= 56 else "Needs Review" if total >= 26 else "Clear"
    )
    confidence: Literal["high", "medium", "low"] = "high"
    data_quality_capped = False
    data_quality_note: str | None = None
    if not video_available or not keystroke_available:
        confidence = "medium"
    if not video_available and not keystroke_available:
        confidence = "low"
        data_quality_capped = True
        data_quality_note = (
            "Both video and keystroke data unavailable — analysis based on detection signals only. "
            "Manual review recommended."
        )
    elif not video_available:
        data_quality_note = (
            "Video not available — analysis based on keystroke data and proctoring signals only."
        )
    if driving == 0 and total > 0:
        confidence = "medium"
    return ScoreBreakdown(
        integrity_score=int(integrity_score),
        cohort_nudge=cohort_nudge,
        total=total,
        band=band,
        confidence=confidence,
        driving_finding_count=driving,
        data_quality_capped=data_quality_capped,
        data_quality_note=data_quality_note,
    )


def reconcile_findings(
    track_a_findings: list[Any],
    track_b_findings: list[Any],
    *,
    video_available: bool = True,
    keystroke_available: bool = True,
) -> ReconcileResult:
    track_b_flagged = [
        f
        for f in track_b_findings
        if _finding_attr(f, "verdict", default="flagged") == "flagged"
        and str(_finding_attr(f, "event_type", "eventType", default="")) != "ok"
    ]
    track_b_cleared = [
        f
        for f in track_b_findings
        if _finding_attr(f, "verdict", default="") == "cleared"
        or str(_finding_attr(f, "event_type", "eventType", default="")) == "ok"
    ]

    used_track_b_ids: set[str] = set()
    merged: list[MergedFinding] = []
    counter = 0

    def next_id() -> str:
        nonlocal counter
        counter += 1
        return f"mf_{counter}"

    track_a_active = [
        f
        for f in track_a_findings
        if _finding_attr(f, "verdict", default="") in {"sustain", "needs_review"}
    ]

    for track_a in track_a_active:
        rule_id = str(_finding_attr(track_a, "rule_id", "ruleId", default=""))
        matches = [
            bf
            for bf in track_b_flagged
            if str(_finding_attr(bf, "id", default="")) not in used_track_b_ids
            and _is_compatible(rule_id, str(_finding_attr(bf, "event_type", "eventType", default="")))
            and _is_time_match(track_a, bf)
        ]
        matches.sort(
            key=lambda bf: abs(_track_a_centre_ms(track_a) - _track_b_centre_ms(bf))
        )
        best = matches[0] if matches else None
        if best is not None:
            used_track_b_ids.add(str(_finding_attr(best, "id")))
            attribution = _reconcile_attribution(track_a, best)
            phase = str(_finding_attr(best, "phase", default="unknown"))
            merged.append(
                _build_merged_finding(
                    id=next_id(),
                    corroboration_status="corroborated",
                    merged_verdict=_verdict_from_corroborated(track_a, attribution, phase, best),
                    event_type=str(_finding_attr(best, "event_type", "eventType", default="unknown")),
                    attribution=attribution,
                    phase=phase,  # type: ignore[arg-type]
                    severity=str(_finding_attr(best, "severity", default="low")),  # type: ignore[arg-type]
                    reasoning=(
                        f"Corroborated: {_finding_attr(track_a, 'reasoning', default='')} | "
                        f"Track B: {_finding_attr(best, 'reasoning', default='')}"
                    ),
                    rule_id=rule_id,
                    track_a_finding=track_a,
                    track_b_finding=best,
                )
            )
        else:
            attribution = str(_finding_attr(track_a, "attribution", default="unclear"))
            if attribution not in {"candidate", "environment", "unclear"}:
                attribution = "unclear"
            merged.append(
                _build_merged_finding(
                    id=next_id(),
                    corroboration_status="track_a_only",
                    merged_verdict="unverified_signal",
                    event_type=rule_id,
                    attribution=attribution,  # type: ignore[arg-type]
                    phase="unknown",
                    severity="low",
                    reasoning=(
                        "Detection signal not corroborated by independent review. "
                        f"{_finding_attr(track_a, 'reasoning', default='')}"
                    ),
                    rule_id=rule_id,
                    track_a_finding=track_a,
                )
            )

    for finding in track_b_flagged:
        if str(_finding_attr(finding, "id", default="")) in used_track_b_ids:
            continue
        attribution_raw = str(_finding_attr(finding, "attribution", default="unclear"))
        attribution: Literal["candidate", "environment", "unclear"]
        if attribution_raw == "other_person":
            attribution = "environment"
        elif attribution_raw in {"candidate", "environment", "unclear"}:
            attribution = attribution_raw  # type: ignore[assignment]
        else:
            attribution = "unclear"
        merged.append(
            _build_merged_finding(
                id=next_id(),
                corroboration_status="track_b_only",
                merged_verdict=_verdict_from_track_b_only(finding, attribution),
                event_type=str(_finding_attr(finding, "event_type", "eventType", default="unknown")),
                attribution=attribution,
                phase=str(_finding_attr(finding, "phase", default="unknown")),  # type: ignore[arg-type]
                severity=str(_finding_attr(finding, "severity", default="low")),  # type: ignore[arg-type]
                reasoning=str(_finding_attr(finding, "reasoning", default="")),
                track_b_finding=finding,
            )
        )

    for finding in track_b_cleared:
        attribution_raw = str(_finding_attr(finding, "attribution", default="unclear"))
        if attribution_raw == "other_person":
            attribution = "environment"
        elif attribution_raw in {"candidate", "environment", "unclear"}:
            attribution = attribution_raw  # type: ignore[assignment]
        else:
            attribution = "unclear"
        merged.append(
            _build_merged_finding(
                id=next_id(),
                corroboration_status="track_b_only",
                merged_verdict="cleared",
                event_type=str(_finding_attr(finding, "event_type", "eventType", default="unknown")),
                attribution=attribution,
                phase=str(_finding_attr(finding, "phase", default="unknown")),  # type: ignore[arg-type]
                severity=str(_finding_attr(finding, "severity", default="low")),  # type: ignore[arg-type]
                reasoning=f"Independently cleared: {_finding_attr(finding, 'reasoning', default='')}",
                track_b_finding=finding,
            )
        )

    score = compute_composite_score(
        merged,
        video_available=video_available,
        keystroke_available=keystroke_available,
    )
    return ReconcileResult(merged_findings=merged, score=score)


def add_proctoring_corroboration(
    findings: list[MergedFinding],
    violation_events: list[dict[str, object]] | None,
) -> list[MergedFinding]:
    if not violation_events:
        return findings
    updated: list[MergedFinding] = []
    for finding in findings:
        if finding.score_weight <= 0 or finding.track_b_finding is None:
            updated.append(finding)
            continue
        centre = _centre(_finding_window(finding.track_b_finding))
        nearest: dict[str, object] | None = None
        best = TIME_TOLERANCE_MS
        for event in violation_events:
            ts = int(event.get("timestampMs") or event.get("timestamp_ms") or 0)
            dist = abs(ts - centre)
            if dist < best:
                best = dist
                nearest = event
        if nearest is None:
            updated.append(finding)
            continue
        ts = int(nearest.get("timestampMs") or nearest.get("timestamp_ms") or 0)
        updated.append(
            replace(
                finding,
                proctoring_corroboration={
                    "kind": str(nearest.get("kind", "violation")),
                    "timestampMs": ts,
                    "offsetSeconds": round((ts - centre) / 1000, 1),
                    "description": str(nearest.get("description", "")),
                },
            )
        )
    return updated
