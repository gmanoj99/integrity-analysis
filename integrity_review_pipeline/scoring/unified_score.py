"""One recommendation score over deliberation signals and Track B findings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..contracts.deliberation import (
    EvidenceSourceType,
    RecommendationCategory,
    ValidatedSignal,
)
from ..duck_helpers import attr
from ..deliberation.rules import apply_confidence_caps
from .merge_analysis import TIME_TOLERANCE_MS, MergedFinding, reconcile_findings

GAZE_OBSERVATION_DISPLAY_FLOOR_MS = 15_000

SIGNAL_TO_TRACK_B_EVENTS: dict[str, frozenset[str]] = {
    "possible_external_consultation": frozenset(
        {
            "phone_usage",
            "external_help",
            "external_resource_open",
            "ai_assistant_ui_visible",
        }
    ),
    "possible_second_person_involvement": frozenset(
        {"external_help", "multiple_faces"}
    ),
    "unauthorized_reference_usage": frozenset(
        {
            "phone_usage",
            "reading_notes",
            "external_resource_open",
            "secondary_workspace_visible",
            "screen_external_paste",
        }
    ),
    "abnormal_paste_workflow": frozenset(
        {"external_paste", "mass_paste", "screen_external_paste", "answer_appeared"}
    ),
    "suspicious_focus_pattern": frozenset(
        {
            "suspicious_eye_movement",
            "fullscreen_exam_lost",
            "left_examination_area",
            "prolonged_absence",
        }
    ),
    "abnormal_correction_pattern": frozenset({"bulk_delete"}),
    "typing_cadence_mismatch": frozenset({"typing_restart", "rapid_typing"}),
    "inconsistent_interaction_sequence": frozenset(
        {"answer_appeared", "typing_restart", "rapid_typing"}
    ),
    "possible_audio_coaching": frozenset({"external_help"}),
    "possible_remote_dictation": frozenset({"external_help"}),
}


@dataclass(frozen=True, slots=True)
class UnifiedScore:
    category: RecommendationCategory
    confidence: float
    independent_source_types: list[EvidenceSourceType]
    supporting_signal_ids: list[str]
    scored_track_b_ids: list[str]
    corroborated_signal_ids: list[str]
    capture_quality_cap_applied: bool
    informative_content_cap_applied: bool


def _signal_time_range(signal: ValidatedSignal) -> tuple[int, int] | None:
    story = signal.integrity_story
    if story and story.proof_anchors.seek_ms_hints:
        hints = story.proof_anchors.seek_ms_hints
        return min(hints), max(hints)
    starts: list[int] = []
    ends: list[int] = []
    for citation in signal.observations_cited:
        window_id = citation.split(".", 1)[0]
        parts = window_id.split("_")
        if len(parts) >= 3 and parts[0] == "w":
            try:
                starts.append(int(parts[1]))
                ends.append(int(parts[2]))
            except ValueError:
                continue
    if starts:
        return min(starts), max(ends)
    return None


def _overlaps_signal(signal: ValidatedSignal, finding: MergedFinding) -> bool:
    if finding.event_type not in SIGNAL_TO_TRACK_B_EVENTS.get(signal.signal_type, frozenset()):
        return False
    signal_range = _signal_time_range(signal)
    raw = finding.track_b_finding
    if signal_range is None or raw is None:
        return True
    window = attr(raw, "timestamp_window_ms", "timestampWindowMs", default=(0, 0))
    finding_mid = (int(window[0]) + int(window[1])) / 2
    signal_mid = (signal_range[0] + signal_range[1]) / 2
    return abs(finding_mid - signal_mid) <= TIME_TOLERANCE_MS


def _track_b_source(finding: MergedFinding) -> EvidenceSourceType:
    source = str(attr(finding.track_b_finding, "source", default="video"))
    if source == "screen":
        return "screen_observation"
    if source == "keystroke":
        return "machine_fact"
    return "visual_observation"


def _finding_duration_ms(finding: MergedFinding) -> int:
    window = attr(
        finding.track_b_finding,
        "timestamp_window_ms",
        "timestampWindowMs",
        default=(0, 0),
    )
    try:
        return max(0, int(window[1]) - int(window[0]))
    except (TypeError, IndexError, ValueError):
        return 0


def _is_scoreable_track_b(finding: MergedFinding) -> bool:
    if finding.score_weight <= 0:
        return False
    if (
        finding.event_type == "suspicious_eye_movement"
        and _finding_duration_ms(finding) < GAZE_OBSERVATION_DISPLAY_FLOOR_MS
    ):
        return False
    return True


def _track_b_confidence(finding: MergedFinding) -> float:
    strength = str(
        attr(
            finding.track_b_finding,
            "evidence_strength",
            "evidenceStrength",
            default="moderate",
        )
    )
    base = {"strong": 0.90, "moderate": 0.75, "thin": 0.55}.get(strength, 0.65)
    if finding.severity == "high":
        base += 0.05
    elif finding.severity == "low":
        base -= 0.10
    return max(0.0, min(1.0, base))


def compute_unified_score(
    validated_signals: list[ValidatedSignal],
    track_b_findings: list[Any],
    *,
    usable_window_ratio: float,
    informative_content_ratio: float,
) -> UnifiedScore:
    """Merge duplicate evidence and derive the one reviewer-facing recommendation."""

    active = [signal for signal in validated_signals if signal.resolution != "honest"]
    reconciled = reconcile_findings([], track_b_findings)
    scoreable = [
        finding
        for finding in reconciled.merged_findings
        if _is_scoreable_track_b(finding)
    ]

    source_types: set[EvidenceSourceType] = {
        source for signal in active for source in signal.source_types
    }
    confidences = [signal.confidence for signal in active]
    supporting_ids = [signal.signal_id for signal in active]
    corroborated: list[str] = []
    scored_track_b: list[str] = []

    for finding in scoreable:
        raw_id = str(attr(finding.track_b_finding, "id", default=finding.id))
        matching = next(
            (signal for signal in active if _overlaps_signal(signal, finding)),
            None,
        )
        source = _track_b_source(finding)
        source_types.add(source)
        track_b_confidence = _track_b_confidence(finding)
        scored_track_b.append(raw_id)
        if matching is not None:
            corroborated.append(matching.signal_id)
            confidences.append(min(0.98, max(matching.confidence, track_b_confidence) + 0.05))
        else:
            supporting_ids.append(raw_id)
            confidences.append(track_b_confidence)

    has_concern = bool(active or scoreable)
    has_assisted = any(signal.resolution == "assisted" for signal in active)
    has_corroboration = bool(corroborated)
    if not has_concern:
        category: RecommendationCategory = "CLEAR"
        raw_confidence = 0.5 + 0.5 * min(
            max(0.0, usable_window_ratio),
            max(0.0, informative_content_ratio),
        )
    elif len(source_types) >= 2 and (has_assisted or has_corroboration):
        category = "STRONG_EVIDENCE"
        raw_confidence = max(confidences, default=0.5)
    else:
        category = "REVIEW_REQUIRED"
        raw_confidence = max(confidences, default=0.5)

    apply_informative_cap = any(
        source in {"visual_observation", "audio_observation"}
        for source in source_types
    )
    confidence, capture_cap, informative_cap = apply_confidence_caps(
        raw_confidence,
        usable_window_ratio,
        informative_content_ratio,
        apply_informative_cap=apply_informative_cap,
    )
    return UnifiedScore(
        category=category,
        confidence=confidence,
        independent_source_types=sorted(source_types),
        supporting_signal_ids=supporting_ids,
        scored_track_b_ids=scored_track_b,
        corroborated_signal_ids=corroborated,
        capture_quality_cap_applied=capture_cap,
        informative_content_cap_applied=informative_cap,
    )


__all__ = [
    "GAZE_OBSERVATION_DISPLAY_FLOOR_MS",
    "UnifiedScore",
    "compute_unified_score",
]
