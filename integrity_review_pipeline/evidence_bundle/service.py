"""Scope 5 evidence bundle assembly (offset ClipRefs + mediaIndex; no physical clips)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from ..contracts.deliberation import DeliberationBundle, ValidatedSignal
from ..contracts.evidence_bundle import (
    ConfidenceSection,
    CorrelatedPatternEntry,
    CorrelatedPatternEvent,
    CorrelatedPatternsSection,
    DetectedSignalEntry,
    DetectedSignalsSection,
    EpisodeAnalysisSection,
    EvidenceBundle,
    EvidenceSourceEntry,
    EvidenceTimelineSection,
    HypothesisArmEntry,
    ProvenanceSection,
    RejectedSignalEntry,
    ReviewerActionSection,
    TrackBObservationCard,
)
from ..duck_helpers import attr
from ..prompts.deliberation import DELIBERATION_PROMPT_VERSION
from .clips import (
    EVIDENCE_BUNDLE_LOGIC_VERSION,
    build_media_index,
    compute_evidence_bundle_version_hash,
    resolve_offset_clip_ref,
)
from ..scoring.unified_score import GAZE_OBSERVATION_DISPLAY_FLOOR_MS
from .section_builders import (
    active_signals,
    build_behavior_summary,
    build_contextual_events_section,
    build_integrity_stories,
    build_key_reasons,
    build_recommendation_section,
    build_smart_student_notes,
    build_timeline_entries,
    build_unknown_panel,
    resolve_section_id,
    signal_time_span,
    to_reviewer_prose,
)


@dataclass(slots=True)
class EvidenceBundleInput:
    candidate_id: str
    assessment_id: str
    deliberation_bundle: DeliberationBundle
    machine_facts_bundle: Any
    perception_bundle: Any
    statistical_baseline: Any
    master_timeline: Any
    correlated_signals: Any | None = None
    track_b_findings: list[Any] | None = None
    contextual_events: list[Any] | None = None
    perception_version_hash: str = "unknown"
    baseline_version_hash: str = "unknown"


def build_curated_track_b_observations(
    *,
    track_b_findings: list[Any],
    media_index: list,
    integrity_stories,
    sections: list[Any] | None = None,
) -> list[TrackBObservationCard]:
    cards: list[TrackBObservationCard] = []
    for finding in track_b_findings:
        event_type = str(attr(finding, "event_type", "eventType", default=""))
        verdict = attr(finding, "verdict", default="flagged")
        if verdict != "flagged" or event_type == "ok":
            continue
        window = attr(finding, "timestamp_window_ms", "timestampWindowMs", default=(0, 0))
        t0, t1 = int(window[0]), int(window[1])
        duration_ms = max(0, t1 - t0)
        if event_type == "suspicious_eye_movement" and duration_ms < GAZE_OBSERVATION_DISPLAY_FLOOR_MS:
            continue
        attribution = attr(finding, "attribution")
        if event_type == "phone_usage" and attribution not in {None, "candidate"}:
            continue
        nested = next(
            (
                story.signal_id
                for story in integrity_stories.stories
                if t0 < story.time_range_ms[1] and t1 > story.time_range_ms[0]
            ),
            None,
        )
        title = {
            "phone_usage": "Phone usage (candidate)",
            "suspicious_eye_movement": "Sustained gaze",
        }.get(event_type, event_type.replace("_", " ").strip().capitalize())
        title = f"{title} · {round(duration_ms / 1000)}s"
        clip_ref = resolve_offset_clip_ref(
            event_ms=t0,
            duration_ms=duration_ms,
            media_index=media_index,
        )
        cards.append(
            TrackBObservationCard(
                id=f"tbobs_{event_type}_{t0}_{t1}",
                event_type=event_type,
                title=title,
                timestamp_window_ms=(t0, t1),
                duration_ms=duration_ms,
                nested_under_signal_id=nested,
                section_id=resolve_section_id(clip_ref, (t0, t1), media_index, sections),
                clip_ref=clip_ref,
                detail=str(attr(finding, "reasoning", default="")),
            )
        )
    return sorted(cards, key=lambda c: c.timestamp_window_ms[0])


def _build_confidence_section(
    *,
    deliberation_bundle: DeliberationBundle,
    perception_bundle: Any,
    statistical_baseline: Any,
    machine_facts_bundle: Any,
    master_timeline: Any,
) -> ConfidenceSection:
    active = active_signals(deliberation_bundle)
    source_map: dict[str, set[str]] = {}
    for signal in active:
        for source in signal.source_types:
            source_map.setdefault(str(source), set()).add(signal.signal_type)
    evidence_source_summary = [
        EvidenceSourceEntry(
            source_type=source_type,
            signal_count=len(types),
            signal_types=sorted(types),
        )
        for source_type, types in sorted(source_map.items())
    ]
    covered = int(attr(perception_bundle, "covered_windows", "coveredWindows", default=0) or 0)
    total = int(attr(perception_bundle, "total_windows", "totalWindows", default=0) or 0)
    usable_ratio = float(
        attr(
            attr(statistical_baseline, "coverage_metrics", "coverageMetrics"),
            "usable_window_ratio",
            "usableWindowRatio",
            default=1.0,
        )
        or 1.0
    )
    screen_spans = attr(master_timeline, "screen_chunk_spans", "screenChunkSpans", default=[]) or []
    screen_analysed_ms = sum(
        max(0, int(attr(span, "end_offset_ms", "endOffsetMs", default=0) or 0) - int(attr(span, "start_offset_ms", "startOffsetMs", default=0) or 0))
        for span in screen_spans
    )
    session_duration_ms = max(0, int(attr(master_timeline, "duration_ms", "durationMs", default=0) or 0))
    screen_coverage_label = None
    if screen_spans:
        if session_duration_ms > 0:
            screen_coverage_label = (
                f"{round((screen_analysed_ms / session_duration_ms) * 100)}% screen "
                f"({len(screen_spans)} chunks)"
            )
        else:
            screen_coverage_label = f"screen recording present ({len(screen_spans)} chunks)"
    return ConfidenceSection(
        capture_quality_cap_applied=deliberation_bundle.capture_quality_cap_applied,
        usable_window_ratio=usable_ratio,
        covered_windows=covered,
        total_windows=total,
        unknown_windows=int(attr(perception_bundle, "unknown_windows", "unknownWindows", default=0) or 0),
        informative_content_ratio=deliberation_bundle.informative_content_ratio,
        informative_content_cap_applied=deliberation_bundle.informative_content_cap_applied,
        evidence_source_summary=evidence_source_summary,
        video_modality_absent=covered == 0,
        video_coverage_label=(
            "0% video (modality absent)"
            if covered == 0
            else f"{round((covered / max(1, total)) * 100)}% video ({covered}/{total} windows)"
        ),
        keystroke_coverage_label=(
            "keystroke telemetry available"
            if str(attr(machine_facts_bundle, "exam_mode", "examMode", default="none"))
            == "rrweb"
            else None
        ),
        screen_coverage_label=screen_coverage_label,
    )


def _build_detected_signals(
    bundle: DeliberationBundle,
    *,
    perception_bundle: Any,
    machine_facts_bundle: Any,
    media_index: list[Any],
    sections: list[Any] | None = None,
) -> DetectedSignalsSection:
    signals: list[DetectedSignalEntry] = []
    for signal in bundle.validated_signals:
        span = signal_time_span(signal, perception_bundle, machine_facts_bundle)
        start_ms = int(span["start_ms"])
        end_ms = int(span["end_ms"])
        signals.append(
            DetectedSignalEntry(
                signal_id=signal.signal_id,
                signal_type=signal.signal_type,
                timestamp_ms=start_ms,
                confidence=signal.confidence,
                resolution=signal.resolution,
                section_id=resolve_section_id(None, (start_ms, end_ms), media_index, sections),
                source_types=[str(t) for t in signal.source_types],
                supporting_observations=signal.observations_cited,
                supporting_machine_facts=signal.machine_facts_cited,
                supporting_baseline_metrics=signal.baseline_metrics_cited,
                honest_hypothesis=HypothesisArmEntry.model_validate(
                    signal.hypothesis_honest.model_dump()
                ),
                assisted_hypothesis=HypothesisArmEntry.model_validate(
                    signal.hypothesis_assisted.model_dump()
                ),
                innocent_explanation_considered=signal.innocent_explanation_considered,
                why_rejected=signal.why_rejected,
                citations=[
                    *signal.machine_facts_cited,
                    *signal.observations_cited,
                    *signal.baseline_metrics_cited,
                ],
            )
        )
    return DetectedSignalsSection(
        signals=signals,
        rejected_signals=[
            RejectedSignalEntry(
                signal_type=item.signal_type,
                rejected_by=item.rejected_by,
                reason=item.reason,
            )
            for item in bundle.rejected_signals
        ],
    )


def _build_correlated_patterns(correlated_signals: Any | None) -> CorrelatedPatternsSection:
    contributions = attr(correlated_signals, "contributions", default=[]) or []
    return CorrelatedPatternsSection(
        patterns=[
            CorrelatedPatternEntry(
                factor_id=str(attr(c, "factor_id", "factorId", default="")),
                label=str(attr(c, "label", default=attr(c, "factor_id", "factorId", default=""))),
                tier=int(attr(c, "tier", default=2) or 2),
                score_contribution=float(
                    attr(c, "score_contribution", "scoreContribution", default=0) or 0
                ),
                requires_corroboration=bool(
                    attr(c, "requires_corroboration", "requiresCorroboration", default=False)
                ),
                rationale=str(attr(c, "rationale", default="") or ""),
                time_range_ms=tuple(attr(c, "time_range_ms", "timeRangeMs", default=(0, 0))),  # type: ignore[arg-type]
                evidence_refs=list(attr(c, "evidence_refs", "evidenceRefs", default=[]) or []),
                events=[
                    CorrelatedPatternEvent(ref=str(ref), kind="evidence", t_ms=None, label=str(ref))
                    for ref in (attr(c, "evidence_refs", "evidenceRefs", default=[]) or [])[:5]
                ],
            )
            for c in contributions
        ],
        total_weighted_score=float(
            attr(correlated_signals, "total_weighted_score", "totalWeightedScore", default=0) or 0
        ),
        logic_version=str(attr(correlated_signals, "logic_version", "logicVersion", default="") or ""),
        notes=[],
    )


def assemble_evidence_bundle(input_data: EvidenceBundleInput) -> EvidenceBundle:
    bundle = input_data.deliberation_bundle
    media_index = build_media_index(input_data.master_timeline)
    sections = attr(input_data.master_timeline, "sections", default=[]) or []
    covered_windows = int(
        attr(input_data.perception_bundle, "covered_windows", "coveredWindows", default=0) or 0
    )
    composite_hash = compute_evidence_bundle_version_hash(
        bundle.provenance.composite_version_hash
    )

    integrity_stories = build_integrity_stories(
        deliberation_bundle=bundle,
        machine_facts_bundle=input_data.machine_facts_bundle,
        perception_bundle=input_data.perception_bundle,
        correlated_signals=input_data.correlated_signals,
        covered_windows=covered_windows,
        media_index=media_index,
        sections=sections,
    )
    track_b_findings = input_data.track_b_findings or []
    track_b_observations = build_curated_track_b_observations(
        track_b_findings=track_b_findings,
        media_index=media_index,
        integrity_stories=integrity_stories,
        sections=sections,
    )
    timeline_entries = build_timeline_entries(
        deliberation_bundle=bundle,
        machine_facts_bundle=input_data.machine_facts_bundle,
        perception_bundle=input_data.perception_bundle,
        media_index=media_index,
        covered_windows=covered_windows,
    )
    correlated_patterns = _build_correlated_patterns(input_data.correlated_signals)
    return EvidenceBundle(
        candidate_id=input_data.candidate_id,
        assessment_id=input_data.assessment_id,
        produced_at=datetime.now(timezone.utc).isoformat(),
        behavior_summary=build_behavior_summary(bundle),
        recommendation=build_recommendation_section(bundle),
        confidence=_build_confidence_section(
            deliberation_bundle=bundle,
            perception_bundle=input_data.perception_bundle,
            statistical_baseline=input_data.statistical_baseline,
            machine_facts_bundle=input_data.machine_facts_bundle,
            master_timeline=input_data.master_timeline,
        ),
        key_reasons=build_key_reasons(bundle),
        integrity_stories=integrity_stories,
        track_b_observations=track_b_observations,
        detected_signals=_build_detected_signals(
            bundle,
            perception_bundle=input_data.perception_bundle,
            machine_facts_bundle=input_data.machine_facts_bundle,
            media_index=media_index,
            sections=sections,
        ),
        correlated_patterns=correlated_patterns,
        episode_analysis=EpisodeAnalysisSection(
            episodes=bundle.episode_analysis,
            emitted_count=sum(1 for e in bundle.episode_analysis if e.will_emit_signal),
        ),
        evidence_timeline=EvidenceTimelineSection(entries=timeline_entries),
        unknown_panel=build_unknown_panel(
            input_data.perception_bundle,
            input_data.master_timeline,
        ),
        smart_student_notes=build_smart_student_notes(bundle, input_data.statistical_baseline),
        contextual_events=build_contextual_events_section(
            input_data.contextual_events
        ),
        reviewer_action=ReviewerActionSection(),
        provenance=ProvenanceSection(
            candidate_id=input_data.candidate_id,
            assessment_id=input_data.assessment_id,
            generated_at=datetime.now(timezone.utc).isoformat(),
            timeline_version="mt-screen-v1",
            machine_facts_version="scope1-py",
            perception_version=input_data.perception_version_hash,
            baseline_version=input_data.baseline_version_hash,
            correlated_signals_version=correlated_patterns.logic_version or "none",
            deliberation_version=bundle.provenance.composite_version_hash,
            prompt_version=DELIBERATION_PROMPT_VERSION,
            evidence_bundle_version=EVIDENCE_BUNDLE_LOGIC_VERSION,
            clip_cache_version="offset-only-v1",
            model_versions={"deliberationModel": bundle.model_version},
            composite_hash=composite_hash,
        ),
        media_index=media_index,
    )


__all__ = [
    "EvidenceBundleInput",
    "EVIDENCE_BUNDLE_LOGIC_VERSION",
    "assemble_evidence_bundle",
    "build_media_index",
    "resolve_offset_clip_ref",
    "compute_evidence_bundle_version_hash",
    "to_reviewer_prose",
]
