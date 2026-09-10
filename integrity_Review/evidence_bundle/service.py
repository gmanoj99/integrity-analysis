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
    EvidenceBundle,
    EvidenceSourceEntry,
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


_TITLE_OVERRIDES = {
    "phone_usage": "Phone usage (candidate)",
    "suspicious_eye_movement": "Sustained gaze",
}

_STATUS_RANK = {"confirmed": 0, "cleared": 1, "context": 2, "unknown": 3}

# Speech classes that carry the integrity meaning; preferred when several
# observations overlap one card's window.
_INTEGRITY_SPEECH = {
    "asking_for_answer",
    "receiving_dictation",
    "discussing_solution",
    "reciting_answer_choices",
}


def _humanize(event_type: str) -> str:
    return _TITLE_OVERRIDES.get(
        event_type, event_type.replace("_", " ").strip().capitalize()
    )


def _titled(event_type: str, duration_ms: int) -> str:
    return f"{_humanize(event_type)} · {round(duration_ms / 1000)}s"


def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _speech_in_window(
    perception_bundle: Any, window: tuple[int, int]
) -> tuple[str | None, list[str], str | None]:
    """Best speech evidence overlapping ``window``.

    The conversation summary is frequently the whole case ("an off-camera voice
    instructs the candidate to select an option"), and until now it existed only
    inside the perception bundle, which the review UI never receives.
    """

    observations = attr(perception_bundle, "observations", default=[]) or []
    best: tuple[int, Any] | None = None
    for observation in observations:
        start = int(attr(observation, "start_ms", "startMs", default=0) or 0)
        end = int(attr(observation, "end_ms", "endMs", default=start) or start)
        if not _overlaps((start, max(end, start + 1)), window):
            continue
        audio = attr(observation, "audio")
        summary = attr(audio, "conversation_summary_en", "conversationSummaryEn")
        if not (isinstance(summary, str) and summary.strip()):
            continue
        speech_class = attr(audio, "speech_content_class", "speechContentClass")
        rank = 1 if speech_class in _INTEGRITY_SPEECH else 0
        if best is None or rank > best[0]:
            best = (rank, audio)
    if best is None:
        return (None, [], None)
    audio = best[1]
    phrases = attr(audio, "notable_phrases_original", "notablePhrasesOriginal", default=[]) or []
    return (
        str(attr(audio, "conversation_summary_en", "conversationSummaryEn") or "").strip() or None,
        [str(p) for p in phrases if str(p).strip()],
        attr(audio, "speech_language", "speechLanguage"),
    )


def _usable_findings(track_b_findings: list[Any]) -> list[tuple[Any, str, tuple[int, int]]]:
    """Deterministic findings worth surfacing, with their window."""

    usable: list[tuple[Any, str, tuple[int, int]]] = []
    for finding in track_b_findings:
        event_type = str(attr(finding, "event_type", "eventType", default=""))
        if event_type == "ok":
            continue
        if attr(finding, "verdict", default="flagged") != "flagged":
            continue
        window = attr(finding, "timestamp_window_ms", "timestampWindowMs", default=(0, 0))
        t0, t1 = int(window[0]), int(window[1])
        duration_ms = max(0, t1 - t0)
        if (
            event_type == "suspicious_eye_movement"
            and duration_ms < GAZE_OBSERVATION_DISPLAY_FLOOR_MS
        ):
            continue
        attribution = attr(finding, "attribution")
        if event_type == "phone_usage" and attribution not in {None, "candidate"}:
            continue
        usable.append((finding, event_type, (t0, t1)))
    return usable


def build_curated_track_b_observations(
    *,
    track_b_findings: list[Any],
    deliberation_bundle: DeliberationBundle,
    integrity_stories,
    perception_bundle: Any,
    media_index: list,
    contextual_events: Any = None,
    unknown_panel: Any = None,
    sections: list[Any] | None = None,
) -> list[TrackBObservationCard]:
    """Build the one reviewer-facing card list, carrying the model's verdict.

    Previously this returned only the deterministic ``flagged`` findings, so
    the UI rendered the pre-deliberation layer and every model judgement was
    discarded: on one test session four of five cards were interactions the
    model had explicitly cleared as invigilator or technical staff.

    Now each signal the model emitted becomes a ``confirmed`` card, each
    episode it dismissed becomes a ``cleared`` card carrying the reason,
    permitted speech becomes ``context`` and coverage gaps become ``unknown``.
    Deterministic findings no longer create cards of their own — they attach to
    whichever card covers their window as ``evidence_refs``, which also
    collapses the duplicate cards two findings on one window used to produce.
    """

    stories_by_signal = {story.signal_id: story for story in integrity_stories.stories}
    signals_by_id = {s.signal_id: s for s in deliberation_bundle.validated_signals}
    findings = _usable_findings(track_b_findings)
    claimed: list[tuple[int, int]] = []
    cards: list[TrackBObservationCard] = []

    def attached(window: tuple[int, int]) -> tuple[list[str], str | None, Any]:
        """Deterministic findings covering ``window``: refs, detail, strength."""

        matched = [(f, et) for f, et, w in findings if _overlaps(w, window)]
        refs = sorted({et for _, et in matched})
        detail = next(
            (
                str(attr(f, "reasoning", default="")).strip()
                for f, _ in matched
                if str(attr(f, "reasoning", default="")).strip()
            ),
            None,
        )
        strength = next(
            (
                attr(f, "evidence_strength", "evidenceStrength")
                for f, _ in matched
                if attr(f, "evidence_strength", "evidenceStrength")
            ),
            None,
        )
        return refs, detail, strength

    # 1. Confirmed — one card per signal the model actually emitted.
    for signal in deliberation_bundle.validated_signals:
        story = stories_by_signal.get(signal.signal_id)
        window = (
            tuple(story.time_range_ms) if story else (0, 0)
        )
        t0, t1 = int(window[0]), int(window[1])
        duration_ms = max(0, t1 - t0)
        clip_ref = (
            story.proof.clip_ref
            if story and story.proof.clip_ref
            else resolve_offset_clip_ref(
                event_ms=t0, duration_ms=max(duration_ms, 1), media_index=media_index
            )
        )
        refs, detail, strength = attached((t0, t1))
        summary, phrases, language = _speech_in_window(perception_bundle, (t0, t1))
        quotes = list(story.proof.audio_quotes) if story else []
        cards.append(
            TrackBObservationCard(
                id=f"tbobs_{signal.signal_id}",
                event_type=str(signal.signal_type),
                title=story.headline if story else _titled(str(signal.signal_type), duration_ms),
                timestamp_window_ms=(t0, t1),
                duration_ms=duration_ms,
                nested_under_signal_id=signal.signal_id,
                signal_id=signal.signal_id,
                section_id=(story.section_id if story else None)
                or resolve_section_id(clip_ref, (t0, t1), media_index, sections),
                clip_ref=clip_ref,
                detail=detail,
                evidence_strength=strength,
                status="confirmed",
                confidence=signal.confidence,
                resolution=str(signal.resolution),
                severity=story.severity if story else None,
                what_happened=story.what_happened if story else None,
                why_it_matters=story.why_it_matters if story else None,
                honest_alternative=story.honest_alternative if story else None,
                audio_summary=summary,
                notable_phrases=phrases or quotes,
                speech_language=language,
                source_types=[str(s) for s in signal.source_types],
                evidence_refs=refs,
            )
        )
        claimed.append((t0, t1))

    # 2. Cleared — episodes the model considered and dismissed, with its reason.
    for episode in deliberation_bundle.episode_analysis:
        if episode.will_emit_signal and any(
            sid in signals_by_id for sid in episode.emitted_signal_ids
        ):
            continue
        t0, t1 = int(episode.time_range_ms[0]), int(episode.time_range_ms[1])
        duration_ms = max(0, t1 - t0)
        refs, detail, strength = attached((t0, t1))
        event_type = (
            episode.suspicious_behavior_type
            if episode.suspicious_behavior_type not in {"none", ""}
            else (refs[0] if refs else "reviewed_no_concern")
        )
        clip_ref = resolve_offset_clip_ref(
            event_ms=t0, duration_ms=max(duration_ms, 1), media_index=media_index
        )
        summary, phrases, language = _speech_in_window(perception_bundle, (t0, t1))
        cards.append(
            TrackBObservationCard(
                id=f"tbobs_cleared_{episode.episode_id}",
                event_type=str(event_type),
                title=_titled(str(event_type), duration_ms),
                timestamp_window_ms=(t0, t1),
                duration_ms=duration_ms,
                section_id=resolve_section_id(clip_ref, (t0, t1), media_index, sections),
                clip_ref=clip_ref,
                detail=detail or episode.episode_summary or None,
                evidence_strength=strength,
                status="cleared",
                what_happened=episode.episode_summary or None,
                reason_cleared=episode.reason_not_signalled,
                audio_summary=summary,
                notable_phrases=phrases,
                speech_language=language,
                evidence_refs=refs,
            )
        )
        claimed.append((t0, t1))

    # 3. Context — permitted interactions (invigilator, technical help).
    for event in attr(contextual_events, "events", default=[]) or []:
        t0 = int(event.timestamp_window_ms[0])
        t1 = int(event.timestamp_window_ms[1])
        duration_ms = max(0, t1 - t0)
        clip_ref = resolve_offset_clip_ref(
            event_ms=t0, duration_ms=max(duration_ms, 1), media_index=media_index
        )
        cards.append(
            TrackBObservationCard(
                id=f"tbobs_context_{event.event_type}_{t0}",
                event_type=str(event.event_type),
                title=_titled(str(event.event_type), duration_ms),
                timestamp_window_ms=(t0, t1),
                duration_ms=duration_ms,
                section_id=resolve_section_id(clip_ref, (t0, t1), media_index, sections),
                clip_ref=clip_ref,
                status="context",
                audio_summary=event.conversation_summary_en,
                notable_phrases=list(event.notable_phrases_original or []),
                speech_language=event.speech_language,
            )
        )

    # 4. Unknown — coverage gaps, so "not analysed" never reads as "nothing happened".
    for interval in attr(unknown_panel, "intervals", default=[]) or []:
        t0, t1 = int(interval.start_ms), int(interval.end_ms)
        duration_ms = max(0, int(interval.duration_ms))
        cards.append(
            TrackBObservationCard(
                id=f"tbobs_unknown_{t0}_{t1}",
                event_type="coverage_gap",
                title=_titled("coverage_gap", duration_ms),
                timestamp_window_ms=(t0, t1),
                duration_ms=duration_ms,
                status="unknown",
                detail=str(interval.reason),
                reason_cleared=str(interval.reason),
            )
        )

    # 5. Any flagged finding no episode or signal covers still gets a card,
    #    so removing the old one-card-per-finding path cannot lose evidence.
    for finding, event_type, window in findings:
        if any(_overlaps(window, c) for c in claimed):
            continue
        t0, t1 = window
        duration_ms = max(0, t1 - t0)
        clip_ref = resolve_offset_clip_ref(
            event_ms=t0, duration_ms=max(duration_ms, 1), media_index=media_index
        )
        summary, phrases, language = _speech_in_window(perception_bundle, window)
        cards.append(
            TrackBObservationCard(
                id=f"tbobs_{event_type}_{t0}_{t1}",
                event_type=event_type,
                title=_titled(event_type, duration_ms),
                timestamp_window_ms=(t0, t1),
                duration_ms=duration_ms,
                section_id=resolve_section_id(clip_ref, window, media_index, sections),
                clip_ref=clip_ref,
                detail=str(attr(finding, "reasoning", default="")) or None,
                evidence_strength=attr(finding, "evidence_strength", "evidenceStrength"),
                status="cleared",
                reason_cleared="Detected deterministically; not adjudicated by deliberation.",
                audio_summary=summary,
                notable_phrases=phrases,
                speech_language=language,
                evidence_refs=[event_type],
            )
        )

    return sorted(
        cards,
        key=lambda c: (
            _STATUS_RANK.get(c.status, 9),
            -(c.confidence or 0),
            c.timestamp_window_ms[0],
        ),
    )


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
        signals.append(
            DetectedSignalEntry(
                signal_id=signal.signal_id,
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
                episode_ref=item.episode_ref,
                citations=list(item.citations),
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
    unknown_panel = build_unknown_panel(
        input_data.perception_bundle,
        input_data.master_timeline,
    )
    contextual_events_section = build_contextual_events_section(input_data.contextual_events)
    track_b_observations = build_curated_track_b_observations(
        track_b_findings=track_b_findings,
        deliberation_bundle=bundle,
        integrity_stories=integrity_stories,
        perception_bundle=input_data.perception_bundle,
        media_index=media_index,
        contextual_events=contextual_events_section,
        unknown_panel=unknown_panel,
        sections=sections,
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
        track_b_observations=track_b_observations,
        detected_signals=_build_detected_signals(
            bundle,
            perception_bundle=input_data.perception_bundle,
            machine_facts_bundle=input_data.machine_facts_bundle,
            media_index=media_index,
            sections=sections,
        ),
        correlated_patterns=correlated_patterns,
        smart_student_notes=build_smart_student_notes(bundle, input_data.statistical_baseline),
        contextual_events=contextual_events_section,
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
