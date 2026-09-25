from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

from ..contracts.deliberation import DeliberationBundle, ValidatedSignal
from ..contracts.evidence_bundle import (
    CorrelatedPatternMoment,
    CorrelatedPatternProof,
    KeystrokeEvidence,
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
from ..lib.plain_text import plain_text, seconds, term
from ..duck_helpers import attr, fact_kind
from ..prompts.deliberation import DELIBERATION_PROMPT_VERSION
from .clips import (
    evidence_stream_for_event,
    EVIDENCE_BUNDLE_LOGIC_VERSION,
    build_media_index,
    compute_evidence_bundle_version_hash,
    resolve_offset_clip_ref,
)
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
    audio_available: bool = True


_TITLE_OVERRIDES = {
    "phone_usage": "Phone usage (candidate)",
    "suspicious_eye_movement": "Sustained gaze",
}

_STATUS_RANK = {"confirmed": 0, "cleared": 1, "context": 2, "unknown": 3}

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


def _titled(event_type: str) -> str:
    return _humanize(event_type)


def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _speech_in_window(
    perception_bundle: Any, window: tuple[int, int]
) -> tuple[str | None, list[str], str | None]:
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
        if attr(audio, "speech_present", "speechPresent") != "yes":
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


AUDIO_SIGNAL_TYPES = frozenset({"possible_audio_coaching", "possible_remote_dictation"})
MAX_SPEECH_MOMENTS = 3


def _heard_in_window(perception_bundle: Any, window: tuple[int, int]) -> str | None:
    """What was said in the window: exam-relevant speech first, then time order."""

    moments: list[tuple[int, int, str]] = []
    seen: set[str] = set()
    for observation in attr(perception_bundle, "observations", default=[]) or []:
        start = int(attr(observation, "start_ms", "startMs", default=0) or 0)
        end = int(attr(observation, "end_ms", "endMs", default=start) or start)
        audio = attr(observation, "audio")
        text = str(attr(audio, "conversation_summary_en", "conversationSummaryEn") or "").strip()
        if (
            not text or text in seen
            or attr(audio, "speech_present", "speechPresent") != "yes"
            or not _overlaps((start, max(end, start + 1)), window)
        ):
            continue
        seen.add(text)
        relevant = attr(audio, "speech_content_class", "speechContentClass") in _INTEGRITY_SPEECH
        moments.append((0 if relevant else 1, start, text))
    if not moments:
        return None
    picked = sorted(sorted(moments)[:MAX_SPEECH_MOMENTS], key=lambda m: m[1])
    return plain_text(" ".join(
        f"At {start // 60_000}:{start // 1000 % 60:02d}, {text[0].lower() + text[1:]}" for _, start, text in picked
    ))


INSTANTANEOUS_EVENT_TYPES = frozenset({
    "external_paste",
    "mass_paste",
    "screen_external_paste",
    "answer_appeared",
})


@dataclass(frozen=True)
class AttachedFindings:
    refs: list[str]
    detail: str | None
    strength: Any
    keystroke_evidence: KeystrokeEvidence | None = None
    is_instantaneous: bool = False


def _inserted_chars(finding: Any) -> int:
    proof = attr(finding, "keystroke_proof", "keystrokeProof")
    return int(attr(proof, "inserted_char_count", "insertedCharCount", default=0) or 0)


def _keystroke_evidence(finding: Any) -> KeystrokeEvidence | None:
    if finding is None:
        return None
    proof = attr(finding, "keystroke_proof", "keystrokeProof")
    screen_proof = attr(finding, "screen_proof", "screenProof")
    if proof is None and screen_proof is None:
        return None
    if proof is None:
        return KeystrokeEvidence(
            pasted_excerpt=attr(screen_proof, "pasted_excerpt", "pastedExcerpt"),
        ) if attr(screen_proof, "pasted_excerpt", "pastedExcerpt") else None
    evidence = KeystrokeEvidence(
        pasted_excerpt=attr(proof, "pasted_excerpt", "pastedExcerpt")
        or attr(screen_proof, "pasted_excerpt", "pastedExcerpt"),
        inserted_char_count=attr(proof, "inserted_char_count", "insertedCharCount"),
        final_value_length=attr(proof, "final_value_length", "finalValueLength"),
        preceding_gap_ms=attr(proof, "preceding_gap_ms", "precedingGapMs"),
        keystrokes_in_window=attr(proof, "keystrokes_in_window", "keystrokesInWindow"),
        field_id=attr(proof, "field_id", "fieldId"),
    )
    if not evidence.model_dump(exclude_none=True):
        return None
    return evidence


def _usable_findings(track_b_findings: list[Any]) -> list[tuple[Any, str, tuple[int, int]]]:
    usable: list[tuple[Any, str, tuple[int, int]]] = []
    for finding in track_b_findings:
        event_type = str(attr(finding, "event_type", "eventType", default=""))
        if event_type == "ok":
            continue
        if attr(finding, "verdict", default="flagged") not in {"flagged", "provisional"}:
            continue
        window = attr(finding, "timestamp_window_ms", "timestampWindowMs", default=(0, 0))
        t0, t1 = int(window[0]), int(window[1])
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
    stories_by_signal = {story.signal_id: story for story in integrity_stories.stories}
    signals_by_id = {s.signal_id: s for s in deliberation_bundle.validated_signals}
    findings = _usable_findings(track_b_findings)
    claimed: list[tuple[int, int]] = []
    cards: list[TrackBObservationCard] = []

    def attached(window: tuple[int, int]) -> AttachedFindings:
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
        keystroke = _keystroke_evidence(
            max(
                (
                    f for f, _ in matched
                    if attr(f, "keystroke_proof", "keystrokeProof")
                    or attr(f, "screen_proof", "screenProof")
                ),
                key=lambda f: _inserted_chars(f),
                default=None,
            )
        )
        return AttachedFindings(
            refs=refs,
            detail=detail,
            strength=strength,
            keystroke_evidence=keystroke,
            is_instantaneous=bool(matched) and all(
                et in INSTANTANEOUS_EVENT_TYPES for _, et in matched
            ),
        )

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
                event_ms=t0,
                duration_ms=max(duration_ms, 1),
                media_index=media_index,
                prefer_evidence_type=evidence_stream_for_event(str(signal.signal_type)),
            )
        )
        found = attached((t0, t1))
        summary, phrases, language = _speech_in_window(perception_bundle, (t0, t1))
        quotes = list(story.proof.audio_quotes) if story else []
        audio_related = str(signal.signal_type) in AUDIO_SIGNAL_TYPES or any(
            str(s) == "audio_observation" for s in signal.source_types
        )
        heard = _heard_in_window(perception_bundle, (t0, t1)) if audio_related else None
        cards.append(
            TrackBObservationCard(
                id=f"tbobs_{signal.signal_id}",
                event_type=str(signal.signal_type),
                title=story.headline if story else _titled(str(signal.signal_type)),
                timestamp_window_ms=(t0, t1),
                duration_ms=duration_ms,
                nested_under_signal_id=signal.signal_id,
                signal_id=signal.signal_id,
                section_id=(story.section_id if story else None)
                or resolve_section_id(clip_ref, (t0, t1), media_index, sections),
                clip_ref=clip_ref,
                detail=found.detail,
                evidence_strength=found.strength,
                status="confirmed",
                confidence=signal.confidence,
                resolution=str(signal.resolution),
                severity=story.severity if story else None,
                what_happened=heard or (story.what_happened if story else None),
                why_it_matters=story.why_it_matters if story else None,
                honest_alternative=story.honest_alternative if story else None,
                audio_summary=summary,
                notable_phrases=phrases or quotes,
                speech_language=language,
                source_types=[str(s) for s in signal.source_types],
                keystroke_evidence=found.keystroke_evidence,
                is_instantaneous=found.is_instantaneous,
                evidence_refs=found.refs,
            )
        )
        claimed.append((t0, t1))

    for episode in deliberation_bundle.episode_analysis:
        if episode.will_emit_signal and any(
            sid in signals_by_id for sid in episode.emitted_signal_ids
        ):
            continue
        t0, t1 = int(episode.time_range_ms[0]), int(episode.time_range_ms[1])
        duration_ms = max(0, t1 - t0)
        found = attached((t0, t1))
        event_type = (
            episode.suspicious_behavior_type
            if episode.suspicious_behavior_type not in {"none", ""}
            else (found.refs[0] if found.refs else "reviewed_no_concern")
        )
        clip_ref = resolve_offset_clip_ref(
            event_ms=t0,
            duration_ms=max(duration_ms, 1),
            media_index=media_index,
            prefer_evidence_type=evidence_stream_for_event(str(event_type)),
        )
        summary, phrases, language = _speech_in_window(perception_bundle, (t0, t1))
        cards.append(
            TrackBObservationCard(
                id=f"tbobs_cleared_{episode.episode_id}",
                event_type=str(event_type),
                title=_titled(str(event_type)),
                timestamp_window_ms=(t0, t1),
                duration_ms=duration_ms,
                section_id=resolve_section_id(clip_ref, (t0, t1), media_index, sections),
                clip_ref=clip_ref,
                detail=found.detail or episode.episode_summary or None,
                evidence_strength=found.strength,
                status="cleared",
                what_happened=episode.episode_summary or None,
                reason_cleared=episode.reason_not_signalled,
                audio_summary=summary,
                notable_phrases=phrases,
                speech_language=language,
                keystroke_evidence=found.keystroke_evidence,
                is_instantaneous=found.is_instantaneous,
                evidence_refs=found.refs,
            )
        )
        claimed.append((t0, t1))

    for event in attr(contextual_events, "events", default=[]) or []:
        t0 = int(event.timestamp_window_ms[0])
        t1 = int(event.timestamp_window_ms[1])
        duration_ms = max(0, t1 - t0)
        clip_ref = resolve_offset_clip_ref(
            event_ms=t0,
            duration_ms=max(duration_ms, 1),
            media_index=media_index,
            prefer_evidence_type=evidence_stream_for_event(str(event.event_type)),
        )
        cards.append(
            TrackBObservationCard(
                id=f"tbobs_context_{event.event_type}_{t0}",
                event_type=str(event.event_type),
                title=_titled(str(event.event_type)),
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

    for interval in attr(unknown_panel, "intervals", default=[]) or []:
        t0, t1 = int(interval.start_ms), int(interval.end_ms)
        duration_ms = max(0, int(interval.duration_ms))
        cards.append(
            TrackBObservationCard(
                id=f"tbobs_unknown_{t0}_{t1}",
                event_type="coverage_gap",
                title=_titled("coverage_gap"),
                timestamp_window_ms=(t0, t1),
                duration_ms=duration_ms,
                status="unknown",
                detail=str(interval.reason),
                reason_cleared=str(interval.reason),
            )
        )

    for finding, event_type, window in findings:
        if any(_overlaps(window, c) for c in claimed):
            continue
        t0, t1 = window
        duration_ms = max(0, t1 - t0)
        clip_ref = resolve_offset_clip_ref(
            event_ms=t0,
            duration_ms=max(duration_ms, 1),
            media_index=media_index,
            prefer_evidence_type=evidence_stream_for_event(event_type),
        )
        summary, phrases, language = _speech_in_window(perception_bundle, window)
        cards.append(
            TrackBObservationCard(
                id=f"tbobs_{event_type}_{t0}_{t1}",
                event_type=event_type,
                title=_titled(event_type),
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
                keystroke_evidence=_keystroke_evidence(finding),
                is_instantaneous=event_type in INSTANTANEOUS_EVENT_TYPES,
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
    keystroke_analysed = (
        str(attr(machine_facts_bundle, "exam_mode", "examMode", default="none")) == "rrweb"
    )
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
            "keystroke telemetry available" if keystroke_analysed else None
        ),
        screen_coverage_label=screen_coverage_label,
        video_analysed=covered > 0,
        screen_analysed=bool(screen_spans),
        keystroke_analysed=keystroke_analysed,
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
                card_id=f"tbobs_{signal.signal_id}",
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


_SECONDS = seconds
_term = term


CORRELATION_RATIONALE_TEMPLATES: tuple[tuple[re.Pattern[str], Any], ...] = (
    (
        re.compile(r"^(\w+) co-occurs with speech classified (\w+) \(within (\d+)ms\)$"),
        lambda m: f"{_term(m[1]).capitalize()} while the conversation was "
                  f"{_term(m[2])} — the two within {_SECONDS(m[3])} of each other.",
    ),
    (
        re.compile(r"^(\w+) co-occurs with (\w+) \(within (\d+)ms\)$"),
        lambda m: f"{_term(m[1]).capitalize()} and {_term(m[2])}, "
                  f"within {_SECONDS(m[3])} of each other.",
    ),
    (
        re.compile(r"^(\w+) ended then (\w+) (\d+)ms later \(no intervening activity\)$"),
        lambda m: f"{_term(m[1]).capitalize()}, then {_term(m[2])} "
                  f"{_SECONDS(m[3])} later with nothing in between.",
    ),
    (
        re.compile(r"^(\d+) focus losses within (\d+)ms \(window \d+ms\)$"),
        lambda m: f"The candidate left the exam window {m[1]} times "
                  f"in {_SECONDS(m[2])}.",
    ),
    (
        re.compile(r"^fullscreen lost → (\w+) (\d+)ms later$"),
        lambda m: f"The exam left fullscreen, then {_term(m[1])} {_SECONDS(m[2])} later.",
    ),
    (
        re.compile(r"^gaze_off → (\w+) → (\w+) chain\.(.*)$"),
        lambda m: f"The candidate looked away from the screen, then there was speech, "
                  f"then {_term(m[2])}.{m[3]}",
    ),
    (
        re.compile(r"^(\w+)(?: of (\d+) chars)? with no in-exam COPY in the preceding (\d+)ms$"),
        lambda m: f"{_term(m[1]).capitalize()}"
                  + (f" ({m[2]} characters)" if m[2] else "")
                  + f" with nothing copied inside the exam in the {_SECONDS(m[3])} before, "
                  "so it came from outside.",
    ),
    (
        re.compile(r"^(.+?) → external LARGE_PASTE (\d+)ms later(.*)$"),
        lambda m: f"{_term(m[1]).capitalize()}, then a large block of text was pasted "
                  f"{_SECONDS(m[2])} later{m[3]}.",
    ),
    (
        re.compile(r"^(\w+) overlapping screen/input activity \((\w+)\)$"),
        lambda m: f"{_term(m[1]).capitalize()} while {_term(m[2])}.",
    ),
)


def humanize_correlation_rationale(rationale: str, label: str) -> str | None:
    text = (rationale or "").strip()
    if not text:
        return None
    for pattern, render in CORRELATION_RATIONALE_TEMPLATES:
        match = pattern.match(text)
        if match:
            return plain_text(render(match))
    out = plain_text(text)
    return out if out != text else label


_WINDOW_RE = re.compile(r"w_(\d+)_(\d+)")
CORRELATION_MOMENT_CLIP_MS = 20_000
MAX_CORRELATION_MOMENTS = 4


CORRELATION_FACTOR_STREAM_HINTS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("gaze", "phone", "second_person", "speech", "whisper", "camera_absent",
      "face_absent", "av_mismatch", "interact", "video_overlap"), "video"),
    (("paste", "blur", "fullscreen", "typing", "external_resource",
      "second_monitor", "idle", "submission"), "screen"),
)


def _correlation_stream(factor_id: str, evidence_refs: list[str]) -> str | None:
    refs = [str(r) for r in evidence_refs]
    if any(r.startswith("perception:") for r in refs):
        return "video"
    if any(r.startswith(("screen:", "fact:")) for r in refs):
        return "screen"
    for needles, stream in CORRELATION_FACTOR_STREAM_HINTS:
        if any(n in factor_id for n in needles):
            return stream
    return None


MOMENT_LABELS: dict[str, str] = {
    "suspicious_eye_movement": "Looked away",
    "external_help": "Second person",
    "phone_usage": "Phone visible",
    "no_candidate": "Not in frame",
    "left_examination_area": "Left the desk",
    "discussing_solution_audio": "Answer discussion",
    "whisper_or_dictation": "Dictation heard",
    "off_camera_voice_coaching": "Off-camera coaching",
    "secondary_workspace_visible": "Second screen",
    "external_resource_open": "Non-exam app",
    "external_paste": "Paste",
    "mass_paste": "Paste",
    "screen_external_paste": "Paste",
}


def _moment_label(kind: str, start_ms: int) -> str:
    phrase = MOMENT_LABELS.get(kind) or (
        kind.replace("_", " ").strip().capitalize() or "Evidence"
    )
    return f"{phrase} · {_fmt_clock(start_ms)}"


def _correlation_moments(
    evidence_refs: list[str],
    time_range: tuple[int, int],
    media_index: list,
    prefer_evidence_type: str | None = None,
    members: list[Any] | None = None,
) -> list[CorrelatedPatternMoment]:
    if members:
        spans = [
            (
                int(attr(m, "t_ms", "tMs", default=0) or 0),
                int(attr(m, "end_ms", "endMs", default=0) or 0),
                str(attr(m, "kind", default="")),
            )
            for m in members
        ]
        spans = [(t, max(t, e), k) for t, e, k in spans]
        overlap_start = max(t for t, _, _ in spans)
        overlap_end = min(e for _, e, _ in spans)
        moments: list[CorrelatedPatternMoment] = []
        seen: dict[tuple, CorrelatedPatternMoment] = {}
        for start, end, kind in spans:
            if overlap_end > overlap_start:
                anchor = min(max(start, overlap_start), end)
            else:
                anchor = start if start >= overlap_start else max(start, end - 1)
            span_ms = min(max(end - anchor, 1), CORRELATION_MOMENT_CLIP_MS)
            clip = resolve_offset_clip_ref(
                event_ms=anchor,
                duration_ms=span_ms,
                media_index=media_index,
                single_segment=True,
                prefer_evidence_type=prefer_evidence_type,  # type: ignore[arg-type]
            )
            label = _moment_label(kind, anchor)
            key = (
                tuple((s.chunk_id, s.seek_to_ms) for s in clip.segments)
                if clip
                else ("none", anchor)
            )
            existing = seen.get(key)
            if existing is not None:
                merged = f"{existing.label.rsplit(' · ', 1)[0]} + {label}"
                seen[key] = existing.model_copy(update={"label": merged})
                continue
            moment = CorrelatedPatternMoment(
                label=label,
                time_range_ms=(anchor, max(anchor, end)),
                clip_ref=clip,
            )
            seen[key] = moment
        if seen:
            return list(seen.values())[:MAX_CORRELATION_MOMENTS]

    windows = sorted({
        (int(a), int(b))
        for ref in evidence_refs
        for a, b in _WINDOW_RE.findall(str(ref))
    })
    if not windows:
        windows = [ (int(time_range[0]), int(time_range[1])) ]

    merged: list[list[int]] = []
    for start, end in windows:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    moments: list[CorrelatedPatternMoment] = []
    for start, end in merged[:MAX_CORRELATION_MOMENTS]:
        span = min(max(end - start, 1), CORRELATION_MOMENT_CLIP_MS)
        moments.append(
            CorrelatedPatternMoment(
                label=f"{_fmt_clock(start)}–{_fmt_clock(end)}",
                time_range_ms=(start, end),
                clip_ref=resolve_offset_clip_ref(
                    event_ms=start,
                    duration_ms=span,
                    media_index=media_index,
                    single_segment=True,
                    prefer_evidence_type=prefer_evidence_type,  # type: ignore[arg-type]
                ),
            )
        )
    return moments


def _fmt_clock(ms: int) -> str:
    total = max(0, int(ms)) // 1000
    return f"{total // 60}:{total % 60:02d}"


def _correlation_machine_facts(
    evidence_refs: list[str], machine_facts_bundle: Any
) -> tuple[list[dict], KeystrokeEvidence | None]:
    wanted = {
        str(ref).split(":", 1)[1]
        for ref in evidence_refs
        if str(ref).startswith("fact:")
    }
    if not wanted:
        return [], None
    rows: list[dict] = []
    excerpt: KeystrokeEvidence | None = None
    for fact in getattr(machine_facts_bundle, "facts", []) or []:
        if str(attr(fact, "id", default="")) not in wanted:
            continue
        detail = attr(fact, "detail", default={}) or {}
        rows.append({
            "kind": fact_kind(fact),
            "startOffsetMs": attr(fact, "start_offset_ms", "startOffsetMs", default=0),
        })
        pasted = detail.get("pastedExcerpt") if isinstance(detail, Mapping) else None
        if pasted and excerpt is None:
            excerpt = KeystrokeEvidence(
                pasted_excerpt=str(pasted),
                inserted_char_count=detail.get("charsAdded"),
                final_value_length=detail.get("totalChars"),
            )
    return rows, excerpt


def _build_correlated_pattern(
    contribution: Any,
    *,
    media_index: list,
    perception_bundle: Any,
    machine_facts_bundle: Any,
) -> CorrelatedPatternEntry:
    c = contribution
    label = str(attr(c, "label", default=attr(c, "factor_id", "factorId", default="")))
    rationale = str(attr(c, "rationale", default="") or "")
    time_range = tuple(attr(c, "time_range_ms", "timeRangeMs", default=(0, 0)))
    refs = list(attr(c, "evidence_refs", "evidenceRefs", default=[]) or [])

    moments = _correlation_moments(
        refs,
        time_range,  # type: ignore[arg-type]
        media_index,
        prefer_evidence_type=_correlation_stream(
            str(attr(c, "factor_id", "factorId", default="")), refs
        ),
        members=list(attr(c, "member_events", "memberEvents", default=[]) or []),
    )
    facts, paste_evidence = _correlation_machine_facts(refs, machine_facts_bundle)
    audio_summary, phrases, language = _speech_in_window(perception_bundle, time_range)  # type: ignore[arg-type]
    window_ids = sorted({
        f"w_{a}_{b}" for ref in refs for a, b in _WINDOW_RE.findall(str(ref))
    })
    primary = next((m.clip_ref for m in moments if m.clip_ref), None)

    return CorrelatedPatternEntry(
        factor_id=str(attr(c, "factor_id", "factorId", default="")),
        label=label,
        tier=int(attr(c, "tier", default=2) or 2),
        score_contribution=float(
            attr(c, "score_contribution", "scoreContribution", default=0) or 0
        ),
        requires_corroboration=bool(
            attr(c, "requires_corroboration", "requiresCorroboration", default=False)
        ),
        rationale=rationale,
        explanation=humanize_correlation_rationale(rationale, label),
        time_range_ms=time_range,  # type: ignore[arg-type]
        evidence_refs=refs,
        events=[
            CorrelatedPatternEvent(ref=str(ref), kind="evidence", t_ms=None, label=str(ref))
            for ref in refs[:5]
        ],
        audio_summary=audio_summary,
        notable_phrases=phrases,
        speech_language=language,
        keystroke_evidence=paste_evidence,
        proof=CorrelatedPatternProof(
            time_range_ms=time_range,  # type: ignore[arg-type]
            video_seek_ms=moments[0].time_range_ms[0] if moments else None,
            clip_ref=primary,
            moments=moments,
            machine_facts=facts,
            perception_window_ids=window_ids,
            telemetry_only=primary is None,
        ),
        question_number=attr(c, "question_number", "questionNumber"),
        section_title=attr(c, "section_title", "sectionTitle"),
    )


def _build_correlated_patterns(
    correlated_signals: Any | None,
    *,
    media_index: list | None = None,
    perception_bundle: Any = None,
    machine_facts_bundle: Any = None,
) -> CorrelatedPatternsSection:
    contributions = attr(correlated_signals, "contributions", default=[]) or []
    return CorrelatedPatternsSection(
        patterns=[
            _build_correlated_pattern(
                c,
                media_index=media_index or [],
                perception_bundle=perception_bundle,
                machine_facts_bundle=machine_facts_bundle,
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
    correlated_patterns = _build_correlated_patterns(
        input_data.correlated_signals,
        media_index=media_index,
        perception_bundle=input_data.perception_bundle,
        machine_facts_bundle=input_data.machine_facts_bundle,
    )
    return EvidenceBundle(
        candidate_id=input_data.candidate_id,
        assessment_id=input_data.assessment_id,
        produced_at=datetime.now(timezone.utc).isoformat(),
        trust_score=bundle.trust_score,
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
            model_versions={
                "deliberationModel": bundle.model_version,
                "audioCapture": "available" if input_data.audio_available else "none",
            },
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
