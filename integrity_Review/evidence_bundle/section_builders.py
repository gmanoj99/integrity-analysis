"""Scope 5 section builders ported from evidenceBundleService.ts (retained scope only)."""

from __future__ import annotations

import re
import uuid
from typing import Any, Literal

from ..contracts.deliberation import DeliberationBundle, ValidatedSignal
from ..contracts.evidence_bundle import (
    BehaviorSummarySection,
    ClipRef,
    ContextualEventsSection,
    ContextualSpeechEvent,
    IntegrityStoriesSection,
    IntegrityStoryEntry,
    IntegrityStoryProof,
    KeyReasonsSection,
    MediaIndexEntry,
    RecommendationSection,
    SmartStudentNotesSection,
    SmartStudentNote,
    UnknownInterval,
    UnknownPanelSection,
)
from ..duck_helpers import attr, fact_kind, fact_start_ms, mapping_view
from ..deliberation.rules import (
    FIELD_PATH_RESOLVERS,
    parse_observation_citation,
    synthesize_integrity_story_fallback,
)
from .clips import evidence_stream_for_event, resolve_offset_clip_ref

LONG_SPAN_MS = 300_000
OBSERVATION_CLIP_DURATION_MS = 20_000
# How far from a signal's cited windows a machine fact still counts as
# corroborating it, and how long a gap ends a burst of telemetry-only facts.
FACT_WINDOW_PAD_MS = 15_000
FACT_CLUSTER_GAP_MS = 60_000

REVIEWER_PROSE_REWRITES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bavgCharsPerSecond\b"), "typing rate"),
    (re.compile(r"\bcadenceRegularity\b"), "typing consistency"),
    (re.compile(r"\bikiDistribution\b"), "typing interval spread"),
)

SIGNAL_HUMAN_LABELS: dict[str, str] = {
    "possible_external_consultation": "Possible external consultation",
    "possible_second_person_involvement": "Possible second-person involvement",
    "unauthorized_reference_usage": "Unauthorized reference usage",
    "abnormal_paste_workflow": "Abnormal paste workflow",
    "suspicious_focus_pattern": "Suspicious focus pattern",
    "abnormal_correction_pattern": "Abnormal correction pattern",
    "typing_cadence_mismatch": "Typing cadence mismatch",
    "inconsistent_interaction_sequence": "Inconsistent interaction sequence",
    "possible_audio_coaching": "Possible audio coaching",
    "possible_remote_dictation": "Possible remote dictation",
}


# What each concern is called when a non-technical reviewer reads it. The
# detector names in SIGNAL_HUMAN_LABELS are still right for the signal cards,
# where they sit beside the evidence; at the top of the page they read as
# system vocabulary, so the summary uses ordinary words instead.
SIGNAL_PLAIN_PHRASES: dict[str, str] = {
    "possible_external_consultation": "used a phone or looked something up outside the exam",
    "possible_second_person_involvement": "been helped by another person",
    "unauthorized_reference_usage": "referred to material that was not allowed",
    "abnormal_paste_workflow": "pasted in text from outside the exam",
    "suspicious_focus_pattern": "repeatedly looked away from the screen",
    "abnormal_correction_pattern": "made unusual corrections while answering",
    "typing_cadence_mismatch": "typed in a way that did not match the rest of the session",
    "inconsistent_interaction_sequence": "behaved inconsistently across the session",
    "possible_audio_coaching": "been told answers by someone nearby",
    "possible_remote_dictation": "been read answers by someone not in the room",
}


def plain_signal_phrase(signal_type: str) -> str:
    return SIGNAL_PLAIN_PHRASES.get(
        signal_type, signal_type.replace("_", " ").strip()
    )


def to_reviewer_prose(text: str | None) -> str | None:
    if not text:
        return text
    out = text
    for pattern, replacement in REVIEWER_PROSE_REWRITES:
        out = pattern.sub(replacement, out)
    return out


def humanize_signal_type(signal_type: str) -> str:
    return SIGNAL_HUMAN_LABELS.get(
        signal_type,
        signal_type.replace("_", " ").strip().capitalize(),
    )


def disposition_prefix(signals: list[ValidatedSignal]) -> str:
    if not signals:
        return "Cleared"
    if all(s.resolution == "assisted" for s in signals):
        return "Confirmed"
    return "Needs review"


def active_signals(bundle: DeliberationBundle) -> list[ValidatedSignal]:
    return [s for s in bundle.validated_signals if s.resolution != "honest"]


def _looks_like_clear_prose(text: str) -> bool:
    lowered = text.lower()
    return any(
        marker in lowered
        for marker in (
            "no significant integrity",
            "no integrity concerns",
            "no concerns were identified",
            "no malpractice",
            "adequate explanations",
            "no action required",
        )
    )


NOTHING_FOUND_TEXT = (
    "No malpractice was found — the candidate completed the exam on their own, with no phone, "
    "no other person and no outside help seen or heard at any point."
)
NOTHING_FOUND_UNVERIFIED_TEXT = (
    "No malpractice was found, but some parts of the session could not be checked clearly."
)


def _cleared_moments_text(rejected: int) -> str:
    """One plain sentence for a session where everything looked at was fine.

    Reviewers here are HR staff, so this never names counts of internal
    objects or how much of the session was analysable — only, in words, that
    the questionable moments turned out to be ordinary.
    """

    if rejected <= 0:
        return NOTHING_FOUND_TEXT
    if rejected == 1:
        return (
            "No malpractice was found — one moment was checked closely "
            "and it had a normal explanation."
        )
    return (
        f"No malpractice was found — {rejected} moments were checked closely "
        "and each had a normal explanation."
    )


def build_behavior_summary(bundle: DeliberationBundle) -> BehaviorSummarySection:
    active = active_signals(bundle)
    if bundle.recommendation.category == "CLEAR":
        return BehaviorSummarySection(
            text=_cleared_moments_text(bundle.rejected_signal_count)
        )
    verbatim = (bundle.recommendation.behavior_summary or "").strip()
    if verbatim and active and not _looks_like_clear_prose(verbatim):
        return BehaviorSummarySection(text=verbatim)
    if not active:
        # Nothing survived adjudication, so the honest thing to tell the
        # reviewer is that nothing was found — not the old placeholder, which
        # rendered as a near-empty card. `reasoning` and `recommendation` are
        # written for the verdict panel and read as jargon at the top of the
        # page, so they are no longer promoted into the summary.
        if verbatim and not _looks_like_clear_prose(verbatim):
            return BehaviorSummarySection(text=verbatim[:500])
        return BehaviorSummarySection(text=NOTHING_FOUND_UNVERIFIED_TEXT)
    stories = [s.integrity_story for s in active if s.integrity_story]
    told = [
        f"{story.headline.rstrip('. ')}. {story.what_happened}"
        for story in stories
        if story.what_happened
    ]
    if told:
        return BehaviorSummarySection(text=" ".join(told)[:600])
    # Last resort: the model gave us no narrative at all. The old fallback
    # printed a raw citation string ("w_479874_539881.people.secondPersonVisible
    # =yes") and detector names straight to the reviewer. Describe the concerns
    # in the words a non-technical reviewer would use instead.
    phrases = sorted({plain_signal_phrase(s.signal_type) for s in active})
    if len(phrases) == 1:
        concerns = phrases[0]
    else:
        concerns = f"{', '.join(phrases[:-1])} and {phrases[-1]}"
    # Both leads take a past participle, which is why SIGNAL_PLAIN_PHRASES is
    # written in that form.
    lead = (
        "The candidate was found to have"
        if disposition_prefix(active) == "Confirmed"
        else "The candidate may have"
    )
    return BehaviorSummarySection(
        text=f"{lead} {concerns}. A reviewer should watch the moments listed below and decide."
    )


def build_recommendation_section(bundle: DeliberationBundle) -> RecommendationSection:
    active = active_signals(bundle)
    source_types = list(bundle.recommendation.independent_source_types)
    if bundle.recommendation.category == "CLEAR":
        rejected = bundle.rejected_signal_count
        suffix = "s" if rejected != 1 else ""
        return RecommendationSection(
            category=bundle.recommendation.category,
            confidence=bundle.recommendation.confidence,
            reasoning=(
                f"{rejected} behaviour{suffix} were assessed during this session. "
                "All were determined to have adequate explanations and do not constitute "
                "evidence of an integrity violation."
            ),
            recommendation="No integrity concerns were identified. No action required.",
            independent_source_count=0,
            independent_source_types=[],
            corroboration_downgrade_applied=bundle.corroboration_downgrade_applied,
            supporting_signal_ids=[],
        )
    if not active:
        reasoning = (bundle.recommendation.reasoning or "").strip()
        if not reasoning or _looks_like_clear_prose(reasoning):
            reasoning = "Independent evidence findings require human review."
        recommendation_text = (bundle.recommendation.recommendation or "").strip()
        if not recommendation_text or _looks_like_clear_prose(recommendation_text):
            recommendation_text = (
                "Review recommended — an independently derived finding warrants human judgment."
            )
        return RecommendationSection(
            category=bundle.recommendation.category,
            confidence=bundle.recommendation.confidence,
            reasoning=reasoning,
            recommendation=recommendation_text,
            independent_source_count=len(source_types),
            independent_source_types=[str(t) for t in source_types],
            corroboration_downgrade_applied=bundle.corroboration_downgrade_applied,
            supporting_signal_ids=bundle.recommendation.supporting_signals,
        )
    labels = ", ".join(sorted({humanize_signal_type(s.signal_type) for s in active}))
    story = active[0].integrity_story
    primary = (
        f"{story.headline}. {story.what_happened}"
        if story and story.headline
        else to_reviewer_prose(active[0].hypothesis_assisted.supporting[0])
    )
    disposition = disposition_prefix(active)
    reasoning = (
        f"{primary} ({disposition}: {labels}.)"
        if primary
        else f"{'A concern' if len(active) == 1 else str(len(active)) + ' concerns'} — "
        f"{disposition.lower()}: {labels}."
    )
    # Prefer what the model actually wrote. The two fixed strings below replaced
    # its recommendation on every non-clear review, so a reviewer read the same
    # sentence whether the evidence was a dictated answer or a glance.
    recommendation = bundle.recommendation.recommendation.strip() or (
        "Escalate for review — strong corroborated evidence from multiple independent sources."
        if bundle.recommendation.category == "STRONG_EVIDENCE"
        else "Review recommended — a behaviour of concern warrants human judgment."
    )
    reasoning = bundle.recommendation.reasoning.strip() or reasoning
    return RecommendationSection(
        category=bundle.recommendation.category,
        confidence=bundle.recommendation.confidence,
        reasoning=reasoning,
        recommendation=recommendation,
        independent_source_count=len(source_types),
        independent_source_types=[str(t) for t in source_types],
        corroboration_downgrade_applied=bundle.corroboration_downgrade_applied,
        supporting_signal_ids=[s.signal_id for s in active],
    )


def build_key_reasons(bundle: DeliberationBundle) -> KeyReasonsSection:
    active = active_signals(bundle)
    if not active:
        return KeyReasonsSection(reasons=[])
    from_model = [
        r.strip()
        for r in (bundle.recommendation.key_reasons or [])
        if r and r.strip()
    ][:5]
    if from_model:
        return KeyReasonsSection(reasons=from_model)
    reasons = []
    seen: set[str] = set()
    for signal in active:
        headline = signal.integrity_story.headline if signal.integrity_story else None
        reason = headline or to_reviewer_prose(
            signal.hypothesis_assisted.supporting[0]
            if signal.hypothesis_assisted.supporting
            else None
        ) or humanize_signal_type(signal.signal_type)
        if reason not in seen:
            seen.add(reason)
            reasons.append(reason)
    return KeyReasonsSection(reasons=reasons)


def _get_nested_value(obs: Any, path: str) -> Any:
    cur: Any = mapping_view(obs)
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def _observation_for_cite(perception_bundle: Any, cite: str) -> Any | None:
    """The observation a citation actually refers to.

    A window holds one observation per perception event, so returning the first
    row in the window would time the signal by whichever event happened to come
    first — a 3s glance standing in for the 40s phone the citation names. Prefer
    the row where the cited field really carries the cited value.
    """

    parsed = parse_observation_citation(cite)
    window_id = parsed[0] if parsed else cite.split(".")[0]
    field_path = parsed[1] if parsed else None
    claimed = parsed[2] if parsed else None

    in_window = [
        obs
        for obs in (attr(perception_bundle, "observations", default=[]) or [])
        if attr(obs, "video_available", "videoAvailable")
        and attr(obs, "window_id", "windowId") == window_id
    ]
    if not in_window:
        return None
    resolver = FIELD_PATH_RESOLVERS.get(field_path) if field_path else None
    if resolver:
        for obs in in_window:
            actual = _get_nested_value(obs, resolver)
            if actual is None or str(actual).upper() == "UNKNOWN":
                continue
            if claimed is None or str(actual).lower() == claimed.lower():
                return obs
    return in_window[0]


def signal_time_span(
    signal: ValidatedSignal,
    perception_bundle: Any,
    machine_facts_bundle: Any,
) -> dict[str, Any]:
    observations = []
    seen: set[str] = set()
    for cite in signal.observations_cited:
        obs = _observation_for_cite(perception_bundle, cite)
        if obs is None:
            continue
        wid = str(attr(obs, "window_id", "windowId"))
        if wid in seen:
            continue
        seen.add(wid)
        observations.append(obs)

    facts = attr(machine_facts_bundle, "facts", default=[]) or []
    cited = [f for f in facts if fact_kind(f) in signal.machine_facts_cited]
    # A cited kind like MCQ_ANSWER_SELECTED fires throughout the exam, so
    # matching on kind alone stretches the span from the first occurrence to the
    # last — the whole session — and drags the clip anchor with it. The facts
    # that corroborate a signal are the ones near the windows it actually cites,
    # the same rule _resolve_story_machine_facts already applies.
    if observations:
        obs_start = min(int(attr(o, "start_ms", "startMs", default=0) or 0) for o in observations)
        obs_end = max(int(attr(o, "end_ms", "endMs", default=0) or 0) for o in observations)
        facts_used = [
            f
            for f in cited
            if obs_start - FACT_WINDOW_PAD_MS <= fact_start_ms(f) <= obs_end + FACT_WINDOW_PAD_MS
        ]
    else:
        # Telemetry-only signal: no window to anchor to, so keep the first
        # contiguous burst rather than every occurrence in the session.
        facts_used = []
        for fact in sorted(cited, key=fact_start_ms):
            if facts_used and fact_start_ms(fact) - fact_start_ms(facts_used[-1]) > FACT_CLUSTER_GAP_MS:
                break
            facts_used.append(fact)

    starts = [
        int(attr(o, "start_ms", "startMs", default=0) or 0) for o in observations
    ] + [fact_start_ms(f) for f in facts_used]
    ends = [
        int(attr(o, "end_ms", "endMs", default=0) or 0) for o in observations
    ] + [fact_start_ms(f) + 1_000 for f in facts_used]

    if not starts:
        ts = 0
        for cite in signal.observations_cited:
            match = re.match(r"^w_(\d+)_\d+$", cite.split(".")[0])
            if match:
                ts = int(match.group(1))
                break
        if ts == 0:
            for kind in signal.machine_facts_cited:
                for fact in facts:
                    if fact_kind(fact) == kind:
                        ts = fact_start_ms(fact)
                        break
                if ts:
                    break
        return {
            "start_ms": ts,
            "end_ms": ts,
            "window_count": 0,
            "observations": [],
            "facts_used": [],
        }

    return {
        "start_ms": min(starts),
        "end_ms": max(ends),
        "window_count": len(observations),
        "observations": observations,
        "facts_used": facts_used,
    }


def signal_clip_anchor_ms(
    start_ms: int,
    end_ms: int,
    observations: list[Any],
    facts_used: list[Any],
) -> int:
    dur_ms = end_ms - start_ms
    if dur_ms <= 5_000:
        return start_ms
    if dur_ms <= LONG_SPAN_MS:
        return round((start_ms + end_ms) / 2)
    first_obs = min(
        (int(attr(o, "start_ms", "startMs", default=0) or 0) for o in observations),
        default=None,
    )
    first_fact = min((fact_start_ms(f) for f in facts_used), default=None)
    candidates = [
        v
        for v in (first_obs, first_fact)
        if v is not None and start_ms <= v <= end_ms
    ]
    return min(candidates) if candidates else start_ms


def _resolve_story_machine_facts(
    signal: ValidatedSignal,
    machine_facts_bundle: Any,
    time_range_ms: tuple[int, int],
    preferred_kinds: list[str],
) -> list[dict[str, object]]:
    kinds = set(preferred_kinds or signal.machine_facts_cited)
    pad = 15_000
    t0, t1 = time_range_ms
    facts = attr(machine_facts_bundle, "facts", default=[]) or []
    hits = [
        f
        for f in facts
        if (not kinds or fact_kind(f) in kinds)
        and t0 - pad <= fact_start_ms(f) <= t1 + pad
    ]
    out: list[dict[str, object]] = []
    seen: set[str] = set()
    for fact in hits[:8]:
        key = f"{fact_kind(fact)}|{fact_start_ms(fact)}"
        if key in seen:
            continue
        seen.add(key)
        detail = mapping_view(attr(fact, "detail", default={}) or {})
        entry: dict[str, object] = {"kind": fact_kind(fact), "tMs": fact_start_ms(fact)}
        if isinstance(detail.get("charsAdded"), (int, float)):
            entry["charsAdded"] = int(detail["charsAdded"])
        excerpt = detail.get("pastedExcerpt")
        if isinstance(excerpt, str) and excerpt:
            entry["pastedExcerpt"] = excerpt[:300]
        element_id = detail.get("elementId")
        if isinstance(element_id, str):
            entry["elementId"] = element_id
        out.append(entry)
    return out


def resolve_section_id(
    clip_ref: ClipRef | None,
    time_range_ms: tuple[int, int],
    media_index: list[MediaIndexEntry],
    sections: list[Any] | None = None,
) -> str | None:
    """Resolve the single section a signal belongs to.

    Tries the anchoring chunk's section first (deterministic, immune to clock
    skew), then falls back to bracketing the signal's midpoint against media
    chunks and finally against canonical session sections.
    """
    if clip_ref is not None and clip_ref.segments:
        anchor_chunk_id = clip_ref.segments[0].chunk_id
        for entry in media_index:
            if entry.chunk_id == anchor_chunk_id and entry.section_id:
                return entry.section_id

    mid_ms = (time_range_ms[0] + time_range_ms[1]) // 2
    for entry in media_index:
        if entry.section_id and entry.session_start_ms <= mid_ms <= entry.session_end_ms:
            return entry.section_id

    for section in sections or []:
        section_id = attr(section, "section_id", "sectionId")
        if not section_id:
            continue
        start_ms = int(attr(section, "start_ms", "startMs", default=0) or 0)
        end_ms = int(attr(section, "end_ms", "endMs", default=0) or 0)
        if start_ms <= mid_ms <= end_ms:
            return section_id
    return None


def build_integrity_stories(
    *,
    deliberation_bundle: DeliberationBundle,
    machine_facts_bundle: Any,
    perception_bundle: Any,
    correlated_signals: Any | None,
    covered_windows: int,
    media_index: list[MediaIndexEntry],
    sections: list[Any] | None = None,
) -> IntegrityStoriesSection:
    active = active_signals(deliberation_bundle)
    episode_by_signal: dict[str, Any] = {}
    for episode in deliberation_bundle.episode_analysis or []:
        for sid in episode.emitted_signal_ids or []:
            episode_by_signal[sid] = episode

    contributions = attr(correlated_signals, "contributions", default=[]) or []
    stories: list[IntegrityStoryEntry] = []

    for signal in active:
        span = signal_time_span(signal, perception_bundle, machine_facts_bundle)
        episode = episode_by_signal.get(signal.signal_id)
        raw_story = signal.integrity_story or synthesize_integrity_story_fallback(
            signal_type=signal.signal_type,
            episode_summary=episode.episode_summary if episode else None,
            assisted_supporting=signal.hypothesis_assisted.supporting,
            honest_supporting=signal.hypothesis_honest.supporting,
            window_ids=episode.window_ids if episode else [attr(o, "window_id", "windowId") for o in span["observations"]],
            machine_fact_kinds=signal.machine_facts_cited,
            factor_ids=episode.factor_ids if episode else [],
        )
        window_ids = raw_story.proof_anchors.window_ids or [
            str(attr(o, "window_id", "windowId")) for o in span["observations"]
        ]
        start_ms = episode.time_range_ms[0] if episode and episode.time_range_ms else span["start_ms"]
        end_ms = episode.time_range_ms[1] if episode and episode.time_range_ms else span["end_ms"]
        time_range_ms = (int(start_ms), int(end_ms))
        seek_hints = raw_story.proof_anchors.seek_ms_hints
        video_seek_ms = (
            seek_hints[0]
            if covered_windows > 0 and seek_hints
            else (
                signal_clip_anchor_ms(
                    span["start_ms"],
                    span["end_ms"],
                    span["observations"],
                    span["facts_used"],
                )
                if covered_windows > 0
                else None
            )
        )
        machine_facts = _resolve_story_machine_facts(
            signal,
            machine_facts_bundle,
            time_range_ms,
            raw_story.proof_anchors.machine_fact_kinds,
        )
        # ``time_range_ms`` is the episode's narrative span — minutes of context
        # the reviewer does not need to watch. The proof clip is the evidence
        # window itself, held to one chunk so the card offers a single clip
        # sitting on the moment rather than a run of consecutive chunks.
        evidence_start = int(span["start_ms"])
        evidence_end = int(span["end_ms"])
        clip_ref = (
            resolve_offset_clip_ref(
                event_ms=evidence_start,
                duration_ms=max(OBSERVATION_CLIP_DURATION_MS, evidence_end - evidence_start),
                media_index=media_index,
                single_segment=True,
                prefer_evidence_type=evidence_stream_for_event(signal.signal_type),
            )
            if covered_windows > 0 and media_index
            else None
        )
        factor_ids = raw_story.proof_anchors.factor_ids or (episode.factor_ids if episode else [])
        supporting_factors = [
            c
            for c in contributions
            if attr(c, "factor_id", "factorId") in factor_ids
            or (
                attr(c, "time_range_ms", "timeRangeMs", default=(0, 0))[0] <= time_range_ms[1]
                and attr(c, "time_range_ms", "timeRangeMs", default=(0, 0))[1] >= time_range_ms[0]
            )
        ]
        question_numbers = raw_story.involved_questions or sorted(
            {
                int(attr(c, "question_number", "questionNumber"))
                for c in supporting_factors
                if attr(c, "question_number", "questionNumber") is not None
            }
        )
        section_id = resolve_section_id(clip_ref, time_range_ms, media_index, sections)
        stories.append(
            IntegrityStoryEntry(
                story_id=f"story_{len(stories) + 1}_{signal.signal_id}",
                signal_id=signal.signal_id,
                signal_type=signal.signal_type,
                headline=raw_story.headline,
                what_happened=to_reviewer_prose(raw_story.what_happened) or raw_story.what_happened,
                why_it_matters=raw_story.why_it_matters,
                honest_alternative=raw_story.honest_alternative,
                severity=raw_story.severity,
                resolution=signal.resolution,
                confidence=min(signal.confidence, deliberation_bundle.recommendation.confidence),
                time_range_ms=time_range_ms,
                question_numbers=question_numbers,
                section_id=section_id,
                proof=IntegrityStoryProof(
                    time_range_ms=time_range_ms,
                    video_seek_ms=video_seek_ms,
                    clip_ref=clip_ref,
                    machine_facts=machine_facts,
                    audio_quotes=(raw_story.proof_anchors.audio_quotes_en or [])[:3],
                    perception_window_ids=window_ids if covered_windows > 0 else [],
                    supporting_factor_ids=[
                        str(attr(c, "factor_id", "factorId")) for c in supporting_factors
                    ],
                    telemetry_only=True if covered_windows == 0 else None,
                ),
            )
        )
    return IntegrityStoriesSection(stories=stories)


def _union_interval_length_ms(intervals: list[tuple[int, int]]) -> int:
    if not intervals:
        return 0
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return sum(max(0, end - start) for start, end in merged)


def build_unknown_panel(
    perception_bundle: Any,
    master_timeline: Any,
) -> UnknownPanelSection:
    covered = int(attr(perception_bundle, "covered_windows", "coveredWindows", default=0) or 0)
    observations = attr(perception_bundle, "observations", default=[]) or []
    video_entirely_absent = covered == 0 and observations and all(
        not attr(o, "video_available", "videoAvailable") for o in observations
    )
    if video_entirely_absent:
        session_ms = int(
            attr(master_timeline, "duration_ms", "durationMs", default=0)
            or attr(perception_bundle, "duration_ms", "durationMs", default=0)
            or 0
        )
        return UnknownPanelSection(
            intervals=[],
            total_intervals=0,
            total_duration_ms=session_ms,
            modality_absent_summary="No video for this session (keystroke-only).",
            disclaimer=(
                "Video modality was not present. Analysis relies on keystroke and application "
                "telemetry only. Absence of video does not increase or decrease confidence "
                "in keystroke findings."
            ),
        )

    raw_intervals: list[UnknownInterval] = []
    for obs in observations:
        if attr(obs, "video_available", "videoAvailable"):
            continue
        start_ms = int(attr(obs, "start_ms", "startMs", default=0) or 0)
        end_ms = int(attr(obs, "end_ms", "endMs", default=start_ms) or start_ms)
        raw_intervals.append(
            UnknownInterval(
                start_ms=start_ms,
                end_ms=end_ms,
                duration_ms=max(0, end_ms - start_ms),
                reason=str(
                    attr(obs, "missing_video_reason", "missingVideoReason", default="no_video_coverage")
                ),
                window_id=str(attr(obs, "window_id", "windowId", default="")),
            )
        )

    unknown_segments = attr(master_timeline, "unknown_segments", "unknownSegments", default=[]) or []
    for seg in unknown_segments:
        seg_start = int(attr(seg, "start_offset_ms", "startOffsetMs", default=0) or 0)
        seg_end = int(attr(seg, "end_offset_ms", "endOffsetMs", default=0) or 0)
        if any(i.start_ms <= seg_start and i.end_ms >= seg_end for i in raw_intervals):
            continue
        raw_intervals.append(
            UnknownInterval(
                start_ms=seg_start,
                end_ms=seg_end,
                duration_ms=max(0, seg_end - seg_start),
                reason=str(attr(seg, "reason", default="unknown_gap")),
                window_id="",
            )
        )

    raw_intervals.sort(key=lambda i: i.start_ms)
    total_duration_ms = _union_interval_length_ms([(i.start_ms, i.end_ms) for i in raw_intervals])
    return UnknownPanelSection(
        intervals=raw_intervals,
        total_intervals=len(raw_intervals),
        total_duration_ms=total_duration_ms,
        disclaimer=(
            "UNKNOWN intervals are excluded from evidence and contribute to neither hypothesis. "
            "Coverage gaps do not increase or decrease confidence in any finding."
        ),
    )


def build_smart_student_notes(
    deliberation_bundle: DeliberationBundle,
    statistical_baseline: Any,
) -> SmartStudentNotesSection:
    notes: list[SmartStudentNote] = []
    for rejection in deliberation_bundle.rejected_signals:
        if rejection.rejected_by != "SmartStudentGuard":
            continue
        notes.append(
            SmartStudentNote(
                behavior=humanize_signal_type(rejection.signal_type),
                explanation=rejection.reason,
                source="smart_student_guard_rejection",
            )
        )
    machine_metrics = attr(
        statistical_baseline, "machine_fact_metrics", "machineFactMetrics", default={}
    ) or {}
    if int(attr(machine_metrics, "large_paste_count", "largePasteCount", default=0) or 0) == 0:
        notes.append(
            SmartStudentNote(
                behavior="No unusual text pasting observed",
                explanation=(
                    "No unusually large blocks of text were pasted into the exam window "
                    "during the session."
                ),
                source="baseline_within_normal",
            )
        )
    coverage = attr(statistical_baseline, "coverage_metrics", "coverageMetrics", default={}) or {}
    usable = float(attr(coverage, "usable_window_ratio", "usableWindowRatio", default=0) or 0)
    if usable >= 0.8:
        notes.append(
            SmartStudentNote(
                behavior="Good video coverage throughout the session",
                explanation=(
                    f"{round(usable * 100)}% of the session was captured on video. "
                    "The analysis has full context for this session."
                ),
                source="baseline_within_normal",
            )
        )
    return SmartStudentNotesSection(notes=notes)


def build_contextual_events_section(
    contextual_events: list[Any] | None = None,
) -> ContextualEventsSection:
    events: list[ContextualSpeechEvent] = []
    for event in contextual_events or []:
        proof = attr(event, "speech_proof", "speechProof")
        events.append(
            ContextualSpeechEvent(
                event_type=str(attr(event, "event_type", "eventType", default="")),
                timestamp_window_ms=tuple(
                    attr(
                        event,
                        "timestamp_window_ms",
                        "timestampWindowMs",
                        default=(0, 0),
                    )
                ),  # type: ignore[arg-type]
                clip_start_ms=int(
                    attr(event, "clip_start_ms", "clipStartMs", default=0) or 0
                ),
                clip_end_ms=int(
                    attr(event, "clip_end_ms", "clipEndMs", default=0) or 0
                ),
                speech_language=attr(proof, "speech_language", "speechLanguage"),
                code_mixing=attr(proof, "code_mixing", "codeMixing"),
                conversation_summary_en=attr(
                    proof, "conversation_summary_en", "conversationSummaryEn"
                ),
                notable_phrases_original=attr(
                    proof,
                    "notable_phrases_original",
                    "notablePhrasesOriginal",
                ),
            )
        )
    return ContextualEventsSection(events=events)
