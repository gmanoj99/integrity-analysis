"""Scope 4 deliberation engine."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

from ..contracts.deliberation import (
    DeliberationBundle,
    DeliberationProvenance,
    DeliberationRecommendation,
    EpisodeAnalysisEntry,
    HypothesisArm,
    RejectedSignal,
    ValidatedSignal,
)
from ..prompts.deliberation import (
    DELIBERATION_PROMPT_VERSION,
    DELIBERATION_SYSTEM_PROMPT,
    TRACK2_CAPS,
    build_deliberation_user_prompt,
)
from ..scoring.unified_score import compute_unified_score
from .episodes import (
    DeliberationEpisode,
    build_episode_analysis_entries,
    build_episode_inventory,
    serialize_episode_inventory,
)
from ..duck_helpers import attr, fact_kind
from .rules import (
    DELIBERATION_MODEL_VERSION,
    FIELD_PATH_RESOLVERS,
    KNOWN_BASELINE_METRICS,
    RawCandidateSignal,
    RawEpisodeAnalysis,
    SIGNAL_TYPES,
    compute_deliberation_version_hash,
    compute_informative_content_ratio,
    derive_source_types,
    filter_audio_quotes_against_perception,
    parse_integrity_story,
    synthesize_integrity_story_fallback,
    validate_citation_rule,
    validate_episode_ref,
    validate_no_fact_invention,
    validate_unknown_rule,
    validate_value_resolving_citations,
)


@dataclass(slots=True)
class DeliberationInput:
    candidate_id: str
    assessment_id: str
    machine_facts_bundle: Any
    perception_bundle: Any
    statistical_baseline: Any
    master_timeline: Any | None = None
    correlated_signals: Any | None = None
    video_findings: list[Any] | None = None
    perception_version_hash: str = "unknown"
    baseline_version_hash: str = "unknown"


LlmCallable = Callable[[str], str]


def _fmt_ms(ms: int) -> str:
    seconds = round(ms / 1000)
    minutes = seconds // 60
    sec = seconds % 60
    return f"{minutes}m{sec}s" if minutes else f"{sec}s"


def _extract_balanced_object(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index, char in enumerate(text[start:], start):
        if escaped:
            escaped = False
            continue
        if char == "\\" and in_string:
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _pick_raw_text(raw: Mapping[str, Any], camel: str, snake: str, default: str = "") -> str:
    value = raw.get(camel)
    if value is None:
        value = raw.get(snake)
    if value is None:
        return default
    return str(value).strip()


def _pick_raw_list(raw: Mapping[str, Any], camel: str, snake: str) -> list[str]:
    value = raw.get(camel)
    if value is None:
        value = raw.get(snake)
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if item is not None and str(item).strip()]


def _looks_like_clear_prose(text: str) -> bool:
    lowered = text.lower()
    return any(
        marker in lowered
        for marker in (
            "no significant integrity",
            "no integrity concerns",
            "no concerns were identified",
            "adequate explanations",
            "no action required",
        )
    )


def _parse_confidence(value: Any, default: float = 0.5) -> float:
    """Match TS: only numeric confidence is trusted; strings default to 0.5."""
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    return default


def parse_raw_output(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.I)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as first_err:
        balanced = _extract_balanced_object(cleaned)
        candidates = [balanced, cleaned + "}"]
        parsed = None
        for candidate in candidates:
            if not candidate:
                continue
            try:
                parsed = json.loads(candidate)
                break
            except json.JSONDecodeError:
                continue
        if parsed is None:
            raise first_err
    if not isinstance(parsed, dict):
        raise ValueError("Deliberation output is not a JSON object")
    return parsed


def _serialize_machine_facts(bundle: Any) -> str:
    facts = getattr(bundle, "facts", []) or []
    lines = [
        f"SESSION: duration={_fmt_ms(int(_attr(bundle, 'duration_ms', 'durationMs', default=0) or 0))}",
        "MACHINE_FACT_DETAIL_ROWS:",
    ]
    counts: dict[str, int] = {}
    for fact in facts:
        kind = _fact_kind(fact)
        if kind:
            counts[kind] = counts.get(kind, 0) + 1
    lines.append(
        "COUNTS: "
        + (", ".join(f"{kind}={count}" for kind, count in sorted(counts.items())) or "(none)")
    )
    for fact in facts[: TRACK2_CAPS["machine_fact_detail_rows"]]:
        kind = _fact_kind(fact)
        start = int(_attr(fact, "start_offset_ms", "startOffsetMs", default=0) or 0)
        end = int(_attr(fact, "end_offset_ms", "endOffsetMs", default=start) or start)
        detail = _attr(fact, "detail", default={}) or {}
        lines.append(
            f"  - {kind} [{start},{end}] source={_attr(fact, 'evidence_source', 'evidenceSource', default='unknown')} "
            f"attribution={_attr(fact, 'attribution', default='unclear')} detail={json.dumps(detail, default=str)}"
        )
    return "\n".join(lines)


def _fact_kind(fact: Any) -> str:
    return fact_kind(fact)


def _attr(obj: Any, *names: str, default: Any = None) -> Any:
    return attr(obj, *names, default=default)


def _serialize_baseline(baseline: Any) -> str:
    if hasattr(baseline, "model_dump"):
        raw = baseline.model_dump(by_alias=True)
    elif isinstance(baseline, Mapping):
        raw = dict(baseline)
    else:
        raw = vars(baseline)
    lines = ["STATISTICAL_BASELINE:"]
    for section_name, section in raw.items():
        if not isinstance(section, Mapping):
            continue
        for metric, value in section.items():
            if metric in KNOWN_BASELINE_METRICS and value is not None:
                lines.append(f"  {metric}={json.dumps(value, default=str)}")
    if len(lines) == 1:
        lines.append("  (none)")
    return "\n".join(lines)


def _serialize_correlated_signals(bundle: Any | None) -> str:
    if bundle is None:
        return "CORRELATED_SIGNALS:\n  (none)"
    contributions = _attr(bundle, "contributions", default=[]) or []
    lines = ["CORRELATED_SIGNALS:"]
    for c in contributions[:20]:
        factor = _attr(c, "factor_id", "factorId")
        score = _attr(c, "score_contribution", "scoreContribution")
        lines.append(f"  - {factor}: score={score}")
    return "\n".join(lines)


def _build_citation_inventory(machine_facts_bundle: Any, perception_bundle: Any) -> str:
    facts = getattr(machine_facts_bundle, "facts", []) or []
    kinds = sorted({_fact_kind(f) for f in facts if _fact_kind(f)})
    observations = getattr(perception_bundle, "observations", []) or []
    lines = [
        "=== CITATION INVENTORY ===",
        "MACHINE_FACT_KINDS (copy a kind verbatim into machineFactsCited):",
        "  " + (", ".join(kinds) if kinds else "(none)"),
        'OBSERVATION CITATIONS (copy verbatim as "windowId.fieldPath=value"):',
    ]
    citation_count = 0
    for observation in observations:
        if citation_count >= TRACK2_CAPS["events"]:
            break
        raw = (
            observation.model_dump(by_alias=True)
            if hasattr(observation, "model_dump")
            else dict(observation)
            if isinstance(observation, Mapping)
            else {}
        )
        window_id = str(raw.get("windowId") or raw.get("window_id") or "")
        if not window_id:
            continue
        for path in FIELD_PATH_RESOLVERS:
            node: Any = raw
            for segment in path.split("."):
                if not isinstance(node, Mapping) or segment not in node:
                    node = None
                    break
                node = node[segment]
            if node is None or str(node).upper() == "UNKNOWN":
                continue
            lines.append(f"  {window_id}.{path}={str(node).lower() if isinstance(node, bool) else node}")
            citation_count += 1
            if citation_count >= TRACK2_CAPS["events"]:
                break
        audio = raw.get("audio")
        if isinstance(audio, Mapping) and audio.get("conversationSummaryEn"):
            lines.append(
                f"  {window_id}.audio.conversationSummaryEn={audio['conversationSummaryEn']}"
            )
    if citation_count == 0:
        lines.append("  (none)")
    lines.extend(
        [
            "VALID_FIELD_PATHS:",
            "  " + ", ".join(FIELD_PATH_RESOLVERS),
            "UNKNOWN values are not citable.",
        ]
    )
    return "\n".join(lines)


def _serialize_perception_observations(perception_bundle: Any) -> str:
    observations = getattr(perception_bundle, "observations", []) or []
    lines = ["PERCEPTION_OBSERVATIONS:"]
    for observation in observations[: TRACK2_CAPS["events"]]:
        raw = (
            observation.model_dump(by_alias=True)
            if hasattr(observation, "model_dump")
            else dict(observation)
            if isinstance(observation, Mapping)
            else {}
        )
        window_id = raw.get("windowId") or raw.get("window_id")
        values: list[str] = []
        for path in FIELD_PATH_RESOLVERS:
            node: Any = raw
            for segment in path.split("."):
                if not isinstance(node, Mapping) or segment not in node:
                    node = None
                    break
                node = node[segment]
            if node is not None and str(node).upper() != "UNKNOWN":
                values.append(f"{path}={node}")
        if values:
            lines.append(
                f"  - {window_id} [{raw.get('startMs', 0)},{raw.get('endMs', 0)}]: "
                + "; ".join(values)
            )
        audio = raw.get("audio")
        if isinstance(audio, Mapping) and audio.get("conversationSummaryEn"):
            lines.append(f"    conversationSummaryEn={audio['conversationSummaryEn']}")
    if len(lines) == 1:
        lines.append("  (none)")
    return "\n".join(lines)


def _parse_raw_signals(raw: dict[str, Any]) -> tuple[list[RawEpisodeAnalysis], list[RawCandidateSignal]]:
    def pick(item: Mapping[str, Any], camel: str, snake: str, default: Any = None) -> Any:
        value = item.get(camel)
        return item.get(snake, default) if value is None else value

    episodes: list[RawEpisodeAnalysis] = []
    for item in raw.get("episodeAnalysis") or raw.get("episode_analysis") or []:
        if not isinstance(item, dict):
            continue
        episodes.append(
            RawEpisodeAnalysis(
                episode_id=str(pick(item, "episodeId", "episode_id", "unknown")),
                time_range=str(pick(item, "timeRange", "time_range", "")),
                window_ids=[str(v) for v in pick(item, "windowIds", "window_ids", []) or []],
                machine_facts_in_range=[
                    str(v)
                    for v in pick(
                        item, "machineFactsInRange", "machine_facts_in_range", []
                    )
                    or []
                ],
                episode_summary=str(pick(item, "episodeSummary", "episode_summary", "")),
                suspicious_behavior_type=str(
                    pick(item, "suspiciousBehaviorType", "suspicious_behavior_type", "none")
                ),
                will_emit_signal=pick(item, "willEmitSignal", "will_emit_signal", False)
                is True,
                reason_not_signalled=pick(
                    item, "reasonNotSignalled", "reason_not_signalled"
                ),
            )
        )
    signals: list[RawCandidateSignal] = []
    for item in raw.get("candidateSignals") or raw.get("candidate_signals") or []:
        if not isinstance(item, dict):
            continue
        signals.append(
            RawCandidateSignal(
                signal_type=str(pick(item, "signalType", "signal_type", "")),
                episode_ref=(
                    str(pick(item, "episodeRef", "episode_ref"))
                    if pick(item, "episodeRef", "episode_ref")
                    else None
                ),
                hypothesis_honest=item.get("hypothesis_honest") or {"supporting": [], "contradicting": []},
                hypothesis_assisted=item.get("hypothesis_assisted") or {"supporting": [], "contradicting": []},
                resolution=str(item.get("resolution", "ambiguous")),
                confidence=_parse_confidence(item.get("confidence", 0.5)),
                innocent_explanation_considered=pick(
                    item,
                    "innocentExplanationConsidered",
                    "innocent_explanation_considered",
                    False,
                )
                is True,
                why_rejected=str(pick(item, "whyRejected", "why_rejected", "")),
                machine_facts_cited=[
                    str(v)
                    for v in pick(item, "machineFactsCited", "machine_facts_cited", [])
                    or []
                ],
                observations_cited=[
                    str(v)
                    for v in pick(item, "observationsCited", "observations_cited", [])
                    or []
                ],
                baseline_metrics_cited=[
                    str(v)
                    for v in pick(
                        item, "baselineMetricsCited", "baseline_metrics_cited", []
                    )
                    or []
                ],
                integrity_story=pick(item, "integrityStory", "integrity_story"),
            )
        )
    return episodes, signals


def build_deliberation_prompt(input_data: DeliberationInput) -> str:
    """Serialize the deterministic Scope 1–3.5 inventory for one Pro call."""

    episode_inventory = build_episode_inventory(
        correlated_signals=input_data.correlated_signals,
        video_findings=input_data.video_findings,
        machine_facts_bundle=input_data.machine_facts_bundle,
        perception_bundle=input_data.perception_bundle,
    )
    user_prompt = build_deliberation_user_prompt(
        citation_inventory=_build_citation_inventory(
            input_data.machine_facts_bundle, input_data.perception_bundle
        ),
        episode_inventory_text=serialize_episode_inventory(episode_inventory),
        machine_facts=_serialize_machine_facts(input_data.machine_facts_bundle),
        perception_obs=_serialize_perception_observations(input_data.perception_bundle),
        baseline_stats=_serialize_baseline(input_data.statistical_baseline),
        correlated_signals_text=_serialize_correlated_signals(
            input_data.correlated_signals
        ),
    )
    return f"{DELIBERATION_SYSTEM_PROMPT}\n\n{user_prompt}"


def build_deliberation_bundle(
    input_data: DeliberationInput,
    *,
    llm_call: LlmCallable | None = None,
    raw_llm_text: str | None = None,
) -> DeliberationBundle:
    """Build Scope 4 bundle. Pass ``raw_llm_text`` or ``llm_call`` for tests."""
    correlated_hash = "none"
    if input_data.correlated_signals is not None:
        bundle = input_data.correlated_signals
        logic = _attr(bundle, "logic_version", "logicVersion", default="")
        score = _attr(bundle, "total_weighted_score", "totalWeightedScore", default=0)
        factor_ids = [
            _attr(c, "factor_id", "factorId")
            for c in (_attr(bundle, "contributions", default=[]) or [])
        ]
        correlated_hash = hashlib.sha256(
            f"{logic}|{score}|{','.join(str(v) for v in factor_ids)}".encode()
        ).hexdigest()[:16]

    composite_hash = compute_deliberation_version_hash(
        input_data.perception_version_hash,
        input_data.baseline_version_hash,
        correlated_hash,
    )

    episode_inventory = build_episode_inventory(
        correlated_signals=input_data.correlated_signals,
        video_findings=input_data.video_findings,
        machine_facts_bundle=input_data.machine_facts_bundle,
        perception_bundle=input_data.perception_bundle,
    )

    prompt = build_deliberation_prompt(input_data)

    if raw_llm_text is None:
        if llm_call is None:
            raise ValueError("Provide raw_llm_text or llm_call")
        raw_llm_text = llm_call(prompt)
    raw = parse_raw_output(raw_llm_text)
    raw_episodes, raw_signals = _parse_raw_signals(raw)

    adjudicated_by_id = {
        ep.episode_id: ep
        for ep in raw_episodes
        if any(inv.episode_id == ep.episode_id for inv in episode_inventory)
    }

    facts = getattr(input_data.machine_facts_bundle, "facts", []) or []
    machine_fact_kinds_present = {_fact_kind(f) for f in facts}
    observations = getattr(input_data.perception_bundle, "observations", []) or []
    observations_by_window_id = {
        _attr(o, "window_id", "windowId"): o for o in observations
    }
    window_ids_present = set(observations_by_window_id.keys())
    episode_by_id = {ep.episode_id: ep for ep in episode_inventory}

    validated: list[ValidatedSignal] = []
    rejected: list[RejectedSignal] = []
    emitted_by_episode: dict[str, list[str]] = {}

    for raw_sig in raw_signals:
        if raw_sig.signal_type not in SIGNAL_TYPES:
            rejected.append(
                RejectedSignal(
                    signal_type=raw_sig.signal_type,
                    rejected_by="SignalTypeRule",
                    reason=f'Unknown signal type "{raw_sig.signal_type}"',
                )
            )
            continue
        episode_ref = validate_episode_ref(raw_sig, adjudicated_by_id)
        if not episode_ref.ok:
            rejected.append(
                RejectedSignal(
                    signal_type=raw_sig.signal_type,
                    rejected_by="EpisodeRefRule",
                    reason=episode_ref.reason or "Invalid episodeRef",
                )
            )
            continue
        value_result = validate_value_resolving_citations(raw_sig, observations_by_window_id)
        if not value_result.ok:
            rejected.append(
                RejectedSignal(
                    signal_type=raw_sig.signal_type,
                    rejected_by="ValueResolutionRule",
                    reason=value_result.reason or "Invalid observation citation",
                )
            )
            continue
        trimmed = RawCandidateSignal(
            signal_type=raw_sig.signal_type,
            episode_ref=raw_sig.episode_ref,
            hypothesis_honest=raw_sig.hypothesis_honest,
            hypothesis_assisted=raw_sig.hypothesis_assisted,
            resolution=raw_sig.resolution,
            confidence=raw_sig.confidence,
            innocent_explanation_considered=raw_sig.innocent_explanation_considered,
            why_rejected=raw_sig.why_rejected,
            machine_facts_cited=raw_sig.machine_facts_cited,
            # Only citations that actually resolved: falling back to the raw
            # list would readmit the ones validation just dropped, and
            # CitationRule below is the floor that must stay honest.
            observations_cited=value_result.kept_observations_cited,
            baseline_metrics_cited=raw_sig.baseline_metrics_cited,
            integrity_story=raw_sig.integrity_story,
        )
        if not validate_citation_rule(trimmed):
            rejected.append(
                RejectedSignal(
                    signal_type=raw_sig.signal_type,
                    rejected_by="CitationRule",
                    reason="Zero citations",
                )
            )
            continue
        if not validate_unknown_rule(trimmed):
            rejected.append(
                RejectedSignal(
                    signal_type=raw_sig.signal_type,
                    rejected_by="UnknownRule",
                    reason="All observation citations UNKNOWN",
                )
            )
            continue
        # No corroboration gate: the model decides whether an episode is a
        # signal. ConjunctionRule used to require >=2 citations across >=2 of
        # {machine_facts, observations, baseline}, which a camera-only session
        # can never satisfy — it has no evidential machine facts and an empty
        # baseline — so every such signal had to match a hand-maintained
        # whitelist of "definitive" field combinations or be discarded. The
        # rules kept below are the anti-hallucination ones: they check that the
        # model cited something and that what it cited exists.
        if not validate_no_fact_invention(
            trimmed,
            machine_fact_kinds_present,
            window_ids_present,
            set(KNOWN_BASELINE_METRICS),
        ):
            rejected.append(
                RejectedSignal(
                    signal_type=raw_sig.signal_type,
                    rejected_by="NoFactInventionRule",
                    reason="Cited fact/window/metric missing",
                )
            )
            continue

        inventory_episode = episode_by_id.get(str(trimmed.episode_ref))
        signal_id = f"sig_{len(validated) + 1}_{trimmed.signal_type}"
        emitted_by_episode.setdefault(str(trimmed.episode_ref), []).append(signal_id)
        story = parse_integrity_story(trimmed.integrity_story) or synthesize_integrity_story_fallback(
            signal_type=trimmed.signal_type,
            episode_summary=adjudicated_by_id.get(str(trimmed.episode_ref), RawEpisodeAnalysis(
                "", "", [], [], "", "none", False
            )).episode_summary,
            assisted_supporting=trimmed.hypothesis_assisted.get("supporting"),
            honest_supporting=trimmed.hypothesis_honest.get("supporting"),
            window_ids=inventory_episode.window_ids if inventory_episode else [],
            machine_fact_kinds=trimmed.machine_facts_cited,
            factor_ids=inventory_episode.factor_ids if inventory_episode else [],
        )
        known_summaries = [
            str(attr(o, "conversation_summary_en", "conversationSummaryEn", default="") or "")
            for o in observations
        ]
        filtered_quotes = filter_audio_quotes_against_perception(
            story.proof_anchors.audio_quotes_en,
            known_summaries,
        )
        if filtered_quotes != story.proof_anchors.audio_quotes_en:
            story = story.model_copy(
                update={
                    "proof_anchors": story.proof_anchors.model_copy(
                        update={"audio_quotes_en": filtered_quotes}
                    )
                }
            )
        resolution = trimmed.resolution if trimmed.resolution in {"honest", "assisted", "ambiguous"} else "ambiguous"
        validated.append(
            ValidatedSignal(
                signal_id=signal_id,
                signal_type=trimmed.signal_type,  # type: ignore[arg-type]
                hypothesis_honest=HypothesisArm(**trimmed.hypothesis_honest),
                hypothesis_assisted=HypothesisArm(**trimmed.hypothesis_assisted),
                resolution=resolution,  # type: ignore[arg-type]
                confidence=max(0.0, min(1.0, trimmed.confidence)),
                innocent_explanation_considered=trimmed.innocent_explanation_considered,
                why_rejected=trimmed.why_rejected,
                machine_facts_cited=trimmed.machine_facts_cited,
                observations_cited=trimmed.observations_cited,
                baseline_metrics_cited=trimmed.baseline_metrics_cited,
                source_types=derive_source_types(trimmed),
                integrity_story=story,
            )
        )

    informative_ratio, _, _ = compute_informative_content_ratio(input_data.perception_bundle)
    usable_ratio = float(
        _attr(
            _attr(input_data.statistical_baseline, "coverage_metrics", "coverageMetrics"),
            "usable_window_ratio",
            "usableWindowRatio",
            default=1.0,
        )
        or 1.0
    )
    unified = compute_unified_score(
        validated,
        input_data.video_findings or [],
        usable_window_ratio=usable_ratio,
        informative_content_ratio=informative_ratio,
    )
    category = unified.category
    final_confidence = unified.confidence
    capture_cap = unified.capture_quality_cap_applied
    informative_cap = unified.informative_content_cap_applied

    raw_category = _pick_raw_text(raw, "category", "category", "REVIEW_REQUIRED")
    downgraded = raw_category == "STRONG_EVIDENCE" and category != "STRONG_EVIDENCE"

    episode_entries = [
        EpisodeAnalysisEntry.model_validate(entry)
        for entry in build_episode_analysis_entries(
            episode_inventory, adjudicated_by_id, emitted_by_episode
        )
    ]

    behavior_summary = _pick_raw_text(raw, "behaviorSummary", "behavior_summary")
    recommendation_text = _pick_raw_text(raw, "recommendation", "recommendation")
    reasoning = _pick_raw_text(raw, "reasoning", "reasoning")
    if category != "CLEAR":
        if _looks_like_clear_prose(behavior_summary):
            behavior_summary = ""
        if _looks_like_clear_prose(reasoning):
            reasoning = "Independent evidence findings require human review."
        if _looks_like_clear_prose(recommendation_text):
            recommendation_text = (
                "Review recommended — an independently derived finding warrants human judgment."
            )

    recommendation = DeliberationRecommendation(
        behavior_summary=behavior_summary,
        recommendation=recommendation_text,
        category=category,
        confidence=final_confidence,
        reasoning=reasoning,
        key_reasons=_pick_raw_list(raw, "keyReasons", "key_reasons")[:8],
        independent_sources_count=len(unified.independent_source_types),
        independent_source_types=unified.independent_source_types,
        supporting_signals=unified.supporting_signal_ids,
    )

    return DeliberationBundle(
        candidate_id=input_data.candidate_id,
        assessment_id=input_data.assessment_id,
        produced_at=datetime.now(timezone.utc).isoformat(),
        model_version=DELIBERATION_MODEL_VERSION,
        prompt_version=DELIBERATION_PROMPT_VERSION,
        provenance=DeliberationProvenance(
            deliberation_prompt_version=DELIBERATION_PROMPT_VERSION,
            perception_version_hash=input_data.perception_version_hash,
            baseline_version_hash=input_data.baseline_version_hash,
            composite_version_hash=composite_hash,
        ),
        validated_signals=validated,
        rejected_signal_count=len(rejected),
        rejected_signals=rejected,
        recommendation=recommendation,
        episode_analysis=episode_entries,
        episode_inventory_count=len(episode_inventory),
        corroboration_downgrade_applied=downgraded,
        capture_quality_cap_applied=capture_cap,
        informative_content_ratio=informative_ratio,
        informative_content_cap_applied=informative_cap,
    )
