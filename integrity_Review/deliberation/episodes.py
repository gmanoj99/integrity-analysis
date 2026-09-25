from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

from ..duck_helpers import attr, fact_kind, fact_start_ms, fact_end_ms, mapping_view
from .rules import RawEpisodeAnalysis

EPISODE_INVENTORY_MAX_TIMED = 60
EPISODE_MERGE_GAP_MS = 30_000
EPISODE_MAX_SPAN_MS = 90_000

EPISODE_SEED_FACT_KINDS = frozenset(
    {
        "LARGE_PASTE",
        "WINDOW_BLUR",
        "TAB_SWITCH",
        "ACTIVITY_GAP",
        "CODE_SUBMISSION",
        "SCREEN_EXTERNAL_PASTE",
        "SCREEN_EXTERNAL_RESOURCE",
        "SCREEN_AI_ASSISTANT_UI",
        "SCREEN_SECONDARY_WORKSPACE",
        "SCREEN_FULLSCREEN_LOST",
    }
)

EpisodeOrigin = Literal["correlated_signal", "video_episode", "machine_fact"]


@dataclass(slots=True)
class EpisodeSeed:
    t_ms: int
    end_ms: int
    origin: EpisodeOrigin
    label: str
    evidence_refs: list[str] = field(default_factory=list)
    section_id: str | None = None
    question_number: int | None = None
    hint_window_ids: list[str] = field(default_factory=list)
    provisional_lane: Literal["red", "amber", "green"] | None = None


@dataclass(slots=True)
class DeliberationEpisode:
    episode_id: str
    scope: Literal["timed", "session"]
    time_range_ms: tuple[int, int]
    origins: list[EpisodeOrigin]
    factor_ids: list[str]
    video_event_types: list[str]
    machine_fact_kinds: list[str]
    window_ids: list[str]
    evidence_refs: list[str]
    modality_count: int
    section_id: str | None = None
    question_number: int | None = None
    provisional_lane: Literal["red", "amber", "green"] | None = None
    evidence_strength_hints: list[str] | None = None
    gaze_diverted_toward_person: bool | None = None


def window_ids_from_evidence_ref(evidence_ref: str | None) -> list[str]:
    if not evidence_ref:
        return []
    match = re.match(r"^perception:w=(.+)$", evidence_ref)
    if not match or not match.group(1):
        return []
    return [part.strip() for part in match.group(1).split("+") if part.strip()]


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(v for v in values if v))


def _attr(obj: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
        if isinstance(obj, Mapping) and name in obj:
            return obj[name]
    return default


def _fact_kind(fact: Any) -> str:
    return str(_attr(fact, "kind", default=""))


def _fact_start(fact: Any) -> int:
    return int(_attr(fact, "start_offset_ms", "startOffsetMs", default=0) or 0)


def _fact_end(fact: Any) -> int:
    end = _attr(fact, "end_offset_ms", "endOffsetMs")
    return int(end if end is not None else _fact_start(fact))


def _collect_provisional_unknown_seeds(
    provisional_unknowns: list[dict[str, Any]] | None,
) -> list[EpisodeSeed]:
    if not provisional_unknowns:
        return []
    seeds: list[EpisodeSeed] = []
    for item in provisional_unknowns:
        window_ids = list(item.get("windowIds") or item.get("window_ids") or [])
        if not window_ids:
            continue
        window = item.get("timestampWindowMs") or item.get("timestamp_window_ms") or (0, 0)
        t0, t1 = int(window[0]), int(window[1])
        seeds.append(
            EpisodeSeed(
                t_ms=t0,
                end_ms=max(t0, t1),
                origin="video_episode",
                label=str(item.get("eventType") or item.get("event_type") or "unknown_speech"),
                evidence_refs=[
                    f"perception:w={'+'.join(window_ids)}",
                    f"unknown:{item.get('reason', 'provisional')}",
                ],
                hint_window_ids=window_ids,
                provisional_lane="amber",
            )
        )
    return seeds


def _fmt_ms(ms: int) -> str:
    seconds = round(ms / 1000)
    minutes = seconds // 60
    sec = seconds % 60
    return f"{minutes}m{sec}s" if minutes else f"{sec}s"


def _fmt_range(time_range_ms: tuple[int, int]) -> str:
    return f"{_fmt_ms(time_range_ms[0])}–{_fmt_ms(time_range_ms[1])}"


def _collect_correlated_seeds(
    correlated_signals: Any | None,
) -> tuple[list[EpisodeSeed], list[EpisodeSeed]]:
    timed: list[EpisodeSeed] = []
    session: list[EpisodeSeed] = []
    if correlated_signals is None:
        return timed, session
    contributions = getattr(correlated_signals, "contributions", None) or []
    for c in contributions:
        factor_id = _attr(c, "factor_id", "factorId", default="unknown")
        time_range = _attr(c, "time_range_ms", "timeRangeMs", default=(0, 0))
        timing_basis = _attr(c, "timing_basis", "timingBasis")
        refs = _attr(c, "evidence_refs", "evidenceRefs", default=[]) or []
        seed = EpisodeSeed(
            t_ms=int(time_range[0]),
            end_ms=max(int(time_range[0]), int(time_range[1])),
            origin="correlated_signal",
            label=str(factor_id),
            evidence_refs=list(refs),
            section_id=_attr(c, "section_id", "sectionId"),
            question_number=_attr(c, "question_number", "questionNumber"),
        )
        has_time = timing_basis != "cohort_only" and (seed.t_ms > 0 or seed.end_ms > 0)
        (timed if has_time else session).append(seed)
    return timed, session


def _collect_video_seeds(video_findings: list[Any]) -> list[EpisodeSeed]:
    seeds: list[EpisodeSeed] = []
    for finding in video_findings:
        verdict = _attr(finding, "verdict")
        if verdict not in {"flagged", "provisional"}:
            continue
        window = _attr(finding, "timestamp_window_ms", "timestampWindowMs")
        event_type = _attr(finding, "event_type", "eventType")
        evidence_ref = _attr(finding, "evidence_ref", "evidenceRef")
        fid = _attr(finding, "id", default="video")
        seeds.append(
            EpisodeSeed(
                t_ms=int(window[0]),
                end_ms=max(int(window[0]), int(window[1])),
                origin="video_episode",
                label=str(event_type),
                evidence_refs=[str(evidence_ref or f"video:{fid}")],
                hint_window_ids=window_ids_from_evidence_ref(str(evidence_ref or "")),
            )
        )
    return seeds


def _collect_machine_fact_seeds(facts: list[Any]) -> list[EpisodeSeed]:
    seeds: list[EpisodeSeed] = []
    for fact in facts:
        kind = _fact_kind(fact)
        if kind not in EPISODE_SEED_FACT_KINDS:
            continue
        detail = _attr(fact, "detail", default={}) or {}
        if not isinstance(detail, Mapping):
            detail = {}
        seeds.append(
            EpisodeSeed(
                t_ms=_fact_start(fact),
                end_ms=_fact_end(fact),
                origin="machine_fact",
                label=kind,
                evidence_refs=[f"fact:{_attr(fact, 'id', default=kind)}"],
                section_id=detail.get("sectionId"),
                question_number=detail.get("questionNumber"),
            )
        )
    return seeds


def _seeds_compatible(a: EpisodeSeed, b: EpisodeSeed) -> bool:
    if a.origin == b.origin:
        return True
    pair = {a.origin, b.origin}
    if "correlated_signal" in pair:
        return True
    return pair == {"video_episode", "machine_fact"}


@dataclass(slots=True)
class _SeedCluster:
    t_ms: int
    end_ms: int
    seeds: list[EpisodeSeed]


def _cluster_seeds(seeds: list[EpisodeSeed], merge_gap_ms: int) -> list[_SeedCluster]:
    sorted_seeds = sorted(seeds, key=lambda s: (s.t_ms, s.end_ms))
    clusters: list[_SeedCluster] = []
    for seed in sorted_seeds:
        current = clusters[-1] if clusters else None
        if (
            current
            and seed.t_ms <= current.end_ms + merge_gap_ms
            and any(_seeds_compatible(s, seed) for s in current.seeds)
            and max(current.end_ms, seed.end_ms) - current.t_ms <= EPISODE_MAX_SPAN_MS
        ):
            current.end_ms = max(current.end_ms, seed.end_ms)
            current.seeds.append(seed)
        else:
            clusters.append(_SeedCluster(seed.t_ms, seed.end_ms, [seed]))
    return clusters


def _eligible_windows(perception_bundle: Any | None) -> list[tuple[str, int, int]]:
    if perception_bundle is None:
        return []
    windows = getattr(perception_bundle, "windows", []) or []
    observations = getattr(perception_bundle, "observations", []) or []
    with_video = {
        _attr(o, "window_id", "windowId")
        for o in observations
        if _attr(o, "video_available", "videoAvailable")
    }
    out: list[tuple[str, int, int]] = []
    for w in windows:
        phase = _attr(w, "phase")
        wid = _attr(w, "window_id", "windowId")
        if phase != "live_exam" or wid not in with_video:
            continue
        start = int(_attr(w, "start_ms", "startMs", default=0) or 0)
        end = int(_attr(w, "end_ms", "endMs", default=start) or start)
        out.append((str(wid), start, end))
    return out


def build_episode_inventory(
    *,
    correlated_signals: Any | None = None,
    video_findings: list[Any] | None = None,
    machine_facts_bundle: Any,
    perception_bundle: Any | None = None,
    provisional_unknowns: list[dict[str, Any]] | None = None,
    merge_gap_ms: int = EPISODE_MERGE_GAP_MS,
    max_timed_episodes: int = EPISODE_INVENTORY_MAX_TIMED,
) -> list[DeliberationEpisode]:
    correlated_timed, correlated_session = _collect_correlated_seeds(correlated_signals)
    facts = attr(machine_facts_bundle, "facts", default=[]) or []
    seeds = [
        *correlated_timed,
        *_collect_video_seeds(video_findings or []),
        *_collect_provisional_unknown_seeds(provisional_unknowns),
        *_collect_machine_fact_seeds(list(facts)),
    ]
    windows = _eligible_windows(perception_bundle)
    known_window_ids = set(w[0] for w in windows)
    for obs in attr(perception_bundle, "observations", default=[]) or []:
        if attr(obs, "video_available", "videoAvailable"):
            wid = attr(obs, "window_id", "windowId")
            if wid:
                known_window_ids.add(str(wid))
    clusters = _cluster_seeds(seeds, merge_gap_ms)

    timed_episodes: list[DeliberationEpisode] = []
    for cluster in clusters:
        origins = _dedupe([s.origin for s in cluster.seeds])
        factor_ids = _dedupe(
            [s.label for s in cluster.seeds if s.origin == "correlated_signal"]
        )
        video_event_types = _dedupe(
            [s.label for s in cluster.seeds if s.origin == "video_episode"]
        )
        machine_fact_kinds = _dedupe(
            [
                _fact_kind(f)
                for f in facts
                if _fact_start(f) >= cluster.t_ms and _fact_start(f) <= cluster.end_ms
            ]
        )
        window_ids = [
            wid
            for wid, start, end in windows
            if start <= cluster.end_ms and cluster.t_ms <= end
        ]
        if not window_ids:
            window_ids = _dedupe(
                hint
                for s in cluster.seeds
                for hint in s.hint_window_ids
                if hint in known_window_ids
            )
        timed_episodes.append(
            DeliberationEpisode(
                episode_id="",
                scope="timed",
                time_range_ms=(cluster.t_ms, cluster.end_ms),
                origins=origins,  # type: ignore[arg-type]
                factor_ids=factor_ids,
                video_event_types=video_event_types,
                machine_fact_kinds=machine_fact_kinds,
                window_ids=window_ids,
                evidence_refs=_dedupe(
                    ref for s in cluster.seeds for ref in s.evidence_refs
                )[:12],
                modality_count=len(origins),
                section_id=next((s.section_id for s in cluster.seeds if s.section_id), None),
                question_number=next(
                    (s.question_number for s in cluster.seeds if s.question_number is not None),
                    None,
                ),
            )
        )

    if len(timed_episodes) > max_timed_episodes:
        timed_episodes = sorted(
            timed_episodes,
            key=lambda e: (e.modality_count, e.time_range_ms[1] - e.time_range_ms[0]),
            reverse=True,
        )[:max_timed_episodes]
        timed_episodes.sort(key=lambda e: e.time_range_ms[0])

    session_episodes = [
        DeliberationEpisode(
            episode_id="",
            scope="session",
            time_range_ms=(0, 0),
            origins=["correlated_signal"],
            factor_ids=[seed.label],
            video_event_types=[],
            machine_fact_kinds=[],
            window_ids=[],
            evidence_refs=_dedupe(seed.evidence_refs),
            modality_count=1,
            section_id=seed.section_id,
            question_number=seed.question_number,
        )
        for seed in correlated_session
    ]

    all_eps = timed_episodes + session_episodes
    chains = attr(correlated_signals, "chains", default=[]) or []
    video_list = video_findings or []

    enriched: list[DeliberationEpisode] = []
    for index, ep in enumerate(all_eps):
        provisional_lane = ep.provisional_lane
        for chain in chains:
            if ep.scope != "timed":
                continue
            chain_range = attr(chain, "time_range_ms", "timeRangeMs", default=(0, 0))
            chain_factor = str(attr(chain, "factor_id", "factorId", default=""))
            if not (
                chain_range[0] <= ep.time_range_ms[1]
                and ep.time_range_ms[0] <= chain_range[1]
                and (not ep.factor_ids or chain_factor in ep.factor_ids)
            ):
                continue
            lane = attr(chain, "provisional_lane", "provisionalLane")
            if lane == "red":
                provisional_lane = "red"
            elif lane == "amber" and provisional_lane != "red":
                provisional_lane = "amber"
            elif lane and not provisional_lane:
                provisional_lane = lane

        if not provisional_lane and ep.scope == "timed":
            seed_amber = any(
                seed.provisional_lane == "amber"
                and seed.t_ms <= ep.time_range_ms[1]
                and ep.time_range_ms[0] <= seed.end_ms
                and label in ep.video_event_types
                for seed in seeds
                for label in [seed.label]
            )
            if seed_amber:
                provisional_lane = "amber"

        overlapping = [
            finding
            for finding in video_list
            if ep.scope == "timed"
            and _finding_window(finding)[0] <= ep.time_range_ms[1]
            and ep.time_range_ms[0] <= _finding_window(finding)[1]
        ]
        hints = _dedupe(
            [
                f"{attr(f, 'event_type', 'eventType')}:{attr(f, 'evidence_strength', 'evidenceStrength')}"
                for f in overlapping
                if attr(f, "evidence_strength", "evidenceStrength")
            ]
        )
        gaze = any(
            attr(f, "candidate_gaze_diverted_toward_person", "candidateGazeDivertedTowardPerson")
            is True
            for f in overlapping
        )

        enriched.append(
            DeliberationEpisode(
                episode_id=f"ep{index + 1}",
                scope=ep.scope,
                time_range_ms=ep.time_range_ms,
                origins=ep.origins,
                factor_ids=ep.factor_ids,
                video_event_types=ep.video_event_types,
                machine_fact_kinds=ep.machine_fact_kinds,
                window_ids=ep.window_ids,
                evidence_refs=ep.evidence_refs,
                modality_count=ep.modality_count,
                section_id=ep.section_id,
                question_number=ep.question_number,
                provisional_lane=provisional_lane,
                evidence_strength_hints=hints or None,
                gaze_diverted_toward_person=True if gaze else None,
            )
        )
    return enriched


def _finding_window(finding: Any) -> tuple[int, int]:
    window = attr(finding, "timestamp_window_ms", "timestampWindowMs", default=(0, 0))
    return int(window[0]), int(window[1])


def build_episode_analysis_entries(
    episodes: list[DeliberationEpisode],
    adjudicated_by_id: Mapping[str, RawEpisodeAnalysis],
    emitted_signal_ids_by_episode: Mapping[str, list[str]],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for episode in episodes:
        adjudicated = adjudicated_by_id.get(episode.episode_id)
        emitted_ids = list(emitted_signal_ids_by_episode.get(episode.episode_id, []))
        will_emit = bool(emitted_ids)
        entries.append(
            {
                "episodeId": episode.episode_id,
                "scope": episode.scope,
                "timeRangeMs": list(episode.time_range_ms),
                "origins": episode.origins,
                "factorIds": episode.factor_ids,
                "windowIds": episode.window_ids,
                "machineFactKinds": episode.machine_fact_kinds,
                "episodeSummary": adjudicated.episode_summary if adjudicated else "",
                "suspiciousBehaviorType": (
                    adjudicated.suspicious_behavior_type if adjudicated else "none"
                ),
                "willEmitSignal": will_emit,
                "reasonNotSignalled": (
                    None
                    if will_emit
                    else (
                        adjudicated.reason_not_signalled
                        if adjudicated and not adjudicated.will_emit_signal
                        else "candidate signal did not pass deterministic validation"
                        if adjudicated
                        else "not adjudicated by model"
                    )
                ),
                "emittedSignalIds": emitted_ids,
            }
        )
    return entries


def serialize_episode_inventory(episodes: list[DeliberationEpisode]) -> str:
    lines = [
        "=== EPISODE INVENTORY ===",
        "These episodes are derived deterministically from Scope 1 facts, Scope 3.5 correlations, and video episodes.",
        "Produce exactly ONE episodeAnalysis entry per episodeId below. Do NOT invent episodeIds.",
        "Every candidateSignal.episodeRef MUST be one of these episodeIds with willEmitSignal=true.",
        "",
    ]
    if not episodes:
        lines.append("(no episodes derived — emit an empty candidateSignals array)")
        return "\n".join(lines)

    timed = [e for e in episodes if e.scope == "timed"]
    session = [e for e in episodes if e.scope == "session"]
    if timed:
        lines.append(
            "TIMED EPISODES (episodeId | timeRange | origins | factorIds | videoEventTypes | "
            "machineFactKinds | windowIds | provisionalLane | evidenceStrength | gazeDiverted):"
        )
        for episode in timed:
            lines.append(
                f"  {episode.episode_id} | {_fmt_range(episode.time_range_ms)} | "
                f"{','.join(episode.origins)} | {','.join(episode.factor_ids) or '-'} | "
                f"{','.join(episode.video_event_types) or '-'} | "
                f"{','.join(episode.machine_fact_kinds) or '-'} | "
                f"{','.join(episode.window_ids) or '-'} | "
                f"{episode.provisional_lane or '-'} | "
                f"{','.join(episode.evidence_strength_hints or []) or '-'} | "
                f"{'true' if episode.gaze_diverted_toward_person else '-'}"
            )
        lines.append(
            "  NOTE: provisionalLane / evidenceStrength / gazeDiverted are citable priors — "
            "address them; UI color uses your resolution only."
        )
        lines.append("")
    if session:
        lines.append("SESSION-SCOPED EPISODES (cohort comparisons — no time range; never timed evidence):")
        for episode in session:
            lines.append(f"  {episode.episode_id} | factorIds={','.join(episode.factor_ids)}")
        lines.append("")
    lines.append("Prefer the windowIds and machineFactKinds listed on the episode you are adjudicating.")
    return "\n".join(lines)
