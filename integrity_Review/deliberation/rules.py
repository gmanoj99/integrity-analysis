"""Scope 4 deterministic deliberation rules."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

from ..contracts.deliberation import (
    EvidenceSourceType,
    IntegrityStory,
    IntegrityStoryProofAnchors,
    IntegrityStorySeverity,
    RecommendationCategory,
    ValidatedSignal,
)
from ..duck_helpers import attr, mapping_view
from ..prompts.deliberation import DELIBERATION_PROMPT_VERSION

# Names the model deliberation actually runs on (prompts.shared.DELIBERATION_MODEL).
# It also feeds the artifact hash, so a model swap must be reflected here or
# two different runs share a provenance fingerprint.
DELIBERATION_MODEL_VERSION = "gemini-3.8-flash"

_DELIBERATION_ARTIFACT_HASH = hashlib.sha256(
    f"{DELIBERATION_MODEL_VERSION}|{DELIBERATION_PROMPT_VERSION}".encode()
).hexdigest()[:16]

SIGNAL_TYPES = frozenset(
    {
        "possible_external_consultation",
        "possible_second_person_involvement",
        "unauthorized_reference_usage",
        "abnormal_paste_workflow",
        "suspicious_focus_pattern",
        "abnormal_correction_pattern",
        "typing_cadence_mismatch",
        "inconsistent_interaction_sequence",
        "possible_audio_coaching",
        "possible_remote_dictation",
    }
)

INTEGRITY_SPEECH_CONTENT_CLASSES = frozenset(
    {
        "asking_for_answer",
        "receiving_dictation",
        "discussing_solution",
        "reciting_answer_choices",
    }
)

DEFINITIVE_SOLO_OBSERVATION_FIELDS: dict[str, frozenset[str]] = {
    "hands.objectInHand": frozenset({"phone"}),
    "audio.speechContentClass": INTEGRITY_SPEECH_CONTENT_CLASSES,
}

DEFINITIVE_SOLO_MACHINE_FACT_KINDS = frozenset(
    {
        "SCREEN_EXTERNAL_RESOURCE",
        "SCREEN_AI_ASSISTANT_UI",
        "SCREEN_EXTERNAL_PASTE",
    }
)

INFORMATIVE_FIELD_PATHS = (
    "objects.phoneVisible",
    "people.secondPersonVisible",
    "hands.handsVisible",
    "attention.gazeDirection",
    "identity.facePresent",
    "interaction.candidateSpeaking",
)

FIELD_PATH_RESOLVERS: dict[str, str] = {
    "people.secondPersonVisible": "people.secondPersonVisible",
    "people.secondPersonInteracting": "people.secondPersonInteracting",
    "people.candidateRespondingToSecondPerson": "people.candidateRespondingToSecondPerson",
    "people.secondPersonActivity": "people.secondPersonActivity",
    "people.peopleInFrame": "people.peopleInFrame",
    "objects.phoneVisible": "objects.phoneVisible",
    "objects.headphonesVisible": "objects.headphonesVisible",
    "objects.headphonesLink": "objects.headphonesLink",
    "objects.earphoneVisible": "objects.earphoneVisible",
    "objects.earphoneLink": "objects.earphoneLink",
    "objects.notebookVisible": "objects.notebookVisible",
    "objects.paperVisible": "objects.paperVisible",
    "hands.handsVisible": "hands.handsVisible",
    "hands.objectInHand": "hands.objectInHand",
    "hands.handActivity": "hands.handActivity",
    "attention.gazeDirection": "attention.gazeDirection",
    "attention.repeatedGazePattern": "attention.repeatedGazePattern",
    "attention.attentionState": "attention.attentionState",
    "identity.facePresent": "identity.facePresent",
    "identity.faceCount": "identity.faceCount",
    "identity.identityConsistent": "identity.identityConsistent",
    "interaction.candidateSpeaking": "interaction.candidateSpeaking",
    "interaction.reachingOutsideFrame": "interaction.reachingOutsideFrame",
    "body.posture": "body.posture",
    "environment.settingType": "environment.settingType",
    "environment.backgroundActivity": "environment.backgroundActivity",
    "audio.speechPresent": "audio.speechPresent",
    "audio.speechSource": "audio.speechSource",
    "audio.speechLanguage": "audio.speechLanguage",
    "audio.speechContentClass": "audio.speechContentClass",
    "audio.speechStyle": "audio.speechStyle",
    "audio.speechOverlapWithLips": "audio.speechOverlapWithLips",
    "audio.codeMixing": "audio.codeMixing",
    # Advertised verbatim by engine._build_citation_inventory, so it has to
    # resolve here too. Free text, so the value comparison below will often
    # not match exactly; that only drops the citation, never the signal.
    "audio.conversationSummaryEn": "audio.conversationSummaryEn",
    "people.secondPersonPosition": "people.secondPersonPosition",
    "people.secondPersonLooksLike": "people.secondPersonLooksLike",
    "people.secondPersonRoleCue": "people.secondPersonRoleCue",
}

KNOWN_BASELINE_METRICS = frozenset(
    {
        "typingCadenceRegularity",
        "typingBurstCount",
        "typingBurstDurations",
        "ikiDistribution",
        "typingSpeedWpm",
        "idleGapDurations",
        "textCorrectionCount",
        "correctionsPerBurstSeries",
        "correctionsPerBurst",
        "pastePlatformCount",
        "pastePlatformFrequencyPerMin",
        "largePasteCount",
        "largePasteSizes",
        "largePasteIntervalMs",
        "largePasteSizeDistribution",
        "largePasteFrequencyPerMin",
        "windowBlurCount",
        "windowBlurTotalDurationMs",
        "windowBlurFrequencyPerMin",
        "sectionLatencyMs",
        "gazeOffScreenWindowCount",
        "handsNotVisibleWindowCount",
        "faceAbsenceWindowCount",
        "phoneVisibleWindowCount",
        "notebookVisibleWindowCount",
        "secondPersonWindowCount",
        "candidateSpeakingWindowCount",
        "identityConsistentRate",
        "postureChangeCount",
        "usableWindowRatio",
        "missingVideoRatio",
        "unknownGazeRatio",
        "meanObservationConfidence",
    }
)

FORBIDDEN_STORY_TEMPLATE_PATTERNS = (
    re.compile(r"suspicious_eye_movement\s+co-occurs", re.I),
    re.compile(r"within\s+\d+\s*ms\)", re.I),
    re.compile(r"median gap \d+ms", re.I),
    re.compile(r"no external paste — answered fast after looking off-screen", re.I),
    re.compile(r"overlapping rapid MCQ burst", re.I),
    re.compile(r"factorId[=:]", re.I),
)


@dataclass(slots=True)
class RawEpisodeAnalysis:
    episode_id: str
    time_range: str
    window_ids: list[str]
    machine_facts_in_range: list[str]
    episode_summary: str
    suspicious_behavior_type: str
    will_emit_signal: bool
    reason_not_signalled: str | None = None


@dataclass(slots=True)
class RawCandidateSignal:
    signal_type: str
    episode_ref: str | None
    hypothesis_honest: dict[str, list[str]]
    hypothesis_assisted: dict[str, list[str]]
    resolution: str
    confidence: float
    innocent_explanation_considered: bool
    why_rejected: str
    machine_facts_cited: list[str]
    observations_cited: list[str]
    baseline_metrics_cited: list[str]
    integrity_story: dict[str, Any] | None = None


@dataclass(slots=True)
class RuleResult:
    ok: bool
    reason: str | None = None
    kept_observations_cited: list[str] | None = None
    # Why each citation was discarded, in "<citation> — <reason>" form. Dropping
    # a citation silently and returning ok=True made the loss surface later as
    # CitationRule's "Zero citations", which reads as "the model cited nothing"
    # when the truth is often "everything it cited failed to resolve". Reviewers
    # cannot tell a hallucinated citation from a value mismatch without this.
    dropped_citations: list[str] | None = None
    # The inventory episode this signal resolved to, which is not always the id
    # the model wrote — see resolve_episode_ref_from_windows.
    resolved_episode_ref: str | None = None


def compute_deliberation_version_hash(
    perception_version_hash: str,
    baseline_version_hash: str,
    correlated_signals_version_hash: str = "none",
) -> str:
    payload = (
        f"{DELIBERATION_PROMPT_VERSION}|{perception_version_hash}|"
        f"{baseline_version_hash}|{correlated_signals_version_hash}|{_DELIBERATION_ARTIFACT_HASH}"
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def parse_observation_citation(ref: str) -> tuple[str, str, str | None] | None:
    trimmed = ref.strip()
    if not trimmed:
        return None
    eq = trimmed.find("=")
    colon = trimmed.find(":")
    if eq == -1:
        sep = colon
    elif colon == -1:
        sep = eq
    else:
        sep = min(eq, colon)
    key_part = trimmed if sep == -1 else trimmed[:sep]
    claimed = None if sep == -1 else trimmed[sep + 1 :].strip() or None
    dot = key_part.find(".")
    if dot <= 0:
        return None
    return key_part[:dot], key_part[dot + 1 :], claimed


def _is_unknown(value: object) -> bool:
    if value is None:
        return True
    return str(value).strip().upper() == "UNKNOWN"


def _get_nested(obs: Any, path: str) -> object:
    parts = path.split(".")
    cur: Any = mapping_view(obs)
    for part in parts:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def resolve_episode_ref_from_windows(
    signal: RawCandidateSignal,
    episodes_by_id: dict[str, RawEpisodeAnalysis],
) -> str | None:
    """Recover the episode a signal means from the windows it cites.

    Models that did not write the inventory's exact id — inventing ``model_ep1``
    is the common shape — still cite real windows, and a window belongs to at
    most one adjudicated episode. When exactly one episode owns every cited
    window the reference is unambiguous, so repairing it keeps a well-evidenced
    judgement that an id mismatch would otherwise discard whole. Anything
    ambiguous stays rejected: guessing between episodes would attach a signal to
    the wrong moment, which is worse than losing it.
    """

    cited_windows = {
        parsed[0]
        for parsed in (parse_observation_citation(ref) for ref in signal.observations_cited)
        if parsed is not None
    }
    if not cited_windows:
        return None
    owners = [
        episode_id
        for episode_id, episode in episodes_by_id.items()
        if episode.will_emit_signal and cited_windows <= set(episode.window_ids)
    ]
    return owners[0] if len(owners) == 1 else None


def validate_episode_ref(
    signal: RawCandidateSignal,
    episodes_by_id: dict[str, RawEpisodeAnalysis],
) -> RuleResult:
    ref = signal.episode_ref
    if not ref or not ref.strip():
        return RuleResult(
            False,
            "Signal has no episodeRef; every signal must reference an inventory episode",
        )
    episode = episodes_by_id.get(ref.strip())
    if episode is None:
        repaired = resolve_episode_ref_from_windows(signal, episodes_by_id)
        if repaired is not None:
            return RuleResult(
                True,
                reason=f'episodeRef "{ref}" repaired to "{repaired}" from its cited windows',
                resolved_episode_ref=repaired,
            )
        return RuleResult(
            False,
            f'episodeRef "{ref}" is not an episodeId from the deterministic episode inventory',
        )
    if not episode.will_emit_signal:
        return RuleResult(
            False,
            f'episodeRef "{ref}" was adjudicated with willEmitSignal=false',
        )
    return RuleResult(True, resolved_episode_ref=ref.strip())


def validate_value_resolving_citations(
    signal: RawCandidateSignal,
    observations_by_window_id: dict[str, Any],
) -> RuleResult:
    kept: list[str] = []
    dropped: list[str] = []
    for ref in signal.observations_cited:
        parsed = parse_observation_citation(ref)
        if parsed is None:
            return RuleResult(
                False,
                f'Citation "{ref}" is not in windowId.fieldPath=value form',
            )
        window_id, field_path, claimed = parsed
        observation = observations_by_window_id.get(window_id)
        if observation is None:
            return RuleResult(False, f'Cited window "{window_id}" does not exist')
        if field_path not in FIELD_PATH_RESOLVERS:
            # Drop the citation rather than the signal: one stray field path
            # must not discard an otherwise well-evidenced judgement.
            dropped.append(f"{ref} — field path is not citable")
            continue
        actual = _get_nested(observation, FIELD_PATH_RESOLVERS[field_path])
        if _is_unknown(actual):
            dropped.append(f"{ref} — observed value is UNKNOWN")
            continue
        if claimed is not None and str(actual).lower() != claimed.lower():
            dropped.append(f"{ref} — cited value does not match observed {actual!r}")
            continue
        kept.append(ref)
    return RuleResult(True, kept_observations_cited=kept, dropped_citations=dropped)


def validate_smart_student_guard(signal: RawCandidateSignal) -> bool:
    speed_only_mf = {"TYPING_STOPPED", "TYPING_STARTED"}
    speed_only_metrics = {"typingSpeedWpm", "typingBurstCount", "typingBurstDurations"}
    mf = signal.machine_facts_cited
    obs = signal.observations_cited
    base = signal.baseline_metrics_cited
    only_speed_facts = bool(mf) and all(k in speed_only_mf for k in mf)
    no_obs = not obs
    only_speed_metrics = not base or all(m in speed_only_metrics for m in base)
    return not (only_speed_facts and no_obs and only_speed_metrics)


def validate_unknown_rule(signal: RawCandidateSignal) -> bool:
    obs = signal.observations_cited
    if not obs:
        return True
    patterns = (":UNKNOWN", "=UNKNOWN", ".UNKNOWN")
    if all(any(p in ref.upper() for p in patterns) for ref in obs):
        return False
    assisted = signal.hypothesis_assisted.get("supporting", [])
    if assisted and all("unknown" in s.lower() for s in assisted):
        return False
    return True


def validate_citation_rule(signal: RawCandidateSignal) -> bool:
    return (
        len(signal.machine_facts_cited)
        + len(signal.observations_cited)
        + len(signal.baseline_metrics_cited)
        > 0
    )


def _cited_observation_value(obs_cited: list[str], field_path: str) -> str | None:
    for ref in obs_cited:
        parsed = parse_observation_citation(ref)
        if parsed and parsed[1] == field_path and parsed[2] is not None:
            return parsed[2].lower()
    return None


def has_definitive_solo_evidence(
    signal: RawCandidateSignal,
    *,
    episode_factor_ids: list[str] | None = None,
) -> bool:
    for ref in signal.observations_cited:
        parsed = parse_observation_citation(ref)
        if not parsed or parsed[2] is None:
            continue
        allowed = DEFINITIVE_SOLO_OBSERVATION_FIELDS.get(parsed[1])
        if allowed and parsed[2].lower() in allowed:
            return True
    if any(k in DEFINITIVE_SOLO_MACHINE_FACT_KINDS for k in signal.machine_facts_cited):
        return True

    obs_cited = signal.observations_cited
    speech = _cited_observation_value(obs_cited, "audio.speechContentClass")
    interaction = _cited_observation_value(obs_cited, "people.secondPersonInteracting")
    activity = _cited_observation_value(obs_cited, "people.secondPersonActivity")
    interaction_confirmed = interaction == "yes" or activity == "speaking_to_candidate"
    if interaction_confirmed and speech in INTEGRITY_SPEECH_CONTENT_CLASSES:
        return True

    gaze = _cited_observation_value(obs_cited, "attention.gazeDirection")
    diverted_gaze = gaze is not None and gaze in {"left", "right", "up", "away"}
    if interaction_confirmed and diverted_gaze:
        return True

    visible = _cited_observation_value(obs_cited, "people.secondPersonVisible") == "yes"
    if visible and diverted_gaze and speech in INTEGRITY_SPEECH_CONTENT_CLASSES:
        return True

    if signal.signal_type == "abnormal_paste_workflow" and episode_factor_ids:
        if "deterministic_paste_workflow" in episode_factor_ids and (
            any(k in {"LARGE_PASTE", "SCREEN_EXTERNAL_PASTE", "TEXT_CORRECTION"} for k in signal.machine_facts_cited)
            or any(m in {"largePasteCount", "largePasteSizes"} for m in signal.baseline_metrics_cited)
        ):
            return True
    return False


def validate_conjunction_rule(
    signal: RawCandidateSignal,
    *,
    episode_factor_ids: list[str] | None = None,
) -> bool:
    mf = signal.machine_facts_cited
    obs = signal.observations_cited
    base = signal.baseline_metrics_cited
    total = len(mf) + len(obs) + len(base)
    source_types = (1 if mf else 0) + (1 if obs else 0) + (1 if base else 0)
    if has_definitive_solo_evidence(signal, episode_factor_ids=episode_factor_ids):
        return total >= 1
    if total < 2 or source_types < 2:
        return False
    return True


def validate_no_fact_invention(
    signal: RawCandidateSignal,
    machine_fact_kinds_present: set[str],
    window_ids_present: set[str],
    baseline_metric_names_present: set[str],
) -> bool:
    for kind in signal.machine_facts_cited:
        if kind not in machine_fact_kinds_present:
            return False
    for ref in signal.observations_cited:
        window_id = ref.split(".")[0]
        if window_id and window_id not in window_ids_present:
            return False
    for metric in signal.baseline_metrics_cited:
        if metric not in baseline_metric_names_present:
            return False
    return True


def active_validated_signals(signals: list[ValidatedSignal]) -> list[ValidatedSignal]:
    return [s for s in signals if s.resolution != "honest"]


def derive_category(active_signals: list[ValidatedSignal]) -> RecommendationCategory:
    if not active_signals:
        return "CLEAR"
    has_assisted = any(s.resolution == "assisted" for s in active_signals)
    source_types: set[EvidenceSourceType] = set()
    for s in active_signals:
        source_types.update(s.source_types)
    if has_assisted and len(source_types) >= 2:
        return "STRONG_EVIDENCE"
    return "REVIEW_REQUIRED"


def derive_confidence(active_signals: list[ValidatedSignal]) -> float:
    if not active_signals:
        return 0.0
    return max(s.confidence for s in active_signals)


def apply_capture_cap(confidence: float, usable_window_ratio: float) -> tuple[float, bool]:
    if usable_window_ratio < 0.30:
        return min(confidence, 0.50), True
    if usable_window_ratio < 0.60:
        return min(confidence, 0.70), True
    return confidence, False


def compute_informative_content_ratio(
    perception_bundle: Any,
) -> tuple[float, int, int]:
    windows = attr(perception_bundle, "windows", default=[]) or []
    observations = attr(perception_bundle, "observations", default=[]) or []
    live_ids = {
        str(attr(w, "window_id", "windowId"))
        for w in windows
        if attr(w, "phase") == "live_exam" and attr(w, "window_id", "windowId")
    }
    considered = [
        o
        for o in observations
        if attr(o, "video_available", "videoAvailable")
        and str(attr(o, "window_id", "windowId")) in live_ids
    ]
    if not considered:
        return 1.0, 0, 0

    def informative(obs: Any) -> bool:
        count = 0
        for path in INFORMATIVE_FIELD_PATHS:
            if not _is_unknown(_get_nested(obs, path)):
                count += 1
            if count >= 2:
                return True
        return False

    informative_count = sum(1 for o in considered if informative(o))
    return informative_count / len(considered), informative_count, len(considered)


def apply_confidence_caps(
    confidence: float,
    usable_window_ratio: float,
    informative_content_ratio: float,
    *,
    apply_informative_cap: bool = True,
) -> tuple[float, bool, bool]:
    capped, capture_applied = apply_capture_cap(confidence, usable_window_ratio)
    informative_cap = 1.0
    informative_applied = False
    if apply_informative_cap:
        if informative_content_ratio < 0.25:
            informative_cap = 0.50
            informative_applied = True
        elif informative_content_ratio < 0.50:
            informative_cap = 0.70
            informative_applied = True
    return min(capped, informative_cap), capture_applied, informative_applied


def _story_has_template_echo(text: str) -> bool:
    return any(pattern.search(text) for pattern in FORBIDDEN_STORY_TEMPLATE_PATTERNS)


def _as_int_list(value: Any) -> list[int]:
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [int(value)]
    if not isinstance(value, list):
        return []
    out: list[int] = []
    for item in value:
        if isinstance(item, bool):
            continue
        if isinstance(item, (int, float)):
            out.append(int(item))
        elif isinstance(item, str) and item.strip().lstrip("-").isdigit():
            out.append(int(item.strip()))
    return out


def _as_str_list(value: Any) -> list[str]:
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if item is not None and str(item).strip()]


def _normalize_proof_anchors(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        window_ids: list[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                window_ids.append(item.strip())
            elif isinstance(item, dict):
                window_id = item.get("windowId") or item.get("window_id")
                if window_id:
                    window_ids.append(str(window_id))
        return {"windowIds": window_ids}
    return {}


def parse_integrity_story(raw: dict[str, Any] | None) -> IntegrityStory | None:
    if not isinstance(raw, dict):
        return None
    headline = str(raw.get("headline", "")).strip()
    what = str(raw.get("whatHappened", raw.get("what_happened", ""))).strip()
    why = str(raw.get("whyItMatters", raw.get("why_it_matters", ""))).strip()
    honest = str(raw.get("honestAlternative", raw.get("honest_alternative", ""))).strip()
    if not headline or not what:
        return None
    combined = f"{headline}\n{what}\n{why}"
    if _story_has_template_echo(combined):
        return None
    severity_raw = str(raw.get("severity", "suspicious")).strip().lower()
    severity: IntegrityStorySeverity = (
        severity_raw if severity_raw in {"probable", "suspicious", "weak"} else "suspicious"
    )
    anchors_raw = _normalize_proof_anchors(
        raw.get("proofAnchors") if raw.get("proofAnchors") is not None else raw.get("proof_anchors")
    )
    anchors = IntegrityStoryProofAnchors(
        window_ids=_as_str_list(
            anchors_raw.get("windowIds") or anchors_raw.get("window_ids")
        ),
        machine_fact_kinds=_as_str_list(
            anchors_raw.get("machineFactKinds") or anchors_raw.get("machine_fact_kinds")
        ),
        audio_quotes_en=_as_str_list(
            anchors_raw.get("audioQuotesEn") or anchors_raw.get("audio_quotes_en")
        ),
        seek_ms_hints=[
            int(item)
            for item in (
                anchors_raw.get("seekMsHints") or anchors_raw.get("seek_ms_hints") or []
            )
            if isinstance(item, (int, float, str)) and str(item).strip().lstrip("-").isdigit()
        ],
        factor_ids=_as_str_list(anchors_raw.get("factorIds") or anchors_raw.get("factor_ids")),
    )
    return IntegrityStory(
        headline=headline,
        what_happened=what,
        why_it_matters=why or "Reviewer should inspect the attached proof.",
        honest_alternative=honest or "An innocent explanation may still apply.",
        severity=severity,
        involved_questions=_as_int_list(
            raw.get("involvedQuestions") or raw.get("involved_questions")
        ),
        proof_anchors=anchors,
    )


def filter_audio_quotes_against_perception(
    quotes: list[str],
    known_summaries: list[str],
) -> list[str]:
    if not quotes:
        return []
    norms = [s.lower() for s in known_summaries if s]
    kept: list[str] = []
    for quote in quotes:
        normalized = quote.lower().strip()
        if len(normalized) < 8:
            continue
        if any(
            normalized in norm or norm[: min(80, len(norm))] in normalized
            for norm in norms
        ):
            kept.append(quote)
    return kept


def humanize_signal_type_for_story(signal_type: str) -> str:
    labels = {
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
    return labels.get(signal_type, signal_type.replace("_", " ").strip().capitalize())


def synthesize_integrity_story_fallback(
    *,
    signal_type: str,
    episode_summary: str | None = None,
    assisted_supporting: list[str] | None = None,
    honest_supporting: list[str] | None = None,
    window_ids: list[str] | None = None,
    machine_fact_kinds: list[str] | None = None,
    factor_ids: list[str] | None = None,
) -> IntegrityStory:
    assisted = assisted_supporting or []
    honest = honest_supporting or []
    what = (
        (episode_summary or "").strip()
        or (
            assisted[0]
            if assisted and not _story_has_template_echo(assisted[0])
            else None
        )
        or f"Cross-modal evidence was associated with {signal_type.replace('_', ' ')}."
    )
    honest_alt = (
        honest[0]
        if honest and not _story_has_template_echo(honest[0])
        else "An innocent explanation may still apply after human review."
    )
    return IntegrityStory(
        headline=humanize_signal_type_for_story(signal_type),
        what_happened=what,
        why_it_matters="This pattern warrants reviewer inspection of the attached proof.",
        honest_alternative=honest_alt,
        severity="suspicious",
        involved_questions=[],
        proof_anchors=IntegrityStoryProofAnchors(
            window_ids=window_ids or [],
            machine_fact_kinds=machine_fact_kinds or [],
            factor_ids=factor_ids or [],
        ),
    )


def derive_source_types(signal: RawCandidateSignal) -> list[EvidenceSourceType]:
    types: list[EvidenceSourceType] = []
    if signal.machine_facts_cited:
        types.append("machine_fact")
    if signal.observations_cited:
        has_audio = False
        for ref in signal.observations_cited:
            parsed = parse_observation_citation(ref)
            if parsed and parsed[1].startswith("audio."):
                has_audio = True
                break
        types.append("audio_observation" if has_audio else "visual_observation")
    if signal.baseline_metrics_cited:
        types.append("statistical_baseline")
    return list(dict.fromkeys(types))
