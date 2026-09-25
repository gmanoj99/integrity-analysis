"""Trust score — AI-judged behaviour impact, deterministic aggregation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

TRUST_SCORE_LOGIC_VERSION = "trust-v1"
DEFAULT_TRUST_SCORE = 100
FALLBACK_RESOLUTION_IMPACT: dict[str, float] = {"assisted": 0.80, "ambiguous": 0.45}
COVERAGE_CEILINGS: tuple[tuple[float, int], ...] = ((0.60, 100), (0.30, 85))
COVERAGE_FLOOR_CEILING = 70


@dataclass(frozen=True, slots=True)
class TrustBehaviour:
    label: str
    impact: float
    disposition: str = "unknown"
    refs: tuple[str, ...] = ()
    why: str = ""


def _clamp_impact(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return max(0.0, min(1.0, float(value)))


def _as_str(value: Any, default: str = "") -> str:
    return str(value).strip() if value is not None else default


def parse_trust_behaviours(raw: Any) -> list[TrustBehaviour]:

    if not isinstance(raw, Mapping):
        return []
    assessment = raw.get("trustAssessment") or raw.get("trust_assessment")
    if not isinstance(assessment, Mapping):
        return []
    entries = assessment.get("behaviours") or assessment.get("behaviors") or []
    if not isinstance(entries, list):
        return []
    behaviours: list[TrustBehaviour] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        impact = _clamp_impact(entry.get("impact"))
        if impact is None:
            continue
        refs = entry.get("refs")
        behaviours.append(
            TrustBehaviour(
                label=_as_str(entry.get("label"), "unlabelled behaviour"),
                impact=impact,
                disposition=_as_str(entry.get("disposition"), "unknown"),
                refs=tuple(_as_str(ref) for ref in refs if ref is not None)
                if isinstance(refs, list)
                else (),
                why=_as_str(entry.get("why")),
            )
        )
    return behaviours


def parse_holistic_score(raw: Any) -> int | None:

    if not isinstance(raw, Mapping):
        return None
    assessment = raw.get("trustAssessment") or raw.get("trust_assessment")
    if not isinstance(assessment, Mapping):
        return None
    value = assessment.get("holisticTrustScore") or assessment.get("holistic_trust_score")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return max(0, min(100, int(value)))


def fallback_behaviours(validated_signals: list[Any]) -> list[TrustBehaviour]:
    behaviours: list[TrustBehaviour] = []
    for signal in validated_signals:
        weight = FALLBACK_RESOLUTION_IMPACT.get(str(getattr(signal, "resolution", "")))
        if weight is None:
            continue
        confidence = float(getattr(signal, "confidence", 0.0) or 0.0)
        behaviours.append(
            TrustBehaviour(
                label=str(getattr(signal, "signal_type", "signal")),
                impact=max(0.0, min(1.0, weight * confidence)),
                disposition="fallback",
                refs=(str(getattr(signal, "signal_id", "")),),
                why="derived from resolution and confidence — no model trust assessment",
            )
        )
    return behaviours


def coverage_ceiling(usable_window_ratio: float) -> int:

    ratio = max(0.0, min(1.0, float(usable_window_ratio)))
    for threshold, ceiling in COVERAGE_CEILINGS:
        if ratio >= threshold:
            return ceiling
    return COVERAGE_FLOOR_CEILING


def combine_trust_score(
    behaviours: list[TrustBehaviour],
    *,
    usable_window_ratio: float = 1.0,
) -> int:

    survival = 1.0
    for behaviour in behaviours:
        survival *= 1.0 - max(0.0, min(1.0, behaviour.impact))
    return min(round(100 * survival), coverage_ceiling(usable_window_ratio))


__all__ = [
    "COVERAGE_CEILINGS",
    "COVERAGE_FLOOR_CEILING",
    "DEFAULT_TRUST_SCORE",
    "FALLBACK_RESOLUTION_IMPACT",
    "TRUST_SCORE_LOGIC_VERSION",
    "TrustBehaviour",
    "combine_trust_score",
    "coverage_ceiling",
    "fallback_behaviours",
    "parse_holistic_score",
    "parse_trust_behaviours",
]
