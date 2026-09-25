from __future__ import annotations

from datetime import UTC, datetime

from ..baseline.contracts import StatisticalBaseline
from ..findings.contracts import EvidenceFinding
from ..machine_facts.contracts import MachineFactsBundle
from .chains import build_candidate_chains
from .contracts import (
    CORRELATED_SCORE_CAP,
    DEFAULT_CORRELATION_CONFIG,
    CorrelatedSignalsBundle,
    CorrelationConfig,
    DetectCtx,
)
from .detectors import detect_all_factors
from .sqb import annotate_facts_with_question_boundaries, derive_boundaries_from_facts
from .timeline import build_correlation_timeline


def derive_correlated_signals(
    *,
    machine_facts: MachineFactsBundle,
    baseline: StatisticalBaseline,
    video_findings: list[EvidenceFinding] | None = None,
    screen_findings: list[EvidenceFinding] | None = None,
    config: CorrelationConfig | None = None,
) -> CorrelatedSignalsBundle:
    cfg = config or DEFAULT_CORRELATION_CONFIG
    notes: list[str] = []
    boundaries = derive_boundaries_from_facts(
        machine_facts.facts,
        candidate_id=machine_facts.candidate_id,
        assessment_id=machine_facts.assessment_id,
        duration_ms=machine_facts.duration_ms,
    )
    if not boundaries:
        notes.append("no SQB boundaries")
    else:
        verified = [
            b
            for b in boundaries
            if not b.timing_unavailable and not b.end_is_fallback and b.end_offset_ms > b.start_offset_ms
        ]
        if not verified:
            notes.append("SQB present but unverified (timing unavailable)")

    annotate_facts_with_question_boundaries(machine_facts.facts, boundaries)

    flagged_video = [
        f for f in (video_findings or []) if f.verdict in {"flagged", "provisional"}
    ]
    flagged_screen = [f for f in (screen_findings or []) if f.verdict == "flagged"]
    timeline = build_correlation_timeline(
        machine_facts.facts,
        boundaries,
        flagged_video,
        flagged_screen,
    )

    ctx = DetectCtx(
        timeline=timeline,
        boundaries=boundaries,
        facts=machine_facts.facts,
        baseline=baseline,
        config=cfg,
        notes=notes,
    )
    contributions = detect_all_factors(ctx)
    if not baseline.cohort_candidate_stats and not baseline.cohort_capabilities:
        notes.append("cohort absent")

    factor_counts: dict[str, int] = {}
    raw_sum = 0.0
    for contribution in contributions:
        factor_counts[contribution.factor_id] = factor_counts.get(contribution.factor_id, 0) + 1
        raw_sum += contribution.score_contribution

    chains = build_candidate_chains(contributions, timeline)
    return CorrelatedSignalsBundle(
        produced_at=datetime.now(tz=UTC).isoformat(),
        candidate_id=machine_facts.candidate_id,
        assessment_id=machine_facts.assessment_id,
        config=cfg,
        timeline_event_count=len(timeline),
        contributions=contributions,
        chains=chains,
        total_weighted_score=min(CORRELATED_SCORE_CAP, round(raw_sum, 2)),
        factor_counts=factor_counts,
        notes=list(dict.fromkeys(notes)),
    )
