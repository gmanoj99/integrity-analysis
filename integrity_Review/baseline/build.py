from __future__ import annotations

from datetime import UTC, datetime

from ..findings.contracts import VideoObservationWindow
from ..machine_facts.contracts import MachineFactsBundle
from .contracts import BASELINE_VERSION, StatisticalBaseline
from .metrics import compute_coverage_metrics, compute_machine_fact_metrics


def build_statistical_baseline(
    machine_facts: MachineFactsBundle,
    *,
    video_windows: list[VideoObservationWindow] | None = None,
) -> StatisticalBaseline:
    metrics = compute_machine_fact_metrics(machine_facts)
    coverage = compute_coverage_metrics(video_windows) if video_windows else None
    return StatisticalBaseline(
        candidate_id=machine_facts.candidate_id,
        assessment_id=machine_facts.assessment_id,
        produced_at=datetime.now(tz=UTC).isoformat(),
        baseline_version=BASELINE_VERSION,
        machine_fact_metrics=metrics,
        coverage_metrics=coverage,
    )
