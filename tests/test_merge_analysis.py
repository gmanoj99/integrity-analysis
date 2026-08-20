"""Focused tests for merge analysis scoring."""

from types import SimpleNamespace

from integrity_review_pipeline.scoring.merge_analysis import reconcile_findings


def test_track_b_only_high_severity_scores() -> None:
    finding = SimpleNamespace(
        id="tb1",
        event_type="phone_usage",
        eventType="phone_usage",
        timestamp_window_ms=(10_000, 20_000),
        timestampWindowMs=(10_000, 20_000),
        attribution="candidate",
        phase="live_exam",
        severity="high",
        verdict="flagged",
        reasoning="Candidate held phone",
        evidence_strength="moderate",
        evidenceStrength="moderate",
    )
    result = reconcile_findings([], [finding])
    assert result.score.cohort_nudge == 0
    assert result.score.total >= 26
    assert result.score.band in {"Needs Review", "Likely Violation"}
    assert result.merged_findings[0].corroboration_status == "track_b_only"
    assert result.merged_findings[0].merged_verdict == "ai_detected"
    assert result.merged_findings[0].score_weight == 0.85


def test_cleared_findings_do_not_score() -> None:
    finding = SimpleNamespace(
        id="tb2",
        event_type="ok",
        eventType="ok",
        timestamp_window_ms=(0, 0),
        timestampWindowMs=(0, 0),
        attribution="candidate",
        phase="live_exam",
        severity="high",
        verdict="cleared",
        reasoning="No issue",
    )
    result = reconcile_findings([], [finding])
    assert result.score.total == 0
    assert result.score.band == "Clear"
