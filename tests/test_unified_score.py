from integrity_review_pipeline.findings.contracts import EvidenceFinding
from integrity_review_pipeline.scoring.unified_score import compute_unified_score


def test_strong_track_b_only_finding_requires_review() -> None:
    finding = EvidenceFinding(
        id="tb-phone",
        source="video",
        event_type="phone_usage",
        timestamp_window_ms=(10_000, 25_000),
        attribution="candidate",
        phase="live_exam",
        severity="high",
        evidence_ref="w_10000_25000",
        verdict="flagged",
        evidence_strength="strong",
        reasoning="Candidate held a phone.",
    )
    result = compute_unified_score(
        [],
        [finding],
        usable_window_ratio=0.9,
        informative_content_ratio=0.9,
    )
    assert result.category == "REVIEW_REQUIRED"
    assert result.confidence > 0.0
    assert result.supporting_signal_ids == ["tb-phone"]


def test_clear_confidence_reflects_evidence_coverage() -> None:
    result = compute_unified_score(
        [],
        [],
        usable_window_ratio=0.8,
        informative_content_ratio=0.9,
    )
    assert result.category == "CLEAR"
    assert result.confidence == 0.9


def test_short_gaze_does_not_force_review() -> None:
    finding = EvidenceFinding(
        id="tb-gaze",
        source="video",
        event_type="suspicious_eye_movement",
        timestamp_window_ms=(10_000, 13_000),
        attribution="candidate",
        phase="live_exam",
        severity="medium",
        evidence_ref="w_10000_13000",
        verdict="flagged",
        evidence_strength="moderate",
        reasoning="Brief glance away.",
    )
    result = compute_unified_score(
        [],
        [finding],
        usable_window_ratio=0.9,
        informative_content_ratio=0.9,
    )
    assert result.category == "CLEAR"
    assert result.scored_track_b_ids == []

