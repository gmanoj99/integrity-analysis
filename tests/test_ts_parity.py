"""TS parity tests for merge analysis, deliberation rules, and evidence bundle."""

from types import SimpleNamespace

from integrity_review_pipeline.deliberation.rules import (
    RawCandidateSignal,
    apply_capture_cap,
    apply_confidence_caps,
    compute_deliberation_version_hash,
    has_definitive_solo_evidence,
)
from integrity_review_pipeline.evidence_bundle.clips import compute_evidence_bundle_version_hash
from integrity_review_pipeline.scoring.merge_analysis import (
    MergedFinding,
    _compute_finding_weight,
    reconcile_findings,
)


def _signal(**overrides) -> RawCandidateSignal:
    base = RawCandidateSignal(
        signal_type="possible_second_person_involvement",
        episode_ref="ep1",
        hypothesis_honest={"supporting": [], "contradicting": []},
        hypothesis_assisted={"supporting": ["interaction"], "contradicting": []},
        resolution="assisted",
        confidence=0.8,
        innocent_explanation_considered=True,
        why_rejected="",
        machine_facts_cited=[],
        observations_cited=[],
        baseline_metrics_cited=[],
    )
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


def test_track_b_only_confirmed_has_zero_weight_like_ts() -> None:
    finding = MergedFinding(
        id="mf_1",
        corroboration_status="track_b_only",
        merged_verdict="confirmed",
        event_type="phone_usage",
        attribution="candidate",
        phase="live_exam",
        severity="high",
        reasoning="strong",
        score_weight=0.0,
    )
    assert _compute_finding_weight(finding) == 0.0


def test_track_b_only_ai_detected_scores_at_085() -> None:
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
    assert result.merged_findings[0].merged_verdict == "ai_detected"
    assert result.merged_findings[0].score_weight == 0.85
    assert result.score.total >= 26


def test_gaze_diverted_with_interaction_is_definitive_solo() -> None:
    signal = _signal(
        observations_cited=[
            "w_0_20000.people.secondPersonInteracting=yes",
            "w_0_20000.attention.gazeDirection=left",
        ]
    )
    assert has_definitive_solo_evidence(signal) is True


def test_capture_cap_thresholds_match_ts() -> None:
    capped, applied = apply_capture_cap(0.95, 0.25)
    assert applied is True
    assert capped == 0.50
    capped, applied = apply_capture_cap(0.95, 0.50)
    assert applied is True
    assert capped == 0.70


def test_informative_content_cap_thresholds_match_ts() -> None:
    final, _, informative_applied = apply_confidence_caps(0.95, 1.0, 0.20)
    assert informative_applied is True
    assert final == 0.50


def test_version_hashes_include_artifact_registry_inputs() -> None:
    deliberation_hash = compute_deliberation_version_hash("p", "b", "none")
    assert len(deliberation_hash) == 16
    bundle_hash = compute_evidence_bundle_version_hash(deliberation_hash)
    assert len(bundle_hash) == 16
    assert bundle_hash != deliberation_hash
