"""Focused tests for deliberation rules."""

from integrity_review_pipeline.deliberation.rules import (
    RawCandidateSignal,
    apply_capture_cap,
    derive_category,
    filter_audio_quotes_against_perception,
    has_definitive_solo_evidence,
    validate_conjunction_rule,
    validate_smart_student_guard,
)
from integrity_review_pipeline.contracts.deliberation import (
    HypothesisArm,
    ValidatedSignal,
)


def _signal(**overrides) -> RawCandidateSignal:
    base = RawCandidateSignal(
        signal_type="possible_external_consultation",
        episode_ref="ep1",
        hypothesis_honest={"supporting": [], "contradicting": []},
        hypothesis_assisted={"supporting": ["phone visible"], "contradicting": []},
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


def test_smart_student_guard_rejects_speed_only() -> None:
    signal = _signal(
        machine_facts_cited=["TYPING_STOPPED"],
        baseline_metrics_cited=["typingSpeedWpm"],
    )
    assert validate_smart_student_guard(signal) is False


def test_definitive_solo_phone_allows_single_citation() -> None:
    signal = _signal(
        observations_cited=["w_0_20000.hands.objectInHand=phone"],
    )
    assert has_definitive_solo_evidence(signal) is True
    assert validate_conjunction_rule(signal) is True


def test_conjunction_requires_two_modalities_without_definitive() -> None:
    signal = _signal(machine_facts_cited=["LARGE_PASTE"])
    assert validate_conjunction_rule(signal) is False
    signal = _signal(
        machine_facts_cited=["LARGE_PASTE"],
        observations_cited=["w_0_20000.objects.phoneVisible=yes"],
    )
    assert validate_conjunction_rule(signal) is True


def test_derive_category_clear_without_active_signals() -> None:
    assert derive_category([]) == "CLEAR"


def test_apply_capture_cap_at_30_percent() -> None:
    capped, applied = apply_capture_cap(0.9, 0.29)
    assert applied is True
    assert capped == 0.50


def test_filter_audio_quotes_against_perception() -> None:
    quotes = filter_audio_quotes_against_perception(
        ["can you read option b aloud"],
        ["The helper asked: can you read option b aloud?"],
    )
    assert quotes == ["can you read option b aloud"]


def test_derive_category_strong_evidence_requires_assisted_and_two_modalities() -> None:
    signals = [
        ValidatedSignal(
            signal_id="sig_1",
            signal_type="possible_external_consultation",
            hypothesis_honest=HypothesisArm(),
            hypothesis_assisted=HypothesisArm(supporting=["x"]),
            resolution="assisted",
            confidence=0.9,
            machine_facts_cited=["LARGE_PASTE"],
            observations_cited=["w_0_20000.objects.phoneVisible=yes"],
            baseline_metrics_cited=[],
            source_types=["machine_fact", "visual_observation"],
        )
    ]
    assert derive_category(signals) == "STRONG_EVIDENCE"
