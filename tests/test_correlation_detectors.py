"""Tests ported from correlationEngine.test.ts (paste_after_blur)."""

from integrity_review_pipeline.baseline.contracts import (
    CoverageMetrics,
    MachineFactMetrics,
    MetricSeries,
    PasteSizeDistribution,
    StatisticalBaseline,
)
from integrity_review_pipeline.correlation.contracts import DetectCtx, DEFAULT_CORRELATION_CONFIG
from integrity_review_pipeline.correlation.detectors import detect_paste_after_blur
from integrity_review_pipeline.correlation.timeline import build_correlation_timeline
from integrity_review_pipeline.machine_facts.contracts import MachineFact
from integrity_review_pipeline.machine_facts.kinds import MachineFactKind


def _series() -> MetricSeries:
    return MetricSeries(samples=[], mean=0, median=0, stddev=0, min=0, max=0, z_scores=[], valid_count=0)


def _baseline() -> StatisticalBaseline:
    return StatisticalBaseline(
        candidate_id="c1",
        assessment_id="a1",
        produced_at="t",
        machine_fact_metrics=MachineFactMetrics(
            session_duration_ms=0,
            session_duration_min=0,
            typing_burst_count=0,
            typing_burst_durations=_series(),
            iki_distribution=_series(),
            typing_speed_wpm=_series(),
            idle_gap_durations=_series(),
            typing_cadence_regularity=0,
            text_correction_count=0,
            corrections_per_burst=0,
            paste_platform_count=0,
            paste_platform_frequency_per_min=0,
            large_paste_count=0,
            large_paste_sizes=_series(),
            large_paste_interval_ms=_series(),
            large_paste_size_distribution=PasteSizeDistribution(
                bucket_1_50=0, bucket_51_200=0, bucket_201_500=0, bucket_500plus=0
            ),
            large_paste_frequency_per_min=0,
            window_blur_count=0,
            window_blur_total_duration_ms=0,
            window_blur_frequency_per_min=0,
            section_latency_ms=_series(),
            mcq_answer_selected_count=0,
            mcq_answer_changed_count=0,
        ),
        coverage_metrics=CoverageMetrics(
            total_windows=1, video_available_windows=1, usable_window_ratio=1.0, mean_observation_confidence=1.0
        ),
    )


def _fact(kind: str, start: int, detail: dict | None = None) -> MachineFact:
    return MachineFact(
        id=f"{kind}-{start}",
        start_offset_ms=start,
        end_offset_ms=start,
        raw_timestamp_ms=start,
        kind=kind,
        source="rrweb_deterministic",
        evidence_source="keystroke",
        attribution="candidate",
        confidence=1.0,
        detail=detail or {},
    )


def test_paste_after_blur_within_15s() -> None:
    facts = [
        _fact(MachineFactKind.WINDOW_BLUR.value, 10_000, {"sectionId": "s1"}),
        _fact(MachineFactKind.LARGE_PASTE.value, 15_000, {"sectionId": "s1", "charsAdded": 80, "pasteOrigin": "external"}),
    ]
    timeline = build_correlation_timeline(facts, [], [], [])
    ctx = DetectCtx(timeline=timeline, boundaries=[], facts=facts, baseline=_baseline(), config=DEFAULT_CORRELATION_CONFIG)
    contribs = detect_paste_after_blur(ctx)
    assert any(c.factor_id == "paste_after_blur" for c in contribs)


def test_paste_after_blur_not_fired_at_20s() -> None:
    facts = [
        _fact(MachineFactKind.WINDOW_BLUR.value, 10_000, {"sectionId": "s1"}),
        _fact(MachineFactKind.LARGE_PASTE.value, 30_000, {"sectionId": "s1", "charsAdded": 80, "pasteOrigin": "external"}),
    ]
    timeline = build_correlation_timeline(facts, [], [], [])
    ctx = DetectCtx(timeline=timeline, boundaries=[], facts=facts, baseline=_baseline(), config=DEFAULT_CORRELATION_CONFIG)
    assert not detect_paste_after_blur(ctx)
