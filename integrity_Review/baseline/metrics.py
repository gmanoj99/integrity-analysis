"""Candidate-only metric aggregation from MachineFacts."""

from __future__ import annotations

import math

from ..findings.contracts import VideoObservationWindow
from ..machine_facts.contracts import MachineFactsBundle
from ..machine_facts.kinds import MachineFactKind
from ..machine_facts.paste_utils import is_reportable_paste
from .contracts import (
    CoverageMetrics,
    MachineFactMetrics,
    MetricSeries,
    PasteSizeDistribution,
)

NON_KEYBOARD_SECTION_TYPES = {"MCQ", "mcq", "MULTIPLE_CHOICE", "multiple_choice"}


def build_metric_series(samples: list[float]) -> MetricSeries:
    valid = [value for value in samples if math.isfinite(value)]
    if not valid:
        return MetricSeries(
            samples=samples,
            mean=0.0,
            median=0.0,
            stddev=0.0,
            min=0.0,
            max=0.0,
            z_scores=[0.0 for _ in samples],
            valid_count=0,
        )
    mean = sum(valid) / len(valid)
    sorted_valid = sorted(valid)
    mid = len(sorted_valid) // 2
    median = (
        (sorted_valid[mid - 1] + sorted_valid[mid]) / 2
        if len(sorted_valid) % 2 == 0
        else sorted_valid[mid]
    )
    variance = sum((value - mean) ** 2 for value in valid) / len(valid)
    stddev = math.sqrt(variance)
    z_scores = [
        0.0 if not math.isfinite(sample) else ((sample - mean) / stddev if stddev > 0 else 0.0)
        for sample in samples
    ]
    return MetricSeries(
        samples=samples,
        mean=mean,
        median=median,
        stddev=stddev,
        min=sorted_valid[0],
        max=sorted_valid[-1],
        z_scores=z_scores,
        valid_count=len(valid),
    )


def _paste_size_distribution(sizes: list[int]) -> PasteSizeDistribution:
    if not sizes:
        return PasteSizeDistribution(
            bucket_1_50=0.0,
            bucket_51_200=0.0,
            bucket_201_500=0.0,
            bucket_500plus=0.0,
        )
    buckets = {"bucket_1_50": 0, "bucket_51_200": 0, "bucket_201_500": 0, "bucket_500plus": 0}
    for size in sizes:
        if size <= 50:
            buckets["bucket_1_50"] += 1
        elif size <= 200:
            buckets["bucket_51_200"] += 1
        elif size <= 500:
            buckets["bucket_201_500"] += 1
        else:
            buckets["bucket_500plus"] += 1
    total = len(sizes)
    return PasteSizeDistribution(
        bucket_1_50=buckets["bucket_1_50"] / total,
        bucket_51_200=buckets["bucket_51_200"] / total,
        bucket_201_500=buckets["bucket_201_500"] / total,
        bucket_500plus=buckets["bucket_500plus"] / total,
    )


def compute_machine_fact_metrics(bundle: MachineFactsBundle) -> MachineFactMetrics:
    facts = sorted(bundle.facts, key=lambda fact: fact.start_offset_ms)
    duration_ms = bundle.duration_ms
    duration_min = duration_ms / 60_000 if duration_ms else 0.0

    typing_stops = [
        fact
        for fact in facts
        if fact.kind == MachineFactKind.TYPING_STOPPED.value
        and fact.detail.get("sectionType") not in NON_KEYBOARD_SECTION_TYPES
    ]
    typing_starts = [fact for fact in facts if fact.kind == MachineFactKind.TYPING_STARTED.value]
    correction_facts = [fact for fact in facts if fact.kind == MachineFactKind.TEXT_CORRECTION.value]

    burst_durations: list[float] = []
    burst_wpm: list[float] = []
    for stop in typing_stops:
        detail = stop.detail
        duration = detail.get("durationMs")
        dur_ms = float(duration) if isinstance(duration, (int, float)) and duration > 0 else math.nan
        burst_durations.append(dur_ms)
        cps = detail.get("avgCharsPerSecond")
        if math.isfinite(dur_ms) and isinstance(cps, (int, float)):
            wpm = (float(cps) * 60) / 5
            burst_wpm.append(wpm if wpm <= 200 else math.nan)
        else:
            burst_wpm.append(math.nan)

    typing_burst_durations = build_metric_series(burst_durations)
    iki_distribution = build_metric_series(list(burst_durations))
    cadence = (
        1 / (1 + typing_burst_durations.stddev / typing_burst_durations.mean)
        if typing_burst_durations.mean > 0
        else 0.0
    )

    idle_gaps: list[float] = []
    for stop in typing_stops:
        next_start = next(
            (start for start in typing_starts if start.start_offset_ms > stop.start_offset_ms),
            None,
        )
        if next_start:
            idle_gaps.append(float(next_start.start_offset_ms - stop.start_offset_ms))

    platform_pastes = [fact for fact in facts if fact.kind == MachineFactKind.PASTE.value]
    large_pastes = [fact for fact in facts if is_reportable_paste(fact)]
    large_sizes = [float(fact.detail.get("charsAdded") or math.nan) for fact in large_pastes]
    large_intervals = [
        float(large_pastes[index].start_offset_ms - large_pastes[index - 1].start_offset_ms)
        for index in range(1, len(large_pastes))
    ]

    blur_events = sorted(
        (fact for fact in facts if fact.kind == MachineFactKind.WINDOW_BLUR.value),
        key=lambda fact: fact.start_offset_ms,
    )
    focus_events = sorted(
        (fact for fact in facts if fact.kind == MachineFactKind.WINDOW_FOCUS.value),
        key=lambda fact: fact.start_offset_ms,
    )
    blur_duration = 0
    focus_idx = 0
    for blur in blur_events:
        while focus_idx < len(focus_events) and focus_events[focus_idx].start_offset_ms <= blur.start_offset_ms:
            focus_idx += 1
        if focus_idx < len(focus_events):
            blur_duration += focus_events[focus_idx].start_offset_ms - blur.start_offset_ms
            focus_idx += 1

    section_starts = [fact for fact in facts if fact.kind == MachineFactKind.SECTION_STARTED.value]
    section_ends = [fact for fact in facts if fact.kind == MachineFactKind.SECTION_COMPLETED.value]
    section_latencies: list[float] = []
    for start in section_starts:
        section_id = start.detail.get("sectionId")
        end = next(
            (fact for fact in section_ends if fact.detail.get("sectionId") == section_id),
            None,
        )
        if end:
            section_latencies.append(float(end.start_offset_ms - start.start_offset_ms))

    large_paste_by_question: dict[str, int] = {}
    correction_by_question: dict[str, int] = {}
    for fact in facts:
        qn = fact.detail.get("questionNumber")
        if not isinstance(qn, (int, float)):
            continue
        key = str(int(qn))
        if is_reportable_paste(fact):
            large_paste_by_question[key] = large_paste_by_question.get(key, 0) + 1
        elif fact.kind == MachineFactKind.TEXT_CORRECTION.value:
            correction_by_question[key] = correction_by_question.get(key, 0) + 1

    return MachineFactMetrics(
        session_duration_ms=duration_ms,
        session_duration_min=duration_min,
        typing_burst_count=len(typing_stops),
        typing_burst_durations=typing_burst_durations,
        iki_distribution=iki_distribution,
        typing_speed_wpm=build_metric_series(burst_wpm),
        idle_gap_durations=build_metric_series(idle_gaps),
        typing_cadence_regularity=cadence,
        text_correction_count=len(correction_facts),
        corrections_per_burst=0.0,
        paste_platform_count=len(platform_pastes),
        paste_platform_frequency_per_min=len(platform_pastes) / duration_min if duration_min else 0.0,
        large_paste_count=len(large_pastes),
        large_paste_sizes=build_metric_series(large_sizes),
        large_paste_interval_ms=build_metric_series(large_intervals),
        large_paste_size_distribution=_paste_size_distribution(
            [int(value) for value in large_sizes if math.isfinite(value)]
        ),
        large_paste_frequency_per_min=len(large_pastes) / duration_min if duration_min else 0.0,
        window_blur_count=len(blur_events),
        window_blur_total_duration_ms=blur_duration,
        window_blur_frequency_per_min=len(blur_events) / duration_min if duration_min else 0.0,
        section_latency_ms=build_metric_series(section_latencies),
        mcq_answer_selected_count=sum(
            1 for fact in facts if fact.kind == MachineFactKind.MCQ_ANSWER_SELECTED.value
        ),
        mcq_answer_changed_count=sum(
            1 for fact in facts if fact.kind == MachineFactKind.MCQ_ANSWER_CHANGED.value
        ),
        large_paste_count_by_question=large_paste_by_question,
        text_correction_count_by_question=correction_by_question,
    )


def compute_coverage_metrics(windows: list[VideoObservationWindow]) -> CoverageMetrics:
    total = len(windows)
    available = sum(1 for window in windows if window.video_available)
    confidence = (
        sum(window.confidence for window in windows) / total if total else 0.0
    )
    return CoverageMetrics(
        total_windows=total,
        video_available_windows=available,
        usable_window_ratio=available / total if total else 0.0,
        mean_observation_confidence=confidence,
    )
