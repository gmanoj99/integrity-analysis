from __future__ import annotations

from pydantic import Field

from ..contracts.base import ContractModel

BASELINE_VERSION = "scope3-v6"


class MetricSeries(ContractModel):
    samples: list[float]
    mean: float
    median: float
    stddev: float
    min: float
    max: float
    z_scores: list[float]
    valid_count: int


class PasteSizeDistribution(ContractModel):
    bucket_1_50: float
    bucket_51_200: float
    bucket_201_500: float
    bucket_500plus: float


class MachineFactMetrics(ContractModel):
    session_duration_ms: int
    session_duration_min: float
    typing_burst_count: int
    typing_burst_durations: MetricSeries
    iki_distribution: MetricSeries
    typing_speed_wpm: MetricSeries
    idle_gap_durations: MetricSeries
    typing_cadence_regularity: float
    text_correction_count: int
    corrections_per_burst: float
    paste_platform_count: int
    paste_platform_frequency_per_min: float
    large_paste_count: int
    large_paste_sizes: MetricSeries
    large_paste_interval_ms: MetricSeries
    large_paste_size_distribution: PasteSizeDistribution
    large_paste_frequency_per_min: float
    window_blur_count: int
    window_blur_total_duration_ms: int
    window_blur_frequency_per_min: float
    section_latency_ms: MetricSeries
    mcq_answer_selected_count: int
    mcq_answer_changed_count: int
    large_paste_count_by_question: dict[str, int] = Field(default_factory=dict)
    text_correction_count_by_question: dict[str, int] = Field(default_factory=dict)


class CoverageMetrics(ContractModel):
    total_windows: int
    video_available_windows: int
    usable_window_ratio: float
    mean_observation_confidence: float


class CohortCandidateStat(ContractModel):
    ref_id: str
    z_score: float | None = None
    percentile: float | None = None


class StatisticalBaseline(ContractModel):
    model_config = ContractModel.model_config | {"frozen": True}

    candidate_id: str
    assessment_id: str
    produced_at: str
    baseline_version: str = BASELINE_VERSION
    machine_fact_metrics: MachineFactMetrics
    coverage_metrics: CoverageMetrics | None = None
    cohort_candidate_stats: list[CohortCandidateStat] = Field(default_factory=list)
    cohort_capabilities: list[str] = Field(default_factory=list)
