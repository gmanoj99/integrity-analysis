from integrity_review_pipeline.baseline import build_metric_series, build_statistical_baseline
from integrity_review_pipeline.contracts.evidence import ExamMode
from integrity_review_pipeline.contracts.timeline import CanonicalTimeline, MasterTimeline
from integrity_review_pipeline.machine_facts import build_machine_facts
from integrity_review_pipeline.machine_facts.contracts import RrwebChunkEvents


def test_build_metric_series_ignores_nan() -> None:
    series = build_metric_series([float("nan"), 10.0, 20.0])
    assert series.valid_count == 2
    assert series.mean == 15.0


def test_baseline_counts_large_pastes() -> None:
    t0 = 1_700_000_000_000
    events = [
        {"type": 3, "timestamp": t0 + 1_000, "data": {"source": 5, "id": 2, "text": "a"}},
        {"type": 3, "timestamp": t0 + 2_000, "data": {"source": 5, "id": 2, "text": "a" + ("b" * 55)}},
    ]
    timeline = MasterTimeline(
        candidate_id="c1",
        assessment_id="a1",
        session_start_ms=t0,
        duration_ms=60_000,
        canonical_timeline=CanonicalTimeline(
            T0=t0,
            total_duration_ms=60_000,
            anchors=[],
            sections=[],
            source="activity_logs",
        ),
        artifact_registry=[],
        gaps=[],
        sections=[],
        merged_video_segments=[],
        merged_keystroke_segments=[],
        merged_screen_segments=[],
    )
    bundle = build_machine_facts(
        timeline,
        exam_mode=ExamMode.RRWEB,
        rrweb_chunks=[RrwebChunkEvents(chunk_id="k1", sequence=0, events=events)],
    )
    baseline = build_statistical_baseline(bundle)
    assert baseline.machine_fact_metrics.large_paste_count >= 1
