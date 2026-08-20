from integrity_review_pipeline.baseline import build_statistical_baseline
from integrity_review_pipeline.contracts.evidence import ExamMode
from integrity_review_pipeline.contracts.timeline import CanonicalTimeline, MasterTimeline
from integrity_review_pipeline.correlation import derive_correlated_signals
from integrity_review_pipeline.findings import derive_video_findings
from integrity_review_pipeline.findings.contracts import VideoObservationWindow
from integrity_review_pipeline.machine_facts import build_machine_facts
from integrity_review_pipeline.machine_facts.contracts import ClientReportedEvent, RrwebChunkEvents
from integrity_review_pipeline.machine_facts.kinds import MachineFactKind


def test_paste_after_blur_factor() -> None:
    t0 = 1_700_000_000_000
    timeline = MasterTimeline(
        candidate_id="c1",
        assessment_id="a1",
        session_start_ms=t0,
        duration_ms=120_000,
        canonical_timeline=CanonicalTimeline(
            T0=t0,
            total_duration_ms=120_000,
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
    rrweb = [
        {
            "type": 5,
            "timestamp": t0 + 10_000,
            "data": {"payload": {"data": {"data": "TAB_SWITCH"}}},
        },
        {"type": 3, "timestamp": t0 + 11_000, "data": {"source": 5, "id": 7, "text": "x"}},
        {
            "type": 3,
            "timestamp": t0 + 12_000,
            "data": {"source": 5, "id": 7, "text": "x" + ("y" * 60)},
        },
    ]
    bundle = build_machine_facts(
        timeline,
        exam_mode=ExamMode.RRWEB,
        rrweb_chunks=[RrwebChunkEvents(chunk_id="k1", sequence=0, events=rrweb)],
        client_reported_events=[
            ClientReportedEvent(
                timestamp_ms=t0 + 10_000,
                kind=MachineFactKind.TAB_SWITCH.value,
            )
        ],
    )
    baseline = build_statistical_baseline(bundle)
    video = derive_video_findings(
        [
            VideoObservationWindow(
                window_id="g1",
                start_ms=8_000,
                end_ms=15_000,
                gaze_direction="off_screen",
            )
        ]
    )
    correlated = derive_correlated_signals(
        machine_facts=bundle,
        baseline=baseline,
        video_findings=video.findings,
    )
    factor_ids = {item.factor_id for item in correlated.contributions}
    assert "paste_after_blur" in factor_ids or "gaze_off_then_input" in factor_ids
    assert correlated.total_weighted_score <= 100
