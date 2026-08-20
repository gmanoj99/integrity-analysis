from integrity_review_pipeline.contracts.evidence import ExamMode
from integrity_review_pipeline.contracts.timeline import ActivityAnchor, CanonicalTimeline, MasterTimeline
from integrity_review_pipeline.findings import (
    derive_keystroke_findings,
    derive_screen_findings,
    derive_video_findings,
)
from integrity_review_pipeline.findings.contracts import ScreenObservation, ScreenPerceptionBundle, VideoObservationWindow
from integrity_review_pipeline.machine_facts import build_machine_facts
from integrity_review_pipeline.machine_facts.contracts import RrwebChunkEvents
from integrity_review_pipeline.machine_facts.kinds import MachineFactKind


def _timeline() -> MasterTimeline:
    return MasterTimeline(
        candidate_id="candidate-1",
        assessment_id="assessment-1",
        session_start_ms=1_700_000_000_000,
        duration_ms=120_000,
        canonical_timeline=CanonicalTimeline(
            T0=1_700_000_000_000,
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


def test_keystroke_findings_flag_external_paste() -> None:
    bundle = build_machine_facts(
        _timeline(),
        exam_mode=ExamMode.RRWEB,
        rrweb_chunks=[
            RrwebChunkEvents(
                chunk_id="keystrokeData-mcq-1",
                sequence=0,
                events=[
                    {
                        "type": 3,
                        "timestamp": 1_700_000_005_000,
                        "data": {"source": 5, "id": 1, "text": "seed"},
                    },
                    {
                        "type": 3,
                        "timestamp": 1_700_000_006_000,
                        "data": {"source": 5, "id": 1, "text": "seed" + ("Z" * 55)},
                    },
                ],
            )
        ],
    )
    result = derive_keystroke_findings(bundle)
    assert any(finding.event_type == "external_paste" for finding in result.findings)


def test_screen_findings_and_synthetic_facts() -> None:
    bundle = ScreenPerceptionBundle(
        observations=[
            ScreenObservation(
                chunk_id="screen-1",
                sequence=0,
                start_ms=10_000,
                end_ms=20_000,
                external_resource_labels=["chatgpt.com"],
                video_available=True,
            )
        ],
        total_chunks=1,
    )
    result, synthetic = derive_screen_findings(bundle)
    assert result.findings[0].event_type == "external_resource_open"
    assert synthetic[0].kind == MachineFactKind.SCREEN_EXTERNAL_RESOURCE.value


def test_video_findings_from_windows() -> None:
    windows = [
        VideoObservationWindow(
            window_id="w1",
            start_ms=0,
            end_ms=5_000,
            phone_visible="yes",
        )
    ]
    result = derive_video_findings(windows)
    assert result.findings[0].event_type == "phone_usage"
