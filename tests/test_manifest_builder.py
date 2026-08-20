from integrity_review_pipeline.contracts.evidence import EvidenceType, ExamMode
from integrity_review_pipeline.contracts.review import ReviewInput
from integrity_review_pipeline.io.manifest_builder import (
    SessionUrlKind,
    build_manifest_set,
    classify_session_url,
    parse_duration_ms,
)


SECTION = {
    "examAttemptId": "attempt-1",
    "examId": "exam-1",
    "sectionType": "mcq",
}


def review_input(session_recordings: list[str]) -> ReviewInput:
    return ReviewInput.model_validate(
        {
            "candidateId": "candidate-1",
            "assessmentId": "assessment-1",
            "cameraRecordings": [
                "https://media.example/camera/attempt-1/1700000000000__60000.webm"
            ],
            "sessionRecordings": session_recordings,
            "activityTimeline": [{"eventType": "ASSESSMENT_STARTED", "timestamp": 1}],
            "sections": [SECTION],
        }
    )


def test_classifies_all_supported_session_formats() -> None:
    assert (
        classify_session_url("https://media.example/session/attempt-1/1700000000000.json")
        == SessionUrlKind.RRWEB_JSON
    )
    assert (
        classify_session_url(
            "https://media.example/session/attempt-1/1700000000000.json.gz"
        )
        == SessionUrlKind.RRWEB_JSON
    )
    assert (
        classify_session_url(
            "https://media.example/session/attempt-1/v3/1700000000000__60000.webm"
        )
        == SessionUrlKind.SCREEN_WEBM
    )
    assert (
        classify_session_url(
            "https://media.example/session/attempt-1/v2/1700000000000__60000.json"
        )
        == SessionUrlKind.SCREEN_JSON
    )
    assert (
        classify_session_url(
            "https://media.example/session/attempt-1/v3/"
            "1700000000000__60000__metadata.json.gz"
        )
        == SessionUrlKind.SIDECAR_METADATA
    )


def test_builds_rrweb_mode_manifests() -> None:
    manifests = build_manifest_set(
        review_input(
            ["https://media.example/session/attempt-1/1700000000000.json"]
        )
    )
    assert manifests.exam_mode == ExamMode.RRWEB
    assert manifests.by_type(EvidenceType.VIDEO).total_chunks == 1
    assert manifests.by_type(EvidenceType.KEYSTROKE_DATA).total_chunks == 1
    assert manifests.by_type(EvidenceType.SCREEN_RECORDING).total_chunks == 0


def test_builds_screen_mode_with_sidecar() -> None:
    manifests = build_manifest_set(
        review_input(
            [
                "https://media.example/session/attempt-1/v3/"
                "1700000000000__60000.webm",
                "https://media.example/session/attempt-1/v3/"
                "1700000000000__60000__metadata.json.gz",
            ]
        )
    )
    screen = manifests.by_type(EvidenceType.SCREEN_RECORDING)
    assert manifests.exam_mode == ExamMode.SCREEN
    assert screen.total_chunks == 2
    assert screen.chunks[1].sidecar_role == "metadata"
    assert parse_duration_ms(str(screen.chunks[0].signed_url)) == 60_000


def test_rejects_mixed_screen_and_rrweb() -> None:
    try:
        build_manifest_set(
            review_input(
                [
                    "https://media.example/session/attempt-1/1700000000000.json",
                    "https://media.example/session/attempt-1/v3/"
                    "1700000000000__60000.webm",
                ]
            )
        )
    except ValueError as error:
        assert "either screen media or rrweb JSON" in str(error)
    else:
        raise AssertionError("mixed session modes must fail")
