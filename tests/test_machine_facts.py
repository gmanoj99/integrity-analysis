import json

from integrity_review_pipeline.contracts.evidence import ExamMode
from integrity_review_pipeline.contracts.timeline import (
    ActivityAnchor,
    CanonicalTimeline,
    MasterTimeline,
    SessionSection,
)
from integrity_review_pipeline.machine_facts import (
    MachineFactKind,
    analyze_keystroke_chunk,
    build_machine_facts,
    is_reportable_paste,
    normalize_rrweb_event_kind,
)
from integrity_review_pipeline.machine_facts.contracts import RrwebChunkEvents
from integrity_review_pipeline.media.keystroke_reader import decode_rrweb_chunk


def _timeline() -> MasterTimeline:
    return MasterTimeline(
        candidate_id="candidate-1",
        assessment_id="assessment-1",
        session_start_ms=1_700_000_000_000,
        duration_ms=120_000,
        canonical_timeline=CanonicalTimeline(
            T0=1_700_000_000_000,
            total_duration_ms=120_000,
            anchors=[
                ActivityAnchor(
                    order=0,
                    type="ASSESSMENT_STARTED",
                    epoch_ms=1_700_000_000_000,
                    session_offset_ms=0,
                ),
                ActivityAnchor(
                    order=1,
                    type="SECTION_STARTED",
                    epoch_ms=1_700_000_010_000,
                    session_offset_ms=10_000,
                    section_id="mcq",
                ),
            ],
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


def test_rrweb_normalizer_maps_window_blur() -> None:
    assert normalize_rrweb_event_kind("WINDOW_BLUR") == MachineFactKind.WINDOW_BLUR


def test_analyze_keystroke_chunk_extracts_custom_and_gap() -> None:
    events = [
        {"type": 5, "timestamp": 1_700_000_000_000, "data": {"tag": "WINDOW_BLUR"}},
        {
            "type": 3,
            "timestamp": 1_700_000_020_000,
            "data": {"source": 5, "id": 1, "text": "hello"},
        },
    ]
    facts = analyze_keystroke_chunk(events)
    kinds = {fact.kind for fact in facts}
    assert MachineFactKind.WINDOW_BLUR.value in kinds


def test_build_machine_facts_rrweb_large_paste() -> None:
    t0 = 1_700_000_000_000
    prev = "a" * 10
    nxt = prev + ("x" * 60)
    events = [
        {"type": 3, "timestamp": t0 + 5_000, "data": {"source": 5, "id": 99, "text": prev}},
        {"type": 3, "timestamp": t0 + 6_000, "data": {"source": 5, "id": 99, "text": nxt}},
    ]
    bundle = build_machine_facts(
        _timeline(),
        exam_mode=ExamMode.RRWEB,
        rrweb_chunks=[
            RrwebChunkEvents(chunk_id="keystrokeData-mcq-1", sequence=0, events=events)
        ],
    )
    pastes = [fact for fact in bundle.facts if fact.kind == MachineFactKind.LARGE_PASTE.value]
    assert len(pastes) == 1
    assert is_reportable_paste(pastes[0])
    assert bundle.exam_mode == "rrweb"


def test_screen_mode_skips_rrweb_typing() -> None:
    bundle = build_machine_facts(_timeline(), exam_mode=ExamMode.SCREEN)
    assert all(
        fact.kind not in {MachineFactKind.LARGE_PASTE.value, MachineFactKind.TYPING_STARTED.value}
        for fact in bundle.facts
    )
    assert any("screen" in note.lower() for note in bundle.limitations)


def test_decode_rrweb_chunk_plain_array() -> None:
    event = {"type": 5, "timestamp": 123, "data": {"tag": "COPY"}}
    assert decode_rrweb_chunk(json.dumps([event]).encode()) == [event]


def test_kind_registry_matches_typescript_source_exactly() -> None:
    assert [kind.value for kind in MachineFactKind] == [
        "FACE_NOT_VISIBLE_WARNING",
        "FACE_WARNING_DISMISSED",
        "CAMERA_BLOCKED",
        "multiple_faces",
        "incident.moved_out_of_window",
        "TAB_SWITCH",
        "WINDOW_BLUR",
        "WINDOW_FOCUS",
        "FULLSCREEN_EXIT",
        "COPY",
        "PASTE",
        "LARGE_TEXT_INSERTION",
        "MIC_DISABLED",
        "SECOND_MONITOR_DETECTED",
        "NOISE_DETECTED",
        "ACTIVITY_GAP",
        "TYPING_STARTED",
        "TYPING_STOPPED",
        "LARGE_PASTE",
        "TEXT_CORRECTION",
        "RIGHT_CLICK",
        "ASSESSMENT_STARTED",
        "ALL_SECTIONS_COMPLETED",
        "SECTION_STARTED",
        "SECTION_COMPLETED",
        "SECTION_TERMINATED",
        "SECTION_TIMEOUT",
        "SECTION_QUIT_BY_USER",
        "MCQ_ANSWER_SELECTED",
        "MCQ_ANSWER_CHANGED",
        "AUTOFORMAT",
        "QUESTION_ENTERED",
        "QUESTION_EXITED",
        "ANSWER_SUBMITTED",
        "SECTION_SCORE",
        "QUESTION_ATTEMPTED",
        "CODE_SUBMISSION",
        "DIFFICULTY_BAND_SCORE",
        "PLAGIARISM_MATCH",
        "TEST_RESULT_OBSERVED",
        "SCREEN_EXTERNAL_PASTE",
        "SCREEN_EXTERNAL_RESOURCE",
        "SCREEN_AI_ASSISTANT_UI",
        "SCREEN_SECONDARY_WORKSPACE",
        "SCREEN_FULLSCREEN_LOST",
        "UNKNOWN",
    ]


def test_custom_event_nested_payload_preserves_primary_evidence() -> None:
    facts = analyze_keystroke_chunk(
        [
            {
                "type": 5,
                "timestamp": 123,
                "data": {
                    "tag": "custom-log-event",
                    "payload": {
                        "data": {
                            "data": "USER_FACE_NOT_DETECTED_POPUP_APPEARED",
                            "message": "Face missing",
                            "image": "data:image/jpeg;base64,abc",
                        }
                    },
                },
            }
        ]
    )
    assert facts[0].kind == MachineFactKind.FACE_NOT_VISIBLE_WARNING.value
    assert facts[0].detail == {
        "rawEventTag": "USER_FACE_NOT_DETECTED_POPUP_APPEARED",
        "message": "Face missing",
        "snapshot": "data:image/jpeg;base64,abc",
    }


def test_rrweb_typing_mcq_mutation_and_reverted_paste_parity() -> None:
    t0 = 1_700_000_000_000
    section = SessionSection(
        section_id="coding",
        label="Coding",
        section_type="CODING",
        start_ms=0,
        end_ms=120_000,
    )
    timeline = _timeline().model_copy(update={"sections": [section]})
    option_a = "11111111-1111-1111-1111-111111111111"
    option_b = "22222222-2222-2222-2222-222222222222"
    base = "abcdefghij"
    pasted = base + ("x" * 60)
    events = [
        {"type": 3, "timestamp": t0 + 1_000, "data": {"source": 5, "id": 10, "text": option_a, "isChecked": True}},
        {"type": 3, "timestamp": t0 + 2_000, "data": {"source": 5, "id": 11, "text": option_b, "isChecked": True}},
        {"type": 3, "timestamp": t0 + 2_005, "data": {"source": 5, "id": 10, "text": option_a, "isChecked": False}},
        {"type": 3, "timestamp": t0 + 3_000, "data": {"source": 5, "id": 20, "text": base}},
        {"type": 3, "timestamp": t0 + 4_000, "data": {"source": 5, "id": 20, "text": pasted}},
        {"type": 3, "timestamp": t0 + 5_000, "data": {"source": 5, "id": 20, "text": base}},
        {
            "type": 3,
            "timestamp": t0 + 6_000,
            "data": {
                "source": 0,
                "texts": [{"id": 50, "value": "3 / 5 test cases passed"}],
            },
        },
        {"type": 3, "timestamp": t0 + 7_000, "data": {"source": 2, "type": 3, "x": 4, "y": 8}},
    ]
    bundle = build_machine_facts(
        timeline,
        exam_mode=ExamMode.RRWEB,
        rrweb_chunks=[
            RrwebChunkEvents(chunk_id="keystrokeData-mcq-1", sequence=0, events=events)
        ],
    )
    by_kind = {}
    for fact in bundle.facts:
        by_kind.setdefault(fact.kind, []).append(fact)
    assert len(by_kind[MachineFactKind.MCQ_ANSWER_SELECTED.value]) == 2
    assert by_kind[MachineFactKind.MCQ_ANSWER_CHANGED.value][0].detail[
        "selectionChangeCount"
    ] == 1
    paste = by_kind[MachineFactKind.LARGE_PASTE.value][0]
    correction = by_kind[MachineFactKind.TEXT_CORRECTION.value][0]
    assert paste.detail["insertionLocality"] == "append_at_end"
    assert paste.detail["revertedEdit"] is True
    assert correction.detail["revertsPaste"] is True
    assert not is_reportable_paste(paste)
    assert by_kind[MachineFactKind.TEST_RESULT_OBSERVED.value][0].detail[
        "passedCount"
    ] == 3
    assert by_kind[MachineFactKind.RIGHT_CLICK.value][0].detail["x"] == 4


def test_autoformat_and_summary_categories() -> None:
    t0 = 1_700_000_000_000
    events = [
        {"type": 3, "timestamp": t0 + 1_000, "data": {"source": 5, "id": 7, "text": "a=1"}},
        {"type": 3, "timestamp": t0 + 1_100, "data": {"source": 5, "id": 7, "text": "a = 1;"}},
    ]
    bundle = build_machine_facts(
        _timeline(),
        exam_mode=ExamMode.RRWEB,
        rrweb_chunks=[RrwebChunkEvents(chunk_id="keystrokeData-text-1", sequence=0, events=events)],
    )
    assert bundle.summary.by_kind[MachineFactKind.AUTOFORMAT.value] == 1
    assert bundle.summary.categories.input_and_clipboard >= 1
    assert bundle.summary.categories.assessment_lifecycle == 1
    assert bundle.summary.categories.section_lifecycle == 1
    assert any("Topin-derived" in item for item in bundle.limitations)
