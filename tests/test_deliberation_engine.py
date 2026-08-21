"""Focused tests for deliberation engine and episode inventory."""

import json

from types import SimpleNamespace

from integrity_review_pipeline.deliberation.engine import (
    DeliberationInput,
    _parse_raw_signals,
    build_deliberation_bundle,
)
from integrity_review_pipeline.deliberation.episodes import build_episode_inventory


def test_build_episode_inventory_from_machine_fact_seed() -> None:
    facts = [
        SimpleNamespace(
            id="f1",
            kind="LARGE_PASTE",
            start_offset_ms=5_000,
            end_offset_ms=6_000,
            detail={"sectionId": "coding"},
        )
    ]
    episodes = build_episode_inventory(
        machine_facts_bundle=SimpleNamespace(facts=facts),
        video_findings=[],
        correlated_signals=None,
        perception_bundle=None,
    )
    assert len(episodes) == 1
    assert episodes[0].episode_id == "ep1"
    assert episodes[0].machine_fact_kinds == ["LARGE_PASTE"]


def test_parse_raw_signals_accepts_snake_case_keys() -> None:
    episodes, signals = _parse_raw_signals(
        {
            "episode_analysis": [
                {
                    "episode_id": "ep1",
                    "time_range": "0-20s",
                    "will_emit_signal": True,
                }
            ],
            "candidate_signals": [
                {
                    "signal_type": "possible_external_consultation",
                    "episode_ref": "ep1",
                    "machine_facts_cited": ["LARGE_PASTE"],
                    "observations_cited": [],
                    "baseline_metrics_cited": [],
                }
            ],
        }
    )
    assert episodes[0].episode_id == "ep1"
    assert signals[0].signal_type == "possible_external_consultation"


def test_build_deliberation_bundle_validates_signals_from_raw_json() -> None:
    machine_facts = SimpleNamespace(
        facts=[SimpleNamespace(id="f1", kind="LARGE_PASTE", start_offset_ms=1000, end_offset_ms=2000, detail={})],
        duration_ms=60_000,
    )
    perception = SimpleNamespace(
        observations=[
            {
                "windowId": "w_0_20000",
                "videoAvailable": True,
                "objects": {"phoneVisible": "yes"},
                "people": {"secondPersonVisible": "no", "secondPersonInteracting": "no"},
                "hands": {"handsVisible": "yes", "objectInHand": "phone"},
                "attention": {"gazeDirection": "screen"},
                "identity": {"facePresent": "yes"},
                "interaction": {"candidateSpeaking": "no"},
            }
        ],
        windows=[SimpleNamespace(window_id="w_0_20000", phase="live_exam", start_ms=0, end_ms=20_000)],
    )
    baseline = SimpleNamespace(
        coverage_metrics=SimpleNamespace(usable_window_ratio=0.9),
        machine_fact_metrics=SimpleNamespace(large_paste_count=1),
    )
    raw = {
        "episodeAnalysis": [
            {
                "episodeId": "ep1",
                "timeRange": "0s-20s",
                "windowIds": ["w_0_20000"],
                "machineFactsInRange": ["LARGE_PASTE"],
                "episodeSummary": "Phone in hand during paste window",
                "suspiciousBehaviorType": "phone",
                "willEmitSignal": True,
            }
        ],
        "candidateSignals": [
            {
                "signalType": "possible_external_consultation",
                "episodeRef": "ep1",
                "hypothesis_honest": {"supporting": [], "contradicting": []},
                "hypothesis_assisted": {"supporting": ["Phone in candidate hand"], "contradicting": []},
                "resolution": "assisted",
                "confidence": 0.85,
                "innocentExplanationConsidered": True,
                "whyRejected": "",
                "machineFactsCited": ["LARGE_PASTE"],
                "observationsCited": ["w_0_20000.hands.objectInHand=phone"],
                "baselineMetricsCited": [],
            }
        ],
        "behaviorSummary": "Candidate used a phone during the exam.",
        "recommendation": "Review phone usage clip.",
        "reasoning": "Definitive phone-in-hand observation.",
        "keyReasons": ["Phone in candidate hand"],
        "category": "STRONG_EVIDENCE",
    }
    bundle = build_deliberation_bundle(
        DeliberationInput(
            candidate_id="c1",
            assessment_id="a1",
            machine_facts_bundle=machine_facts,
            perception_bundle=perception,
            statistical_baseline=baseline,
            video_findings=[],
        ),
        raw_llm_text=json.dumps(raw),
    )
    assert len(bundle.validated_signals) == 1
    assert bundle.recommendation.category in {"REVIEW_REQUIRED", "STRONG_EVIDENCE"}


def test_build_deliberation_bundle_accepts_string_confidence() -> None:
    machine_facts = SimpleNamespace(
        facts=[SimpleNamespace(id="f1", kind="LARGE_PASTE", start_offset_ms=1000, end_offset_ms=2000, detail={})],
        duration_ms=60_000,
    )
    perception = SimpleNamespace(
        observations=[
            {
                "windowId": "w_0_20000",
                "videoAvailable": True,
                "objects": {"phoneVisible": "yes"},
                "people": {"secondPersonVisible": "no", "secondPersonInteracting": "no"},
                "hands": {"handsVisible": "yes", "objectInHand": "phone"},
                "attention": {"gazeDirection": "screen"},
                "identity": {"facePresent": "yes"},
                "interaction": {"candidateSpeaking": "no"},
            }
        ],
        windows=[SimpleNamespace(window_id="w_0_20000", phase="live_exam", start_ms=0, end_ms=20_000)],
    )
    baseline = SimpleNamespace(
        coverage_metrics=SimpleNamespace(usable_window_ratio=0.9),
        machine_fact_metrics=SimpleNamespace(large_paste_count=1),
    )
    raw = {
        "episodeAnalysis": [
            {
                "episodeId": "ep1",
                "timeRange": "0s-20s",
                "windowIds": ["w_0_20000"],
                "machineFactsInRange": ["LARGE_PASTE"],
                "episodeSummary": "Phone in hand during paste window",
                "suspiciousBehaviorType": "phone",
                "willEmitSignal": True,
            }
        ],
        "candidateSignals": [
            {
                "signalType": "possible_external_consultation",
                "episodeRef": "ep1",
                "hypothesis_honest": {"supporting": [], "contradicting": []},
                "hypothesis_assisted": {"supporting": ["Phone in candidate hand"], "contradicting": []},
                "resolution": "assisted",
                "confidence": "high",
                "innocentExplanationConsidered": True,
                "whyRejected": "",
                "machineFactsCited": ["LARGE_PASTE"],
                "observationsCited": ["w_0_20000.hands.objectInHand=phone"],
                "baselineMetricsCited": [],
            }
        ],
        "behaviorSummary": "Candidate used a phone during the exam.",
        "recommendation": "Review phone usage clip.",
        "reasoning": "Definitive phone-in-hand observation.",
        "keyReasons": ["Phone in candidate hand"],
        "category": "STRONG_EVIDENCE",
    }
    bundle = build_deliberation_bundle(
        DeliberationInput(
            candidate_id="c1",
            assessment_id="a1",
            machine_facts_bundle=machine_facts,
            perception_bundle=perception,
            statistical_baseline=baseline,
            video_findings=[],
        ),
        raw_llm_text=json.dumps(raw),
    )
    assert len(bundle.validated_signals) == 1
    assert bundle.validated_signals[0].confidence == 0.5


def test_parse_integrity_story_accepts_list_proof_anchors() -> None:
    from integrity_review_pipeline.deliberation.rules import parse_integrity_story

    story = parse_integrity_story(
        {
            "headline": "Phone in hand",
            "whatHappened": "Candidate held a phone during the exam.",
            "proofAnchors": ["w_0_20000", "w_20000_40000"],
        }
    )
    assert story is not None
    assert story.proof_anchors.window_ids == ["w_0_20000", "w_20000_40000"]
    assert parse_integrity_story(["not", "a", "story"]) is None

