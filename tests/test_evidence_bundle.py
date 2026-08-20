"""Focused tests for evidence bundle assembly."""

from types import SimpleNamespace

from integrity_review_pipeline.contracts.deliberation import (
    DeliberationBundle,
    DeliberationProvenance,
    DeliberationRecommendation,
    HypothesisArm,
    IntegrityStory,
    IntegrityStoryProofAnchors,
    ValidatedSignal,
)
from integrity_review_pipeline.evidence_bundle.service import (
    EvidenceBundleInput,
    assemble_evidence_bundle,
    build_media_index,
    resolve_offset_clip_ref,
)
from integrity_review_pipeline.contracts.evidence_bundle import MediaIndexEntry


def test_resolve_offset_clip_ref_uses_media_index_segments() -> None:
    media_index = [
        MediaIndexEntry(
            chunk_id="1700000060000__60000.webm",
            evidence_type="video",
            session_start_ms=0,
            session_end_ms=60_000,
            duration_ms=60_000,
            sequence=0,
        )
    ]
    clip = resolve_offset_clip_ref(event_ms=10_000, duration_ms=5_000, media_index=media_index)
    assert clip is not None
    assert clip.segments[0].chunk_id == "1700000060000__60000.webm"
    assert clip.segments[0].seek_to_ms == 10_000
    assert clip.cache_key


def test_assemble_evidence_bundle_omits_question_and_performance_sections() -> None:
    deliberation = DeliberationBundle(
        candidate_id="c1",
        assessment_id="a1",
        produced_at="2026-01-01T00:00:00Z",
        model_version="test",
        prompt_version="scope4-v13",
        provenance=DeliberationProvenance(
            deliberation_prompt_version="scope4-v13",
            perception_version_hash="p1",
            baseline_version_hash="b1",
            composite_version_hash="d1",
        ),
        validated_signals=[
            ValidatedSignal(
                signal_id="sig_1",
                signal_type="possible_external_consultation",
                hypothesis_honest=HypothesisArm(),
                hypothesis_assisted=HypothesisArm(supporting=["phone"]),
                resolution="assisted",
                confidence=0.8,
                machine_facts_cited=["LARGE_PASTE"],
                observations_cited=["w_0_20000.hands.objectInHand=phone"],
                baseline_metrics_cited=[],
                source_types=["machine_fact", "visual_observation"],
                integrity_story=IntegrityStory(
                    headline="Phone use",
                    what_happened="Candidate held phone.",
                    why_it_matters="Review clip.",
                    honest_alternative="Unused phone on desk.",
                    proof_anchors=IntegrityStoryProofAnchors(window_ids=["w_0_20000"]),
                ),
            )
        ],
        recommendation=DeliberationRecommendation(
            behavior_summary="Phone observed.",
            recommendation="Review required.",
            category="REVIEW_REQUIRED",
            confidence=0.8,
            reasoning="Phone in hand.",
            key_reasons=["Phone in hand"],
        ),
    )
    timeline = SimpleNamespace(
        artifact_registry=[
            SimpleNamespace(
                artifact_id="1700000060000__60000.webm",
                artifact_type="video",
                session_start_ms=0,
                session_end_ms=60_000,
                local_end_ms=60_000,
                sequence=0,
            )
        ]
    )
    bundle = assemble_evidence_bundle(
        EvidenceBundleInput(
            candidate_id="c1",
            assessment_id="a1",
            deliberation_bundle=deliberation,
            machine_facts_bundle=SimpleNamespace(facts=[]),
            perception_bundle=SimpleNamespace(
                covered_windows=1,
                total_windows=1,
                unknown_windows=0,
                windows=[],
            ),
            statistical_baseline=SimpleNamespace(
                coverage_metrics=SimpleNamespace(usable_window_ratio=0.9)
            ),
            master_timeline=timeline,
            track_b_findings=[
                SimpleNamespace(
                    eventType="phone_usage",
                    verdict="flagged",
                    timestampWindowMs=(10_000, 25_000),
                    attribution="candidate",
                    reasoning="Phone visible",
                )
            ],
        )
    )
    dumped = bundle.model_dump(by_alias=True)
    assert "trackBObservations" in dumped
    assert len(dumped["trackBObservations"]) == 1
    assert dumped["trackBObservations"][0]["clipRef"]["segments"][0]["seekToMs"] == 10_000
    assert "questionInsights" not in dumped
    assert "performanceEvidence" not in dumped
    assert len(dumped["mediaIndex"]) == 1


def test_build_media_index_skips_rrweb() -> None:
    timeline = SimpleNamespace(
        artifact_registry=[
            SimpleNamespace(
                artifact_id="1700000060000.json",
                artifact_type="rrweb",
                session_start_ms=0,
                session_end_ms=90_000,
                local_end_ms=90_000,
                sequence=0,
            ),
            SimpleNamespace(
                artifact_id="1700000060000__60000.webm",
                artifact_type="screenRecording",
                session_start_ms=0,
                session_end_ms=60_000,
                local_end_ms=60_000,
                sequence=0,
            ),
        ]
    )
    index = build_media_index(timeline)
    assert len(index) == 1
    assert index[0].evidence_type == "screen"
