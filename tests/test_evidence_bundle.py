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
from integrity_review_pipeline.evidence_bundle.section_builders import (
    build_behavior_summary,
    build_recommendation_section,
    resolve_section_id,
)
from integrity_review_pipeline.evidence_bundle.service import (
    EvidenceBundleInput,
    assemble_evidence_bundle,
    build_media_index,
    resolve_offset_clip_ref,
)
from integrity_review_pipeline.contracts.evidence_bundle import ClipRef, ClipSegment, MediaIndexEntry


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
                section_id="section-2",
                session_start_ms=0,
                session_end_ms=60_000,
                local_end_ms=60_000,
                sequence=0,
            )
        ],
        sections=[],
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
    assert dumped["mediaIndex"][0]["sectionId"] == "section-2"
    assert "finalConfidence" not in dumped["confidence"]
    assert "corroborationDowngradeApplied" not in dumped["confidence"]
    assert "totalValidated" not in dumped["detectedSignals"]
    assert "totalStories" not in dumped["integrityStories"]
    assert "totalEpisodes" not in dumped["episodeAnalysis"]
    assert "totalEntries" not in dumped["evidenceTimeline"]
    assert "deliberationCompositeHash" not in dumped["provenance"]
    assert "durationMs" not in dumped["trackBObservations"][0]["clipRef"]
    assert "seekToMs" not in dumped["trackBObservations"][0]["clipRef"]
    assert dumped["integrityStories"]["stories"][0]["sectionId"] == "section-2"
    assert dumped["detectedSignals"]["signals"][0]["sectionId"] == "section-2"
    assert dumped["trackBObservations"][0]["sectionId"] == "section-2"


def test_build_media_index_skips_rrweb() -> None:
    timeline = SimpleNamespace(
        artifact_registry=[
            SimpleNamespace(
                artifact_id="1700000060000.json",
                artifact_type="rrweb",
                section_id="section-1",
                session_start_ms=0,
                session_end_ms=90_000,
                local_end_ms=90_000,
                sequence=0,
            ),
            SimpleNamespace(
                artifact_id="1700000060000__60000.webm",
                artifact_type="screenRecording",
                section_id="section-2",
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
    assert index[0].section_id == "section-2"


def test_resolve_section_id_prefers_anchoring_chunk() -> None:
    media_index = [
        MediaIndexEntry(
            chunk_id="video-section-2-1700000060000__60000",
            evidence_type="video",
            section_id="section-2",
            session_start_ms=0,
            session_end_ms=60_000,
            duration_ms=60_000,
            sequence=0,
        ),
        MediaIndexEntry(
            chunk_id="video-section-7-1700000120000__60000",
            evidence_type="video",
            section_id="section-7",
            session_start_ms=60_000,
            session_end_ms=120_000,
            duration_ms=60_000,
            sequence=1,
        ),
    ]
    clip_ref = ClipRef(
        segments=[
            ClipSegment(
                chunk_id="video-section-2-1700000060000__60000",
                evidence_type="video",
                seek_to_ms=10_000,
            )
        ],
        clip_start_ms=10_000,
        clip_end_ms=20_000,
        cache_key="abc123",
    )
    assert (
        resolve_section_id(clip_ref, (10_000, 20_000), media_index, sections=[]) == "section-2"
    )


def test_resolve_section_id_falls_back_to_time_bracket_without_clip() -> None:
    media_index = [
        MediaIndexEntry(
            chunk_id="video-section-7-1700000120000__60000",
            evidence_type="video",
            section_id="section-7",
            session_start_ms=60_000,
            session_end_ms=120_000,
            duration_ms=60_000,
            sequence=0,
        )
    ]
    assert resolve_section_id(None, (70_000, 80_000), media_index, sections=[]) == "section-7"


def test_resolve_section_id_falls_back_to_canonical_sections_for_rrweb_only() -> None:
    # No video/screen chunks in media_index (e.g. rrweb-only signal); must use
    # the canonical session sections as the last-resort resolution source.
    sections = [
        SimpleNamespace(section_id="section-1", start_ms=0, end_ms=100_000),
        SimpleNamespace(section_id="section-2", start_ms=100_000, end_ms=200_000),
    ]
    assert resolve_section_id(None, (120_000, 130_000), [], sections=sections) == "section-2"


def test_resolve_section_id_returns_none_when_unresolvable() -> None:
    assert resolve_section_id(None, (0, 0), [], sections=[]) is None


def test_behavior_summary_survives_track_b_only_review() -> None:
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
        validated_signals=[],
        recommendation=DeliberationRecommendation(
            behavior_summary="No significant integrity concerns were identified.",
            recommendation="No action required.",
            category="REVIEW_REQUIRED",
            confidence=0.8,
            reasoning="No integrity concerns were identified.",
            supporting_signals=["tb-phone"],
        ),
    )
    summary = build_behavior_summary(deliberation)
    assert "Independent evidence findings" in summary.text
    section = build_recommendation_section(deliberation)
    assert section.category == "REVIEW_REQUIRED"
    assert "No action required" not in section.recommendation
    assert section.supporting_signal_ids == ["tb-phone"]

