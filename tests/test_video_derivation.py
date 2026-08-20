"""Tests ported from idealReviewEvidenceUx.test.ts (buildEpisodes)."""

from integrity_review_pipeline.findings.video_derivation import DERIVATION_CONFIG, build_episodes
from tests.fixtures.perception import make_observation, make_perception_bundle


def test_build_episodes_merges_within_gap_including_flicker() -> None:
    obs = [
        make_observation("w1", start_ms=0, end_ms=10_000, objects={"phone_visible": "yes"}),
        make_observation("w2", start_ms=10_000, end_ms=15_000, objects={"phone_visible": "UNKNOWN"}),
        make_observation("w3", start_ms=15_000, end_ms=25_000, objects={"phone_visible": "yes"}),
    ]
    episodes = build_episodes(obs, lambda o: o.objects.phone_visible == "yes")
    assert len(episodes) == 1
    assert episodes[0].window_ids == ["w1", "w3"]


def test_build_episodes_splits_on_large_gap() -> None:
    gap = DERIVATION_CONFIG["EPISODE_MAX_GAP_SEC"] * 1000 + 1_000
    obs = [
        make_observation("w1", start_ms=0, end_ms=10_000, objects={"phone_visible": "yes"}),
        make_observation("w2", start_ms=10_000 + gap, end_ms=20_000 + gap, objects={"phone_visible": "yes"}),
    ]
    assert len(build_episodes(obs, lambda o: o.objects.phone_visible == "yes")) == 2


def test_audio_wearable_clears_without_contact() -> None:
    from integrity_review_pipeline.findings.video_derivation import derive_video_findings_from_perception

    obs = [
        make_observation(
            "w_0_20000",
            start_ms=0,
            objects={"earphone_visible": "yes", "earphone_link": "wireless"},
            audio={"speech_present": "no"},
        ),
        make_observation(
            "w_15000_35000",
            start_ms=15_000,
            objects={"earphone_visible": "yes", "earphone_link": "wireless"},
            audio={"speech_present": "no"},
        ),
    ]
    result = derive_video_findings_from_perception(make_perception_bundle(obs))
    flagged = [f for f in result.findings if f.verdict == "flagged" and f.event_type == "audio_wearable_assisted_comms"]
    cleared = [f for f in result.findings if f.verdict == "cleared" and f.event_type == "audio_wearable_assisted_comms"]
    assert not flagged
    assert cleared
    assert "not contacting" in cleared[0].reasoning.lower()
