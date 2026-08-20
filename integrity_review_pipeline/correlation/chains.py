"""Candidate chains and contribution deduplication."""

from __future__ import annotations

from .contracts import (
    CONTRIBUTION_DEDUP_BUCKET_MS,
    CandidateChain,
    ProvisionalLane,
    RiskContribution,
    TimelineEvent,
)
from .helpers import SPEECH_INTEGRITY_EVENT_TYPES


def dedupe_contributions(
    contributions: list[RiskContribution],
    bucket_ms: int = CONTRIBUTION_DEDUP_BUCKET_MS,
) -> list[RiskContribution]:
    best: dict[str, RiskContribution] = {}
    for contribution in contributions:
        q_key = (
            f"q{contribution.question_number}"
            if contribution.question_number is not None
            else (contribution.section_id or "session")
        )
        t0 = contribution.time_range_ms[0] or 0
        bucket = t0 // bucket_ms
        key = f"{contribution.factor_id}|{q_key}|{bucket}"
        existing = best.get(key)
        if (
            existing is None
            or contribution.score_contribution > existing.score_contribution
            or (
                contribution.score_contribution == existing.score_contribution
                and contribution.intensity > existing.intensity
            )
        ):
            best[key] = contribution
    return sorted(best.values(), key=lambda c: c.time_range_ms[0] or 0)


def _lane_from_contribution(contribution: RiskContribution) -> ProvisionalLane:
    if contribution.intensity >= 0.75 and not contribution.requires_corroboration:
        return "red"
    if contribution.intensity >= 0.5:
        return "amber"
    return "green"


def build_candidate_chains(
    contributions: list[RiskContribution],
    timeline: list[TimelineEvent],
) -> list[CandidateChain]:
    chain_factor_ids = {
        "interact_speech_input_correct",
        "speech_then_correct_hard",
        "speech_coaching_then_fast_correct",
        "second_person_then_paste_or_correct",
        "phone_fast_correct",
        "gaze_fast_correct",
        "gaze_off_then_speech_then_answer",
        "whisper_then_paste",
    }
    chains: list[CandidateChain] = []
    idx = 0
    for contribution in contributions:
        if contribution.factor_id not in chain_factor_ids:
            continue
        t0, t1 = contribution.time_range_ms
        overlapping = [
            e
            for e in timeline
            if e.t_ms <= t1 + 2_000 and (e.end_ms or e.t_ms) >= t0 - 2_000
        ]
        speech = next(
            (
                e
                for e in overlapping
                if e.source == "perception_episode"
                and (
                    e.kind in SPEECH_INTEGRITY_EVENT_TYPES
                    or e.kind in {"external_help", "multiple_faces"}
                )
            ),
            None,
        )
        audio_summary = speech.detail.get("conversationSummaryEn") if speech else None
        speech_class = speech.detail.get("speechContentClass") if speech else None
        role_cue = speech.detail.get("secondPersonLooksLike") if speech else None
        members: list[str] = []
        if speech:
            members.append(speech.kind)
        elif "phone" in contribution.factor_id:
            members.append("phone_visible")
        elif "gaze" in contribution.factor_id:
            members.append("gaze_off")
        members.append("input_on_question")
        if "correct" in contribution.rationale or "correct" in contribution.factor_id:
            members.append("correct")
        modalities = set()
        for event in overlapping:
            modalities.add("video" if event.source == "perception_episode" else "machine")
        idx += 1
        chains.append(
            CandidateChain(
                chain_id=f"chain_{idx}_{contribution.factor_id}",
                factor_id=contribution.factor_id,
                time_range_ms=(t0, t1),
                duration_ms=max(0, t1 - t0),
                question_number=contribution.question_number,
                question_id=contribution.question_id,
                section_id=contribution.section_id,
                members=members,
                modality_count=max(1, len(modalities)),
                outcome="correct" if "correct" in contribution.rationale else "unknown",
                audio_summary_en=str(audio_summary) if audio_summary else None,
                speech_content_class=str(speech_class) if speech_class else None,
                role_cue=str(role_cue) if role_cue else None,
                provisional_lane=_lane_from_contribution(contribution),
                evidence_refs=contribution.evidence_refs,
                rationale=contribution.rationale,
            )
        )
    return chains
