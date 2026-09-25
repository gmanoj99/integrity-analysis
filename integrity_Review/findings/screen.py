from __future__ import annotations

import re
import uuid

from ..machine_facts.contracts import MachineFact
from ..machine_facts.kinds import MachineFactKind
from .contracts import (
    EvidenceFinding,
    EvidenceFindingsResult,
    ScreenObservation,
    ScreenPerceptionBundle,
    ScreenProof,
)

GAP_MERGE_MS = 90_000


def _parse_question_number(ref: str | None) -> int | None:
    if not ref:
        return None
    match = re.search(r"(?:q(?:uestion)?\s*[#:]?\s*)(\d+)", ref, re.IGNORECASE) or re.search(
        r"\b(\d{1,3})\b", ref
    )
    if not match:
        return None
    value = int(match.group(1))
    return value if 0 < value < 500 else None


def _screen_proof(
    obs: ScreenObservation,
    *,
    pasted_excerpt: str | None = None,
    question_number: int | None = None,
) -> ScreenProof:
    local_offset = max(0, round((obs.end_ms - obs.start_ms) / 2))
    qn = question_number or _parse_question_number(obs.visible_question_ref)
    return ScreenProof(
        chunk_id=obs.chunk_id,
        approx_frame_timestamp_ms=obs.start_ms + local_offset,
        local_offset_ms=local_offset,
        clip_label=f"screen seq {obs.sequence}",
        pasted_excerpt=pasted_excerpt or obs.pasted_text_excerpt,
        question_number=qn,
        visible_question_ref=obs.visible_question_ref,
    )


def _coalesce(
    observations: list[ScreenObservation],
    predicate,
) -> list[tuple[ScreenObservation, ScreenObservation, list[ScreenObservation]]]:
    hits = sorted((obs for obs in observations if predicate(obs)), key=lambda obs: obs.start_ms)
    episodes: list[tuple[ScreenObservation, ScreenObservation, list[ScreenObservation]]] = []
    for obs in hits:
        if episodes and obs.start_ms - episodes[-1][1].end_ms <= GAP_MERGE_MS:
            start, _, members = episodes[-1]
            members.append(obs)
            episodes[-1] = (start, obs, members)
        else:
            episodes.append((obs, obs, [obs]))
    return episodes


def _fact_from_obs(
    kind: MachineFactKind, obs: ScreenObservation, detail: dict
) -> MachineFact:
    offset = obs.start_ms + max(0, round((obs.end_ms - obs.start_ms) / 2))
    return MachineFact(
        id=str(uuid.uuid4()),
        start_offset_ms=offset,
        end_offset_ms=offset,
        raw_timestamp_ms=None,
        kind=kind.value,
        source="video_cv",
        evidence_source="screen",
        attribution="candidate",
        confidence=obs.confidence,
        detail={"chunkId": obs.chunk_id, **detail},
    )


EXTERNAL_SOURCE_PAD_MS = 60_000

NON_EXAM_FOREGROUND = frozenset({"browser", "notes", "ai_chat", "messaging"})


def _external_source_near(
    observations: list[ScreenObservation], start_ms: int, end_ms: int
) -> bool:
    lo = start_ms - EXTERNAL_SOURCE_PAD_MS
    hi = end_ms + EXTERNAL_SOURCE_PAD_MS
    for obs in observations:
        if obs.end_ms < lo or obs.start_ms > hi:
            continue
        if obs.external_resource_labels:
            return True
        if obs.ai_assistant_ui_visible == "yes":
            return True
        if obs.secondary_workspace_visible == "yes":
            return True
        if obs.exam_ui_visible == "no" and obs.foreground_app_class in NON_EXAM_FOREGROUND:
            return True
    return False


def derive_screen_findings(
    bundle: ScreenPerceptionBundle,
) -> tuple[EvidenceFindingsResult, list[MachineFact]]:
    observations = [obs for obs in bundle.observations if obs.video_available]
    findings: list[EvidenceFinding] = []
    synthetic_facts: list[MachineFact] = []
    index = 0

    for event_type, predicate, kind, detail_fn in (
        (
            "external_resource_open",
            lambda obs: bool(obs.external_resource_labels),
            MachineFactKind.SCREEN_EXTERNAL_RESOURCE,
            lambda obs: {"labels": obs.external_resource_labels},
        ),
        (
            "ai_assistant_ui_visible",
            lambda obs: obs.ai_assistant_ui_visible == "yes",
            MachineFactKind.SCREEN_AI_ASSISTANT_UI,
            lambda _obs: {},
        ),
        (
            "secondary_workspace_visible",
            lambda obs: obs.secondary_workspace_visible == "yes",
            MachineFactKind.SCREEN_SECONDARY_WORKSPACE,
            lambda _obs: {},
        ),
        (
            "fullscreen_exam_lost",
            lambda obs: obs.fullscreen_exam_likely == "no" and obs.exam_ui_visible == "no",
            MachineFactKind.SCREEN_FULLSCREEN_LOST,
            lambda _obs: {},
        ),
        (
            "screen_external_paste",
            lambda obs: obs.paste_cue_visible == "yes",
            MachineFactKind.SCREEN_EXTERNAL_PASTE,
            lambda obs: {
                "pastedExcerpt": obs.pasted_text_excerpt,
                "questionNumber": _parse_question_number(obs.visible_question_ref),
            },
        ),
    ):
        for start, end, members in _coalesce(observations, predicate):
            peak = max(members, key=lambda obs: obs.confidence)
            excerpt = peak.pasted_text_excerpt
            qn = _parse_question_number(peak.visible_question_ref)
            emit_type, emit_kind = event_type, kind
            verdict = "flagged"
            severity = "medium"
            reasoning = f"Screen episode: {event_type.replace('_', ' ')}."
            if event_type == "screen_external_paste" and not _external_source_near(
                observations, start.start_ms, end.end_ms
            ):
                emit_type, emit_kind = "screen_paste", MachineFactKind.SCREEN_PASTE
                verdict = "provisional"
                severity = "low"
                reasoning = (
                    "Paste seen on screen with no external source visible around it; "
                    "origin not established."
                )
            findings.append(
                EvidenceFinding(
                    id=f"scr_{index}",
                    source="screen",
                    event_type=emit_type,
                    timestamp_window_ms=(start.start_ms, end.end_ms),
                    attribution="candidate",
                    severity=severity,
                    evidence_ref=f"screen:{start.chunk_id}",
                    verdict=verdict,
                    evidence_strength="moderate" if verdict == "flagged" else "thin",
                    occurrence_count=len(members),
                    reasoning=reasoning,
                    screen_proof=_screen_proof(
                        peak,
                        pasted_excerpt=excerpt,
                        question_number=qn,
                    ),
                    keystroke_proof=None,
                )
            )
            synthetic_facts.append(_fact_from_obs(emit_kind, peak, detail_fn(peak)))
            index += 1

    parts: list[str] = []
    if any(finding.event_type == "external_resource_open" for finding in findings):
        parts.append("External websites/apps appeared on screen.")
    if any(finding.event_type == "screen_external_paste" for finding in findings):
        parts.append("On-screen paste activity with recoverable text.")
    if not parts:
        parts.append(
            "No flagged screen activity in the recorded portion."
            if observations
            else "Screen recording not available."
        )

    return (
        EvidenceFindingsResult(
            findings=findings,
            screen_observation=" ".join(parts),
            screen_available=bool(observations),
        ),
        synthetic_facts,
    )
