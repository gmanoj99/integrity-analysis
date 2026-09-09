"""Deterministic keystroke findings from MachineFacts."""

from __future__ import annotations

from ..machine_facts.contracts import MachineFactsBundle
from ..machine_facts.kinds import MachineFactKind
from ..machine_facts.paste_utils import is_reportable_paste
from .contracts import EvidenceFinding, EvidenceFindingsResult, KeystrokeProof

PRECURSOR_LOOKBACK_MS = 30_000
MASS_PASTE_GAP_MS = 60_000
RAPID_CPS_THRESHOLD = 30
RAPID_MIN_INSERTION_EVENTS = 3


def _mk(
    index: int,
    event_type: str,
    start_ms: int,
    end_ms: int,
    **extra: object,
) -> EvidenceFinding:
    payload = {
        "id": f"mf_ks_{index}",
        "source": "keystroke",
        "event_type": event_type,
        "timestamp_window_ms": (start_ms, end_ms),
        "attribution": "candidate",
        "severity": "medium",
        "evidence_ref": f"machine_facts:t={start_ms}-{end_ms}ms",
        "verdict": "flagged",
        "evidence_strength": "moderate",
        **extra,
    }
    return EvidenceFinding(**payload)


def derive_keystroke_findings(bundle: MachineFactsBundle) -> EvidenceFindingsResult:
    findings: list[EvidenceFinding] = []
    index = 0
    large_pastes = sorted(
        (fact for fact in bundle.facts if is_reportable_paste(fact)),
        key=lambda fact: fact.start_offset_ms,
    )

    def has_precursor(paste_start_ms: int) -> bool:
        return any(
            fact.start_offset_ms <= paste_start_ms
            and paste_start_ms - fact.start_offset_ms <= PRECURSOR_LOOKBACK_MS
            and fact.kind
            in {
                MachineFactKind.WINDOW_BLUR.value,
                MachineFactKind.TAB_SWITCH.value,
                MachineFactKind.FULLSCREEN_EXIT.value,
                MachineFactKind.ACTIVITY_GAP.value,
            }
            for fact in bundle.facts
        )

    clusters: list[list] = []
    for fact in large_pastes:
        qn = fact.detail.get("questionNumber")
        prev = clusters[-1] if clusters else None
        prev_last = prev[-1] if prev else None
        prev_qn = prev_last.detail.get("questionNumber") if prev_last else None
        same_question = qn == prev_qn if qn is not None and prev_qn is not None else qn is None and prev_qn is None
        within_gap = (
            prev_last is not None
            and fact.start_offset_ms - prev_last.start_offset_ms <= MASS_PASTE_GAP_MS
        )
        if prev and same_question and within_gap:
            prev.append(fact)
        else:
            clusters.append([fact])

    for cluster in clusters:
        first, last = cluster[0], cluster[-1]
        total_chars = sum(item.detail.get("charsAdded") or 0 for item in cluster)
        max_single = max(item.detail.get("charsAdded") or 0 for item in cluster)
        excerpts = [
            (item, item.detail.get("pastedExcerpt"))
            for item in cluster
            if item.detail.get("pastedExcerpt")
        ]
        best_excerpt = max(excerpts, key=lambda pair: pair[0].detail.get("charsAdded") or 0)[1] if excerpts else None
        qn = first.detail.get("questionNumber")
        proof = KeystrokeProof(
            inserted_char_count=total_chars,
            pasted_excerpt=best_excerpt,
        )
        if len(cluster) >= 3:
            corroborated = any(has_precursor(item.start_offset_ms) for item in cluster)
            findings.append(
                _mk(
                    index,
                    "mass_paste",
                    first.start_offset_ms,
                    last.start_offset_ms + 1000,
                    severity="high" if total_chars >= 500 and corroborated else "medium",
                    evidence_strength="strong"
                    if total_chars >= 500 and corroborated
                    else "moderate"
                    if corroborated
                    else "thin",
                    occurrence_count=len(cluster),
                    reasoning=(
                        f"{len(cluster)} large external paste event(s)"
                        f"{f' on question {qn}' if qn is not None else ''} "
                        f"(~{total_chars} chars total; largest {max_single})."
                    ),
                    keystroke_proof=proof,
                )
            )
            index += 1
        else:
            for fact in cluster:
                chars_added = fact.detail.get("charsAdded") or 0
                total = fact.detail.get("totalChars") or 0
                excerpt = fact.detail.get("pastedExcerpt")
                corroborated = has_precursor(fact.start_offset_ms)
                findings.append(
                    _mk(
                        index,
                        "external_paste",
                        fact.start_offset_ms,
                        fact.start_offset_ms + 1000,
                        severity="high"
                        if chars_added >= 200 and corroborated
                        else "medium",
                        evidence_strength="moderate"
                        if chars_added >= 200 and corroborated
                        else "thin",
                        reasoning=(
                            f"External paste of {chars_added} chars"
                            f"{f' (Q{qn})' if qn is not None else ''} "
                            f"— field reached {total} chars."
                        ),
                        keystroke_proof=KeystrokeProof(
                            inserted_char_count=chars_added,
                            final_value_length=total,
                            pasted_excerpt=excerpt,
                        ),
                    )
                )
                index += 1

    rapid_sessions = [
        fact
        for fact in bundle.facts
        if fact.kind == MachineFactKind.TYPING_STOPPED.value
        and (fact.detail.get("avgCharsPerSecond") or 0) > RAPID_CPS_THRESHOLD
    ]
    for fact in rapid_sessions:
        detail = fact.detail
        if detail.get("sectionType") == "MCQ":
            continue
        if (detail.get("insertionEvents") or 0) < RAPID_MIN_INSERTION_EVENTS:
            continue
        start_ms = detail.get("typingStartOffsetMs", fact.start_offset_ms)
        duration_ms = detail.get("durationMs") or 0
        cps = round(detail.get("avgCharsPerSecond") or 0)
        findings.append(
            _mk(
                index,
                "rapid_typing",
                start_ms,
                start_ms + duration_ms,
                severity="high",
                evidence_strength="thin",
                reasoning=(
                    f"Input area received text at ~{cps} chars/second over "
                    f"{round(duration_ms / 1000)}s — exceeds human typing speed."
                ),
                keystroke_proof=KeystrokeProof(
                    keystrokes_in_window=detail.get("insertionEvents"),
                    inserted_char_count=detail.get("finalCharCount"),
                ),
            )
        )
        index += 1

    obs_parts: list[str] = []
    if large_pastes:
        total = sum(item.detail.get("charsAdded") or 0 for item in large_pastes)
        obs_parts.append(
            f"{len(large_pastes)} large paste event(s) (≥50 chars each; ~{total} chars total)."
        )
    if rapid_sessions:
        obs_parts.append("Rapid input detected from machine facts.")
    if not obs_parts:
        obs_parts.append("No anomalous keystroke patterns detected from machine facts.")

    return EvidenceFindingsResult(
        findings=findings,
        keystroke_observation=" ".join(obs_parts),
        keystroke_available=True,
    )
