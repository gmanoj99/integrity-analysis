"""Paste origin refinement and reportability gates."""

from __future__ import annotations

import re

from .contracts import MachineFact
from .kinds import MachineFactKind

PASTE_DELETE_PAIR_GAP_MS = 5_000
PASTE_DELETE_SIZE_RATIO = 0.7
PASTED_EXCERPT_MAX_CHARS = 300
COPY_LOOKBACK_MS = 120_000


def compute_paste_excerpt(prev_text: str, new_text: str) -> str:
    if not new_text:
        return ""
    if new_text.startswith(prev_text):
        return new_text[
            len(prev_text) : len(prev_text) + PASTED_EXCERPT_MAX_CHARS
        ]
    shared_prefix = 0
    max_prefix = min(len(prev_text), len(new_text))
    while shared_prefix < max_prefix and prev_text[shared_prefix] == new_text[shared_prefix]:
        shared_prefix += 1
    shared_suffix = 0
    while (
        shared_suffix < len(prev_text) - shared_prefix
        and shared_suffix < len(new_text) - shared_prefix
        and prev_text[len(prev_text) - 1 - shared_suffix]
        == new_text[len(new_text) - 1 - shared_suffix]
    ):
        shared_suffix += 1
    start = shared_prefix
    end = max(start, len(new_text) - shared_suffix)
    return new_text[start:end][:PASTED_EXCERPT_MAX_CHARS]


def normalize_code_for_match(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def excerpt_matches_source(excerpt: str, source: str) -> bool:
    if not excerpt or len(excerpt) < 20 or not source:
        return False
    normalized_excerpt = normalize_code_for_match(excerpt)
    normalized_source = normalize_code_for_match(source)
    if len(normalized_excerpt) < 20 or len(normalized_source) < 20:
        return False
    if normalized_excerpt in normalized_source:
        return True
    if len(normalized_excerpt) >= 40:
        middle = normalized_excerpt[5:-5]
        if len(middle) >= 20 and middle in normalized_source:
            return True
    return False


def is_reportable_paste(fact: MachineFact, min_chars: int = 50) -> bool:
    if fact.kind not in (MachineFactKind.LARGE_PASTE.value, MachineFactKind.PASTE.value):
        return False
    detail = fact.detail
    if detail.get("pasteOrigin") == "internal":
        return False
    if detail.get("revertedEdit") is True:
        return False
    chars = detail.get("charsAdded") or detail.get("charCount") or 0
    if fact.kind == MachineFactKind.LARGE_PASTE.value and chars > 0 and chars < min_chars:
        return False
    return True


def refine_paste_origins(facts: list[MachineFact]) -> None:
    submissions = [
        fact for fact in facts if fact.kind == MachineFactKind.CODE_SUBMISSION.value
    ]
    starter_sources = [
        value
        for fact in submissions
        if isinstance((value := fact.detail.get("starterCode")), str) and value
    ]
    submission_sources = [
        value
        for fact in submissions
        if isinstance((value := fact.detail.get("sourceCode")), str) and value
    ]
    corpus_parts: list[str] = list(submission_sources)
    for fact in facts:
        if fact.kind not in (
            MachineFactKind.LARGE_PASTE.value,
            MachineFactKind.TEXT_CORRECTION.value,
        ):
            continue
        for key in ("prevText", "newText", "fieldText", "pastedExcerpt"):
            value = fact.detail.get(key)
            if isinstance(value, str) and len(value) >= 20:
                corpus_parts.append(value)
    session_corpus = "\n".join(corpus_parts)
    seen: list[str] = []
    for fact in sorted(
        (f for f in facts if f.kind == MachineFactKind.LARGE_PASTE.value),
        key=lambda item: item.start_offset_ms,
    ):
        detail = fact.detail
        if detail.get("pasteOrigin") == "internal":
            continue
        excerpt = detail.get("pastedExcerpt")
        if isinstance(excerpt, str):
            if any(excerpt_matches_source(excerpt, source) for source in starter_sources):
                detail["pasteOrigin"] = "internal"
                detail["pasteOriginReason"] = "starter_code_match"
            elif any(
                excerpt_matches_source(excerpt, source)
                for source in submission_sources
            ):
                detail["pasteOrigin"] = "internal"
                detail["pasteOriginReason"] = "own_submission_match"
        if detail.get("pasteOrigin") == "internal":
            if isinstance(excerpt, str) and len(excerpt) >= 20:
                seen.append(normalize_code_for_match(excerpt))
            continue
        if isinstance(excerpt, str) and len(excerpt) >= 20 and session_corpus:
            normalized = normalize_code_for_match(excerpt)
            if normalize_code_for_match(session_corpus).count(normalized) >= 2:
                detail["pasteOrigin"] = "internal"
                detail["pasteOriginReason"] = "in_session_corpus_match"
                seen.append(normalized)
                continue
        if isinstance(excerpt, str) and len(excerpt) >= 20:
            normalized = normalize_code_for_match(excerpt)
            if any(
                prev == normalized
                or (len(normalized) >= 40 and prev in normalized)
                or (len(prev) >= 40 and normalized in prev)
                for prev in seen
            ):
                detail["pasteOrigin"] = "internal"
                detail["pasteOriginReason"] = "duplicate_paste"
            else:
                seen.append(normalized)
        section_id = detail.get("sectionId")
        if detail.get("pasteOrigin") != "internal" and any(
            candidate.kind == MachineFactKind.COPY.value
            and candidate.start_offset_ms <= fact.start_offset_ms
            and fact.start_offset_ms - candidate.start_offset_ms <= COPY_LOOKBACK_MS
            and (
                section_id is None
                or candidate.detail.get("sectionId") in (None, section_id)
            )
            for candidate in facts
        ):
            detail["pasteOrigin"] = "internal"
            detail["pasteOriginReason"] = "preceding_copy"
        elif detail.get("pasteOrigin") != "internal":
            # Not matched to anything inside the session is not the same as
            # having come from outside it. The corpus only holds what the
            # recorder captured, so a candidate reusing her own code from a
            # place we did not capture lands here too. "unknown" keeps the
            # paste reportable while leaving the accusation unmade.
            detail["pasteOrigin"] = "unknown"


def mark_paste_delete_pairs(facts: list[MachineFact]) -> None:
    pastes = sorted(
        (f for f in facts if f.kind == MachineFactKind.LARGE_PASTE.value),
        key=lambda item: item.start_offset_ms,
    )
    corrections = sorted(
        (f for f in facts if f.kind == MachineFactKind.TEXT_CORRECTION.value),
        key=lambda item: item.start_offset_ms,
    )
    used: set[str] = set()
    for paste in pastes:
        detail = paste.detail
        element_id = detail.get("elementId")
        chars_added = detail.get("charsAdded") or 0
        if not element_id or chars_added <= 0:
            continue
        partner = next(
            (
                correction
                for correction in corrections
                if correction.id not in used
                and correction.detail.get("elementId") == element_id
                and correction.start_offset_ms >= paste.start_offset_ms
                and correction.start_offset_ms - paste.start_offset_ms
                <= PASTE_DELETE_PAIR_GAP_MS
                and (correction.detail.get("charsDeleted") or 0)
                >= chars_added * PASTE_DELETE_SIZE_RATIO
            ),
            None,
        )
        if partner is None:
            continue
        used.add(partner.id)
        detail["revertedEdit"] = True
        partner.detail["revertsPaste"] = True
        partner.detail["pairedPasteId"] = paste.id
        detail["pasteOrigin"] = "internal"
        detail["pasteOriginReason"] = "reverted_edit"
