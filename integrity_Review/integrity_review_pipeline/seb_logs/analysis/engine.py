"""SEB Log AI Analysis engine.

This module provides the AI analysis layer that interprets reduced
SEB log evidence using Gemini Pro.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Protocol

from pydantic import ValidationError

from ....prompts.seb_log_analysis import (
    SEB_LOG_ANALYSIS_MODEL,
    SEB_LOG_ANALYSIS_PROMPT_VERSION,
    SEB_LOG_ANALYSIS_SYSTEM_PROMPT,
    build_seb_log_analysis_user_prompt,
)
from ..contracts import ReducedSessionEvidence
from .contracts import (
    SebLogAiAnalysisResult,
    SebLogReviewResult,
    SebLogSessionAnalysis,
)
from .rules import filter_rejected_findings, validate_ai_response


STEP_SEB_LOG_ANALYSIS = "SEB_LOG_ANALYSIS"

SEB_LOG_ANALYSIS_CONFIG = {
    "temperature": 0,
    "maxOutputTokens": 16_384,
    "responseMimeType": "application/json",
    "thinkingConfig": {"thinkingBudget": 1_024},
}


class GeminiClient(Protocol):
    async def generate(
        self,
        *,
        model: str,
        contents: list[Any],
        config: dict[str, Any],
    ) -> dict[str, Any]: ...


class GeminiLimiter(Protocol):
    async def __aenter__(self) -> None: ...
    async def __aexit__(self, *args: Any) -> None: ...


class Logger(Protocol):
    def info(self, message: str, **fields: Any) -> None: ...
    def warning(self, message: str, **fields: Any) -> None: ...
    def error(self, message: str, **fields: Any) -> None: ...


class AiUsageLogger(Protocol):
    def record(
        self,
        *,
        step: str,
        model_name: str,
        model_version: str,
        status: str,
        latency_seconds: float,
        usage: dict[str, int] | None,
        extra_meta: dict[str, Any],
    ) -> None: ...


class SebLogAiDeps(Protocol):
    """Dependencies for SEB Log AI analysis."""

    gemini: GeminiClient
    limiter: GeminiLimiter
    logger: Logger
    ai_usage_logger: AiUsageLogger


def _parse_ai_response(raw_text: str) -> dict[str, Any] | None:
    """Parse the AI response JSON."""
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        text = raw_text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            return None


# The schema declares these as list[str], but the model sometimes answers with
# richer list[dict] entries such as {"signal": ..., "context": ...}. That is a
# shape deviation rather than missing information, so flatten it instead of
# discarding an otherwise complete analysis.
REVIEWER_SUMMARY_TEXT_LISTS = (
    "significantSecuritySignals",
    "significantTechnicalProblems",
    "configurationDeviations",
    "coverageLimitations",
    "eventsRequiringInvestigation",
)


def _entry_to_text(entry: Any) -> str:
    """Render one summary list entry as text, collapsing a dict to its values.

    Joining the values keeps both halves of the common {"signal", "context"}
    shape, so the signal id stays greppable for reviewers.
    """
    if isinstance(entry, dict):
        parts = [
            str(value).strip()
            for value in entry.values()
            if isinstance(value, (str, int, float)) and str(value).strip()
        ]
        return ": ".join(parts)
    return str(entry)


def _normalize_reviewer_summary(parsed: dict[str, Any]) -> list[str]:
    """Coerce reviewer-summary text lists in place, reporting what was changed."""
    summary = parsed.get("reviewerSummary")
    if not isinstance(summary, dict):
        return []

    coerced = []
    for field in REVIEWER_SUMMARY_TEXT_LISTS:
        entries = summary.get(field)
        if isinstance(entries, list) and any(
            not isinstance(entry, str) for entry in entries
        ):
            summary[field] = [_entry_to_text(entry) for entry in entries]
            coerced.append(field)
    return coerced


def _build_analysis_result(
    parsed: dict[str, Any],
    raw_text: str,
    validation_errors: list[str],
) -> SebLogAiAnalysisResult:
    """Build a SebLogAiAnalysisResult from parsed AI response."""
    return SebLogAiAnalysisResult(
        session_journey=parsed.get("sessionJourney", []),
        simplified_journey=parsed.get("simplifiedJourney", []),
        findings=parsed.get("findings", []),
        correlated_incidents=parsed.get("correlatedIncidents", []),
        technical_problems=parsed.get("technicalProblems", []),
        configuration_assessment=parsed.get("configurationAssessment"),
        coverage_assessment=parsed.get("coverageAssessment"),
        reviewer_summary=parsed.get("reviewerSummary"),
        raw_llm_response=raw_text,
        validation_errors=validation_errors,
        analysis_failed=False,
        failure_reason=None,
    )


def _build_failed_result(reason: str, raw_text: str | None = None) -> SebLogAiAnalysisResult:
    """Build a failed analysis result."""
    return SebLogAiAnalysisResult(
        raw_llm_response=raw_text,
        validation_errors=[],
        analysis_failed=True,
        failure_reason=reason,
    )


async def analyze_session(
    deps: SebLogAiDeps,
    evidence: ReducedSessionEvidence,
    *,
    max_retries: int = 2,
) -> SebLogAiAnalysisResult:
    """Analyze a single session's reduced evidence with AI.

    Args:
        deps: AI dependencies (gemini client, limiter, logger, usage logger)
        evidence: The reduced session evidence
        max_retries: Maximum retry attempts on parse or schema failure

    Returns:
        SebLogAiAnalysisResult with analysis or failure info
    """
    evidence_dict = evidence.model_dump(mode="json", by_alias=True)
    user_prompt = build_seb_log_analysis_user_prompt(reduced_evidence=evidence_dict)

    session_folder = evidence.session_ref.session_folder

    contents = [
        {"role": "user", "parts": [{"text": user_prompt}]},
    ]

    config = {
        **SEB_LOG_ANALYSIS_CONFIG,
        "systemInstruction": SEB_LOG_ANALYSIS_SYSTEM_PROMPT,
    }

    import time

    for attempt in range(max_retries + 1):
        start = time.perf_counter()
        try:
            async with deps.limiter:
                response = await deps.gemini.generate(
                    model=SEB_LOG_ANALYSIS_MODEL,
                    contents=contents,
                    config=config,
                )
            latency = time.perf_counter() - start

            deps.ai_usage_logger.record(
                step=STEP_SEB_LOG_ANALYSIS,
                model_name=SEB_LOG_ANALYSIS_MODEL,
                model_version=SEB_LOG_ANALYSIS_PROMPT_VERSION,
                status="SUCCESS",
                latency_seconds=latency,
                usage=response.get("usage"),
                extra_meta={"session_folder": session_folder, "attempt": attempt + 1},
            )

            raw_text = response.get("text", "")
            parsed = _parse_ai_response(raw_text)

            if parsed is None:
                if attempt < max_retries:
                    deps.logger.warning(
                        "seb-log-analysis: JSON parse failed, retrying",
                        session_folder=session_folder,
                        attempt=attempt + 1,
                    )
                    continue
                else:
                    deps.logger.error(
                        "seb-log-analysis: JSON parse failed after retries",
                        session_folder=session_folder,
                    )
                    return _build_failed_result("JSON parse failure", raw_text)

            validation = validate_ai_response(parsed, evidence_dict)

            if validation.rejected_findings:
                parsed = filter_rejected_findings(parsed, validation.rejected_findings)

            all_errors = validation.errors + validation.warnings

            if coerced := _normalize_reviewer_summary(parsed):
                deps.logger.warning(
                    "seb-log-analysis: coerced reviewer summary fields to text",
                    session_folder=session_folder,
                    fields=coerced,
                )

            # A parseable response that deviates from the schema is the same
            # class of problem as unparseable JSON, so it earns the same
            # retries rather than discarding a complete analysis outright.
            try:
                return _build_analysis_result(parsed, raw_text, all_errors)
            except ValidationError as error:
                if attempt < max_retries:
                    deps.logger.warning(
                        "seb-log-analysis: schema validation failed, retrying",
                        session_folder=session_folder,
                        attempt=attempt + 1,
                        error=str(error),
                    )
                    continue
                deps.logger.error(
                    "seb-log-analysis: schema validation failed after retries",
                    session_folder=session_folder,
                    error=str(error),
                )
                return _build_failed_result(
                    f"Schema validation failure: {error}", raw_text
                )

        except Exception as error:
            latency = time.perf_counter() - start
            deps.ai_usage_logger.record(
                step=STEP_SEB_LOG_ANALYSIS,
                model_name=SEB_LOG_ANALYSIS_MODEL,
                model_version=SEB_LOG_ANALYSIS_PROMPT_VERSION,
                status="FAILURE",
                latency_seconds=latency,
                usage=None,
                extra_meta={
                    "session_folder": session_folder,
                    "attempt": attempt + 1,
                    "error": str(error),
                },
            )
            if attempt < max_retries:
                deps.logger.warning(
                    "seb-log-analysis: AI call failed, retrying",
                    session_folder=session_folder,
                    attempt=attempt + 1,
                    error=str(error),
                )
                await asyncio.sleep(3 * (attempt + 1))
                continue

            deps.logger.error(
                "seb-log-analysis: AI call failed",
                session_folder=session_folder,
                error=str(error),
            )
            return _build_failed_result(f"AI call failed: {error}")

    return _build_failed_result("Exhausted retries")


async def run_seb_log_ai_analysis(
    deps: SebLogAiDeps,
    sessions: list[ReducedSessionEvidence],
    *,
    review_id: str,
) -> SebLogReviewResult:
    """Run AI analysis on multiple reduced sessions concurrently.

    Args:
        deps: AI dependencies
        sessions: List of reduced session evidence
        review_id: The review ID

    Returns:
        SebLogReviewResult with all session analyses
    """
    async def analyze_one(evidence: ReducedSessionEvidence) -> SebLogSessionAnalysis:
        ai_result = await analyze_session(deps, evidence)
        return SebLogSessionAnalysis(
            session_folder=evidence.session_ref.session_folder,
            org_assessment_id=evidence.session_ref.org_assessment_id,
            user_id=evidence.session_ref.user_id,
            reduction=evidence.model_dump(mode="json", by_alias=True),
            ai_analysis=ai_result,
        )

    session_analyses = await asyncio.gather(
        *(analyze_one(session) for session in sessions),
        return_exceptions=True,
    )

    result_sessions: list[SebLogSessionAnalysis] = []
    sessions_failed = 0

    for i, result in enumerate(session_analyses):
        if isinstance(result, Exception):
            deps.logger.error(
                "seb-log-analysis: Session analysis raised exception",
                session_index=i,
                error=str(result),
            )
            result_sessions.append(
                SebLogSessionAnalysis(
                    session_folder=sessions[i].session_ref.session_folder,
                    org_assessment_id=sessions[i].session_ref.org_assessment_id,
                    user_id=sessions[i].session_ref.user_id,
                    reduction=sessions[i].model_dump(mode="json", by_alias=True),
                    ai_analysis=_build_failed_result(f"Exception: {result}"),
                )
            )
            sessions_failed += 1
        else:
            result_sessions.append(result)
            if result.ai_analysis and result.ai_analysis.analysis_failed:
                sessions_failed += 1

    sessions_with_ai = sum(
        1 for s in result_sessions
        if s.ai_analysis and not s.ai_analysis.analysis_failed
    )

    return SebLogReviewResult(
        review_id=review_id,
        sessions=result_sessions,
        analysis_version=SEB_LOG_ANALYSIS_PROMPT_VERSION,
        total_sessions=len(sessions),
        sessions_with_ai=sessions_with_ai,
        sessions_failed=sessions_failed,
    )
