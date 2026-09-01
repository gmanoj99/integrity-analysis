"""Pipeline orchestrator for SEB log reduction.

This module provides the main entry point for running the SEB log
reduction pipeline across multiple sessions.
"""

from __future__ import annotations

from collections.abc import Iterator

from .contracts import ReducedSessionEvidence, SebLogSessionRef
from .deps import SebLogPipelineDeps
from .discovery import discover_sessions
from .reduction.session_reducer import SessionReducer


def run_seb_log_reduction(
    deps: SebLogPipelineDeps,
    bucket_prefix: str,
    org_assessment_id: str | None = None,
    user_id: str | None = None,
) -> Iterator[ReducedSessionEvidence]:
    """Run the SEB log reduction pipeline.

    This is the main entry point for Process 2 (Deterministic Analysis).
    It discovers sessions under the given S3 prefix and reduces each one.

    Args:
        deps: Pipeline dependencies (object store, logger)
        bucket_prefix: S3 prefix to search for sessions
        org_assessment_id: Optional filter for specific org assessment
        user_id: Optional filter for specific user

    Yields:
        ReducedSessionEvidence for each discovered session
    """
    deps.logger.info(
        "Starting SEB log reduction pipeline",
        bucket_prefix=bucket_prefix,
        org_filter=org_assessment_id,
        user_filter=user_id,
    )

    reducer = SessionReducer(deps)

    session_count = 0
    for session_ref in discover_sessions(
        deps, bucket_prefix, org_assessment_id, user_id
    ):
        session_count += 1
        deps.logger.info(
            "Processing session",
            session_number=session_count,
            org_assessment_id=session_ref.org_assessment_id,
            user_id=session_ref.user_id,
            session_folder=session_ref.session_folder,
            files_present=[f.value for f in session_ref.files_present],
            files_missing=[f.value for f in session_ref.files_missing],
        )

        try:
            evidence = reducer.reduce_session(session_ref)
            deps.logger.info(
                "Session reduced successfully",
                session_folder=session_ref.session_folder,
                signals_count=len(evidence.signals),
                noise_categories=len(evidence.noise_summary),
                bulk_terminations=len(evidence.bulk_terminations),
            )
            yield evidence
        except Exception as e:
            deps.logger.error(
                "Failed to reduce session",
                session_folder=session_ref.session_folder,
                error=str(e),
            )
            raise

    deps.logger.info(
        "SEB log reduction pipeline complete",
        total_sessions=session_count,
    )


def reduce_single_session(
    deps: SebLogPipelineDeps,
    session_ref: SebLogSessionRef,
) -> ReducedSessionEvidence:
    """Reduce a single session.

    Useful for testing or when the session reference is already known.

    Args:
        deps: Pipeline dependencies
        session_ref: Reference to the session to reduce

    Returns:
        ReducedSessionEvidence for the session
    """
    reducer = SessionReducer(deps)
    return reducer.reduce_session(session_ref)
