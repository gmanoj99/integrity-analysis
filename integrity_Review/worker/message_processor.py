"""Idempotent single-message processing: parse, dedupe, run, publish, ack.

Ordering invariant: the staged request message is deleted only *after* the
result has been durably published (object written + SQS message sent). If
publishing fails, the message is left in place for SQS to redeliver.

Key design: the combined result object is the source of truth for what still
needs work. The request carries no per-analysis flags; instead a branch whose
payload key is already present is never recomputed, and a branch recorded under
``unavailableAnalysis`` is impossible for this attempt and is never retried.

Terminal-vs-retryable policy: every path that ends this review must publish
either SUCCESS or FAILURE before the message is deleted, because the backend
leaves the review IN_PROGRESS until a callback arrives. A condition that
redelivery cannot fix (missing or malformed staged payload, a review_id
mismatch, the final delivery attempt) is therefore reported as FAILURE and
acked, and only genuinely transient errors are left for redelivery.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import ValidationError

from ..adapters.memory_cache import InMemoryPerceptionCache
from ..adapters.s3_object_store import ObjectNotFoundError
from ..deps import PipelineDeps
from ..integrity_review_pipeline.seb_logs.analysis.engine import run_seb_log_ai_analysis
from ..integrity_review_pipeline.seb_logs.deps import SebLogAiDeps, SebLogPipelineDeps
from ..integrity_review_pipeline.seb_logs.pipeline import run_seb_log_reduction
from ..pipeline import run_integrity_review
from .contracts import SqsRequestEnvelope, StagedReviewPayload, build_review_request
from .fair_limiter import FairGeminiLimiter
from .task_protection import TaskProtection
from .visibility_heartbeat import VisibilityHeartbeat

REQUEST_MESSAGE_TYPE = "AI_ANALYSIS_REQUEST"
RESULT_MESSAGE_TYPE = "AI_ANALYSIS_RESPONSE"

REVIEW_STATUS_SUCCESS = "SUCCESS"
REVIEW_STATUS_FAILURE = "FAILURE"
# The backend validates this against IntegrityReviewFailureReasonEnum, which
# only accepts STALE / ENQUEUE_FAILED / ANALYSIS_FAILED — the cause goes in the
# logs, not in this field.
FAILURE_REASON_ANALYSIS_FAILED = "ANALYSIS_FAILED"

COMBINED_RESULT_KEY_TEMPLATE = (
    "{s3_media_prefix}/media/ai_integrity_review_results/"
    "{org_assess_id}/{attempt_user_id}/evidence_bundle.json"
)

# Discovery appends the org/user segments itself, so the prefix stops at the root.
SEB_LOG_PREFIX_TEMPLATE = "{s3_media_prefix}/media/tsb_logs/"

PUBLISH_ATTEMPTS = 3
PUBLISH_RETRY_DELAY_SECONDS = 2.0

BRANCH_VIDEO = "video"
BRANCH_SEB_LOG = "sebLog"

UNAVAILABLE_KEY = "unavailableAnalysis"

# Insertion order also fixes the key order of the written object.
PAYLOAD_KEY_BY_BRANCH = {
    BRANCH_VIDEO: "videoAnalysis",
    BRANCH_SEB_LOG: "sebLogAnalysis",
}

OUTCOME_SUCCESS = "SUCCESS"
OUTCOME_RETRYABLE = "RETRYABLE_FAILURE"
OUTCOME_UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True, slots=True)
class BranchOutcome:
    """One branch's result, before it is merged into the combined result.

    ``reason`` is log-only for retryable failures; for unavailable branches it is
    persisted so later deliveries can tell the work apart from work not yet done.
    """

    kind: str
    payload: dict[str, Any] | None = None
    reason: str | None = None

    @classmethod
    def success(cls, payload: dict[str, Any]) -> BranchOutcome:
        return cls(OUTCOME_SUCCESS, payload=payload)

    @classmethod
    def retryable(cls, reason: str) -> BranchOutcome:
        return cls(OUTCOME_RETRYABLE, reason=reason)

    @classmethod
    def unavailable(cls, reason: str) -> BranchOutcome:
        return cls(OUTCOME_UNAVAILABLE, reason=reason)


@dataclass(slots=True)
class IntegrityReviewResult:
    """Combined result for both branches, written once to S3."""

    review_id: str
    analyses: dict[str, dict[str, Any]] = field(default_factory=dict)
    unavailable: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_existing(
        cls, data: Mapping[str, Any], *, review_id: str
    ) -> IntegrityReviewResult:
        """Read back a previously written object, keeping only usable entries."""
        analyses: dict[str, dict[str, Any]] = {}
        for branch, payload_key in PAYLOAD_KEY_BY_BRANCH.items():
            payload = data.get(payload_key)
            if isinstance(payload, dict) and payload:
                analyses[branch] = payload

        unavailable: dict[str, str] = {}
        recorded = data.get(UNAVAILABLE_KEY)
        if isinstance(recorded, Mapping):
            for branch in PAYLOAD_KEY_BY_BRANCH:
                reason = recorded.get(branch)
                if isinstance(reason, str) and reason:
                    unavailable[branch] = reason

        return cls(review_id=review_id, analyses=analyses, unavailable=unavailable)

    def pending_branches(self) -> list[str]:
        return [
            branch
            for branch in PAYLOAD_KEY_BY_BRANCH
            if branch not in self.analyses and branch not in self.unavailable
        ]

    def merge(self, branch: str, outcome: BranchOutcome) -> None:
        """Fold one branch's outcome in, leaving other branches untouched."""
        if outcome.kind == OUTCOME_SUCCESS and outcome.payload is not None:
            self.analyses[branch] = outcome.payload
            self.unavailable.pop(branch, None)
        elif outcome.kind == OUTCOME_UNAVAILABLE:
            self.unavailable[branch] = outcome.reason or OUTCOME_UNAVAILABLE

    @property
    def status(self) -> str:
        resolved = all(
            branch in self.analyses or branch in self.unavailable
            for branch in PAYLOAD_KEY_BY_BRANCH
        )
        return REVIEW_STATUS_SUCCESS if resolved else REVIEW_STATUS_FAILURE

    @property
    def failure_reason(self) -> str | None:
        if self.status == REVIEW_STATUS_SUCCESS:
            return None
        return FAILURE_REASON_ANALYSIS_FAILED

    def to_json(self) -> bytes:
        result: dict[str, Any] = {"reviewId": self.review_id, "status": self.status}
        for branch, payload_key in PAYLOAD_KEY_BY_BRANCH.items():
            payload = self.analyses.get(branch)
            if payload is not None:
                result[payload_key] = payload
        if self.unavailable:
            result[UNAVAILABLE_KEY] = dict(self.unavailable)
        return json.dumps(result, default=str).encode("utf-8")


class ObjectStoreWithHead(Protocol):
    async def get_json(self, ref: str) -> Any: ...

    async def head_object(self, ref: str) -> bool: ...

    async def put_object(self, ref: str, body: bytes, *, content_type: str = "application/json") -> None: ...

    async def media_uri_for(self, chunk: Any) -> str: ...

    async def get_bytes(self, ref: str) -> bytes: ...


class WorkerSqsClient(Protocol):
    async def delete(self, receipt_handle: str) -> None: ...

    async def change_visibility(self, receipt_handle: str, visibility_timeout: int) -> None: ...

    async def send(self, body: str) -> Any: ...


class LoggerFactory(Protocol):
    def __call__(self, review_id: str) -> Any: ...


class AiUsageLoggerFactory(Protocol):
    def __call__(
        self, *, review_id: str, org_assess_id: str, attempt_user_id: str
    ) -> Any: ...


class SebLogObjectStore(Protocol):
    def list_keys(self, prefix: str) -> Any: ...

    def iter_lines(self, key: str) -> Any: ...

    def get_json(self, key: str) -> dict[str, Any]: ...

    def get_object_size(self, key: str) -> int: ...


@dataclass(frozen=True, slots=True)
class WorkerContext:
    s3_media_prefix: str
    organization_id: str
    media_store: ObjectStoreWithHead
    request_store: ObjectStoreWithHead
    result_store: ObjectStoreWithHead
    request_queue: WorkerSqsClient
    response_queue: WorkerSqsClient
    gemini: Any
    limiter: FairGeminiLimiter
    task_protection: TaskProtection
    make_logger: LoggerFactory
    make_ai_usage_logger: AiUsageLoggerFactory
    heartbeat_interval_seconds: int
    visibility_timeout_seconds: int
    seb_log_object_store: SebLogObjectStore
    max_receive_count: int = 5


class TerminalPayloadError(Exception):
    """A staged request that redelivery can never make processable."""

    def __init__(self, cause: str, detail: str) -> None:
        super().__init__(f"{cause}: {detail}")
        self.cause = cause
        self.detail = detail


@dataclass(frozen=True, slots=True)
class MessageMeta:
    """The queue-message facts worth logging for every delivery."""

    message_id: str | None
    receive_count: int
    sent_timestamp_ms: int | None

    @classmethod
    def from_message(cls, message: dict[str, Any]) -> MessageMeta:
        attributes = message.get("Attributes") or {}
        return cls(
            message_id=message.get("MessageId"),
            receive_count=_int_or_default(attributes.get("ApproximateReceiveCount"), 1),
            sent_timestamp_ms=_int_or_default(attributes.get("SentTimestamp"), None),
        )

    def as_log_fields(self) -> dict[str, Any]:
        fields: dict[str, Any] = {
            "message_id": self.message_id,
            "receive_count": self.receive_count,
        }
        if self.sent_timestamp_ms is not None:
            fields["queue_age_seconds"] = round(
                max(0.0, time.time() - self.sent_timestamp_ms / 1000), 1
            )
        return fields


def _int_or_default(value: Any, default: int | None) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def seb_log_s3_prefix(*, s3_media_prefix: str) -> str:
    return SEB_LOG_PREFIX_TEMPLATE.format(s3_media_prefix=s3_media_prefix)


def combined_result_object_key(
    *, s3_media_prefix: str, org_assess_id: str, attempt_user_id: str
) -> str:
    return COMBINED_RESULT_KEY_TEMPLATE.format(
        s3_media_prefix=s3_media_prefix,
        org_assess_id=org_assess_id,
        attempt_user_id=attempt_user_id,
    )


async def _load_existing_result(
    ctx: WorkerContext,
    result_key: str,
    *,
    review_id: str,
    logger: Any,
) -> IntegrityReviewResult | None:
    """Load the combined result so completed branches can be reused.

    Returns an empty result when there is nothing usable to reuse, and ``None``
    when an existing object could not be read. ``None`` leaves the message for
    redelivery rather than overwriting analyses that may already have succeeded.
    """
    try:
        exists = await ctx.result_store.head_object(result_key)
    except Exception as error:  # noqa: BLE001 - unknown state, retain the message
        logger.error(
            "message-processor: existence check failed, retaining message",
            result_key=result_key,
            error=error,
            exc_info=True,
        )
        return None

    if not exists:
        return IntegrityReviewResult(review_id=review_id)

    try:
        data = await ctx.result_store.get_json(result_key)
    except Exception as error:  # noqa: BLE001 - object is there but unreadable
        logger.error(
            "message-processor: existing result unreadable, retaining message",
            result_key=result_key,
            error=error,
            exc_info=True,
        )
        return None

    if not isinstance(data, Mapping) or not data:
        logger.warning(
            "message-processor: existing result empty, running all branches",
            result_key=result_key,
        )
        return IntegrityReviewResult(review_id=review_id)

    return IntegrityReviewResult.from_existing(data, review_id=review_id)


async def _publish_result(
    ctx: WorkerContext,
    *,
    review_id: str,
    review_status: str,
    failure_reason: str | None,
    logger: Any,
) -> None:
    """Send the callback, retrying briefly so one flaky SQS call does not force
    a full redelivery (and, on the success path, a re-run of the pipeline)."""

    body = json.dumps(
        {
            "message_type": RESULT_MESSAGE_TYPE,
            "review_id": review_id,
            "review_status": review_status,
            "failure_reason": failure_reason,
        }
    )
    last_error: Exception | None = None
    for attempt in range(1, PUBLISH_ATTEMPTS + 1):
        try:
            message_id = await ctx.response_queue.send(body)
        except Exception as error:  # noqa: BLE001 - retried below, re-raised at the end
            last_error = error
            logger.warning(
                "message-processor: response publish attempt failed",
                review_status=review_status,
                attempt=attempt,
                attempts=PUBLISH_ATTEMPTS,
                error=error,
            )
            if attempt < PUBLISH_ATTEMPTS:
                await asyncio.sleep(PUBLISH_RETRY_DELAY_SECONDS * attempt)
            continue
        logger.info(
            "message-processor: response published",
            review_status=review_status,
            failure_reason=failure_reason,
            response_message_id=message_id,
        )
        return
    raise last_error if last_error is not None else RuntimeError("response publish failed")


async def _publish_combined_result(
    ctx: WorkerContext, combined: IntegrityReviewResult, logger: Any
) -> None:
    await _publish_result(
        ctx,
        review_id=combined.review_id,
        review_status=combined.status,
        failure_reason=combined.failure_reason,
        logger=logger,
    )


async def _fail_and_ack(
    ctx: WorkerContext,
    *,
    review_id: str,
    receipt_handle: str,
    logger: Any,
    cause: str,
    detail: str,
) -> None:
    """Report FAILURE for a review that cannot succeed, then ack the message.

    If the callback itself cannot be sent, the message is retained so a later
    delivery can retry it — acking here would strand the review IN_PROGRESS.
    """

    logger.error(
        "message-processor: review terminally failed",
        cause=cause,
        detail=detail,
        failure_reason=FAILURE_REASON_ANALYSIS_FAILED,
    )
    try:
        await _publish_result(
            ctx,
            review_id=review_id,
            review_status=REVIEW_STATUS_FAILURE,
            failure_reason=FAILURE_REASON_ANALYSIS_FAILED,
            logger=logger,
        )
    except Exception as publish_error:  # noqa: BLE001
        logger.error(
            "message-processor: failure publish failed, retaining message",
            cause=cause,
            error=publish_error,
        )
        return
    await ctx.request_queue.delete(receipt_handle)


async def _run_video_analysis_branch(
    ctx: WorkerContext,
    payload: StagedReviewPayload,
    logger: Any,
) -> BranchOutcome:
    """Run video/audio analysis and return its outcome in memory.

    An attempt the backend did not ask for is unavailable rather than
    retryable: no amount of redelivery makes video analysis wanted for this
    attempt.
    """
    if not payload.should_analyse_video:
        logger.info("message-processor: video analysis not requested for this attempt")
        return BranchOutcome.unavailable("VIDEO_ANALYSIS_NOT_REQUESTED")

    started = time.perf_counter()
    try:
        normalized = build_review_request(payload)
        summary = normalized.summary
        log_fields = summary.as_log_fields()
        if summary.has_degradations:
            # Expected for real attempts (unopened sections, partial uploads);
            # logged as a warning so it is searchable without being treated as
            # a failure.
            logger.warning(
                "message-processor: staged payload accepted with degradations",
                **log_fields,
            )
        else:
            logger.info("message-processor: staged payload accepted", **log_fields)

        logger.info(
            "message-processor: video analysis started",
            exam_mode=summary.exam_mode,
            camera_chunks=summary.camera_chunks,
            screen_chunks=summary.screen_chunks,
            screen_camera_chunks=summary.screen_camera_chunks,
            rrweb_chunks=summary.rrweb_chunks,
        )
        deps = PipelineDeps(
            object_store=ctx.media_store,
            gemini=ctx.gemini,
            cache=InMemoryPerceptionCache(),
            limiter=ctx.limiter.for_review(),
            logger=logger,
            media_uri_provider=ctx.media_store,
            organization_id=ctx.organization_id,
            ai_usage_logger=ctx.make_ai_usage_logger(
                review_id=payload.review_id,
                org_assess_id=payload.org_assess_id,
                attempt_user_id=payload.attempt_user_id,
            ),
        )
        bundle = await run_integrity_review(normalized.request, deps)
    except Exception as error:  # noqa: BLE001 - every video failure is retryable
        logger.error(
            "message-processor: video analysis failed",
            elapsed_seconds=round(time.perf_counter() - started, 1),
            error=error,
            exc_info=True,
        )
        return BranchOutcome.retryable("ANALYSIS_FAILED")

    logger.info(
        "message-processor: video analysis completed",
        elapsed_seconds=round(time.perf_counter() - started, 1),
        **_bundle_log_fields(bundle),
    )
    return BranchOutcome.success(
        bundle.model_dump(mode="json", by_alias=True, exclude_none=True)
    )


def _bundle_log_fields(bundle: Any) -> dict[str, Any]:
    """The bundle's headline numbers — never the bundle itself, which is large
    and carries candidate content."""

    try:
        return _bundle_headline(bundle)
    except Exception:  # noqa: BLE001 - logging must never fail a completed review
        return {}


def _bundle_headline(bundle: Any) -> dict[str, Any]:
    recommendation = bundle.recommendation
    confidence = bundle.confidence
    return {
        "recommendation_category": str(recommendation.category),
        "recommendation_confidence": round(recommendation.confidence, 3),
        "detected_signals": len(bundle.detected_signals.signals),
        "rejected_signals": len(bundle.detected_signals.rejected_signals),
        "track_b_observations": len(bundle.track_b_observations),
        "media_index_entries": len(bundle.media_index),
        "covered_windows": confidence.covered_windows,
        "total_windows": confidence.total_windows,
        "unknown_windows": confidence.unknown_windows,
    }


def _run_seb_log_reduction_sync(
    seb_deps: SebLogPipelineDeps,
    seb_log_s3_prefix: str,
    org_assessment_id: str,
    user_id: str,
) -> list[Any]:
    """Synchronous reduction wrapper for asyncio.to_thread."""
    return list(
        run_seb_log_reduction(
            seb_deps,
            seb_log_s3_prefix,
            org_assessment_id=org_assessment_id,
            user_id=user_id,
        )
    )


async def _run_seb_log_analysis_branch(
    ctx: WorkerContext,
    payload: StagedReviewPayload,
    logger: Any,
) -> BranchOutcome:
    """Run SEB log reduction and AI analysis, returning the outcome in memory.

    An attempt the backend did not ask for and an empty session list are both
    unavailable rather than retryable: no amount of redelivery makes logs appear
    for this attempt.
    """
    if not payload.should_analyse_seb_logs:
        logger.info("message-processor: SEB analysis not requested for this attempt")
        return BranchOutcome.unavailable("SEB_ANALYSIS_NOT_REQUESTED")

    started = time.perf_counter()
    try:
        seb_deps = SebLogPipelineDeps(
            object_store=ctx.seb_log_object_store,
            logger=logger,
        )
        sessions = await asyncio.to_thread(
            _run_seb_log_reduction_sync,
            seb_deps,
            seb_log_s3_prefix(s3_media_prefix=ctx.s3_media_prefix),
            payload.org_assess_id,
            payload.attempt_user_id,
        )
    except Exception as error:  # noqa: BLE001 - reduction is retryable
        logger.error(
            "message-processor: seb log reduction failed",
            error=error,
            exc_info=True,
        )
        return BranchOutcome.retryable("REDUCTION_FAILED")

    if not sessions:
        logger.warning("message-processor: no SEB sessions found")
        return BranchOutcome.unavailable("NO_SEB_SESSIONS_FOUND")

    ai_deps = SebLogAiDeps(
        gemini=ctx.gemini,
        limiter=ctx.limiter.for_review(),
        logger=logger,
        ai_usage_logger=ctx.make_ai_usage_logger(
            review_id=payload.review_id,
            org_assess_id=payload.org_assess_id,
            attempt_user_id=payload.attempt_user_id,
        ),
    )

    try:
        review_result = await run_seb_log_ai_analysis(
            ai_deps,
            sessions,
            review_id=payload.review_id,
        )
    except Exception as error:  # noqa: BLE001 - AI failure is retryable
        logger.error(
            "message-processor: seb log ai analysis failed",
            error=error,
            exc_info=True,
        )
        return BranchOutcome.retryable("AI_ANALYSIS_FAILED")

    if review_result.sessions_with_ai == 0:
        logger.error(
            "message-processor: no SEB session produced an AI analysis",
            total_sessions=review_result.total_sessions,
            sessions_failed=review_result.sessions_failed,
        )
        return BranchOutcome.retryable("ALL_AI_SESSIONS_FAILED")

    logger.info(
        "message-processor: seb log analysis completed",
        elapsed_seconds=round(time.perf_counter() - started, 1),
        total_sessions=review_result.total_sessions,
        sessions_with_ai=review_result.sessions_with_ai,
        sessions_failed=review_result.sessions_failed,
    )
    return BranchOutcome.success(
        review_result.model_dump(mode="json", by_alias=True, exclude_none=True)
    )


BRANCH_RUNNERS = {
    BRANCH_VIDEO: _run_video_analysis_branch,
    BRANCH_SEB_LOG: _run_seb_log_analysis_branch,
}


async def _run_and_publish(
    ctx: WorkerContext,
    payload: StagedReviewPayload,
    combined: IntegrityReviewResult,
    branches: list[str],
    result_key: str,
    receipt_handle: str,
    logger: Any,
) -> None:
    """Run the pending branches concurrently, merge them into the loaded result,
    write once, and publish."""
    await ctx.task_protection.acquire()
    try:
        async with VisibilityHeartbeat(
            ctx.request_queue,
            receipt_handle,
            interval_seconds=ctx.heartbeat_interval_seconds,
            visibility_timeout_seconds=ctx.visibility_timeout_seconds,
            logger=logger,
        ):
            started = time.perf_counter()
            outcomes = await asyncio.gather(
                *(BRANCH_RUNNERS[branch](ctx, payload, logger) for branch in branches),
                return_exceptions=True,
            )

            for branch, outcome in zip(branches, outcomes, strict=True):
                if isinstance(outcome, BaseException):
                    # Left unresolved, so the branch stays pending and retriggerable.
                    logger.error(
                        "message-processor: branch raised unexpectedly",
                        branch=branch,
                        error=outcome,
                    )
                    continue
                combined.merge(branch, outcome)

            body = combined.to_json()
            try:
                await ctx.result_store.put_object(result_key, body)
            except Exception as error:  # noqa: BLE001 - leave message for redelivery
                logger.error(
                    "message-processor: write combined result failed",
                    error=error,
                    exc_info=True,
                )
                return

            logger.info(
                "message-processor: combined result written",
                elapsed_seconds=round(time.perf_counter() - started, 1),
                result_bytes=len(body),
                result_key=result_key,
                review_status=combined.status,
            )
            try:
                await _publish_combined_result(ctx, combined, logger)
            except Exception as error:  # noqa: BLE001 - leave message for redelivery
                logger.error(
                    "message-processor: publish failed after successful run, retaining message",
                    error=error,
                    exc_info=True,
                )
                return
            await ctx.request_queue.delete(receipt_handle)
            logger.info(
                "message-processor: review acknowledged",
                elapsed_seconds=round(time.perf_counter() - started, 1),
            )
    finally:
        await ctx.task_protection.release()


async def _load_staged_payload(
    ctx: WorkerContext, envelope: SqsRequestEnvelope, logger: Any
) -> StagedReviewPayload:
    """Fetch and validate the staged payload, classifying terminal failures."""

    try:
        raw_payload = await ctx.request_store.get_json(envelope.payload_s3_key)
    except ObjectNotFoundError as error:
        # The backend writes this object once, before enqueueing; if it is gone
        # no redelivery will bring it back.
        raise TerminalPayloadError("STAGED_PAYLOAD_MISSING", str(error)) from error
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise TerminalPayloadError("STAGED_PAYLOAD_NOT_JSON", str(error)) from error

    if not isinstance(raw_payload, dict):
        raise TerminalPayloadError(
            "STAGED_PAYLOAD_NOT_OBJECT", f"top-level {type(raw_payload).__name__}"
        )

    logger.debug(
        "message-processor: staged payload fetched",
        payload_s3_key=envelope.payload_s3_key,
        top_level_keys=sorted(raw_payload.keys()),
    )
    try:
        return StagedReviewPayload.model_validate(raw_payload)
    except ValidationError as error:
        # The staged payload never changes, so redelivery can only repeat this.
        raise TerminalPayloadError("STAGED_PAYLOAD_INVALID", str(error)) from error


async def _process_envelope(
    ctx: WorkerContext, envelope: SqsRequestEnvelope, receipt_handle: str, logger: Any
) -> None:
    payload = await _load_staged_payload(ctx, envelope, logger)

    if payload.review_id != envelope.review_id:
        # Nothing else will ever resolve the review the envelope names, so it
        # has to be failed explicitly rather than silently dropped.
        await _fail_and_ack(
            ctx,
            review_id=envelope.review_id,
            receipt_handle=receipt_handle,
            logger=logger,
            cause="REVIEW_ID_MISMATCH",
            detail=f"staged payload carries review_id {payload.review_id}",
        )
        return

    result_key = combined_result_object_key(
        s3_media_prefix=ctx.s3_media_prefix,
        org_assess_id=payload.org_assess_id,
        attempt_user_id=payload.attempt_user_id,
    )

    combined = await _load_existing_result(
        ctx, result_key, review_id=payload.review_id, logger=logger
    )
    if combined is None:
        return

    branches = combined.pending_branches()
    if not branches:
        logger.info(
            "message-processor: nothing pending, republishing without Gemini",
            result_key=result_key,
            review_status=combined.status,
        )
        try:
            await _publish_combined_result(ctx, combined, logger)
        except Exception as error:  # noqa: BLE001 - leave message for redelivery
            logger.error(
                "message-processor: republish failed, retaining message",
                error=error,
            )
            return
        await ctx.request_queue.delete(receipt_handle)
        return

    await _run_and_publish(
        ctx,
        payload,
        combined,
        branches,
        result_key,
        receipt_handle,
        logger,
    )


async def handle_message(ctx: WorkerContext, message: dict[str, Any]) -> None:
    # Poison-message policy: a malformed envelope or unsupported message_type can
    # never become processable by redelivery, so it is deleted directly here
    # rather than left in place for the queue's own redrive-to-DLQ policy.
    receipt_handle = message["ReceiptHandle"]
    meta = MessageMeta.from_message(message)
    body = message.get("Body", "")

    try:
        envelope = SqsRequestEnvelope.model_validate(json.loads(body))
    except (json.JSONDecodeError, ValueError) as error:
        ctx.make_logger("unparsed").error(
            "message-processor: malformed envelope, dropping",
            error=error,
            body_bytes=len(body),
            **meta.as_log_fields(),
        )
        await ctx.request_queue.delete(receipt_handle)
        return

    logger = ctx.make_logger(envelope.review_id)
    logger.info(
        "message-processor: message received",
        message_type=envelope.message_type,
        payload_s3_key=envelope.payload_s3_key,
        **meta.as_log_fields(),
    )

    if envelope.message_type != REQUEST_MESSAGE_TYPE:
        logger.error(
            "message-processor: unsupported message_type, dropping",
            message_type=envelope.message_type,
        )
        await ctx.request_queue.delete(receipt_handle)
        return

    is_last_attempt = meta.receive_count >= ctx.max_receive_count
    if meta.receive_count > 1:
        logger.warning(
            "message-processor: redelivered message",
            receive_count=meta.receive_count,
            max_receive_count=ctx.max_receive_count,
            is_last_attempt=is_last_attempt,
        )

    try:
        await _process_envelope(ctx, envelope, receipt_handle, logger)
    except TerminalPayloadError as error:
        await _fail_and_ack(
            ctx,
            review_id=envelope.review_id,
            receipt_handle=receipt_handle,
            logger=logger,
            cause=error.cause,
            detail=error.detail,
        )
    except Exception as error:  # noqa: BLE001 - worker safety net
        if is_last_attempt:
            # Last delivery before the DLQ: report the failure now, otherwise
            # the review stays IN_PROGRESS until the backend's stale sweeper.
            await _fail_and_ack(
                ctx,
                review_id=envelope.review_id,
                receipt_handle=receipt_handle,
                logger=logger,
                cause="RETRIES_EXHAUSTED",
                detail=f"{type(error).__name__}: {error}",
            )
            return
        logger.error(
            "message-processor: unhandled error, retaining message",
            receive_count=meta.receive_count,
            error=error,
            exc_info=True,
        )
