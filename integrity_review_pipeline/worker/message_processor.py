"""Idempotent single-message processing: parse, dedupe, run, publish, ack.

Ordering invariant: the staged request message is deleted only *after* the
result has been durably published (object written + SQS message sent). If
publishing fails, the message is left in place for SQS to redeliver.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

from ..adapters.memory_cache import InMemoryPerceptionCache
from ..deps import PipelineDeps
from ..pipeline import run_integrity_review
from .contracts import SqsRequestEnvelope, StagedReviewPayload, build_review_request
from .fair_limiter import FairGeminiLimiter
from .task_protection import TaskProtection
from .visibility_heartbeat import VisibilityHeartbeat

REQUEST_MESSAGE_TYPE = "VIDEO_ANALYSIS_REQUEST"
RESULT_MESSAGE_TYPE = "VIDEO_ANALYSIS_RESPONSE"
RESULT_KEY_TEMPLATE = (
    "{stage}/media/ai_integrity_review_results/"
    "{org_assess_id}/{attempt_user_id}/{review_id}/evidence_bundle.json"
)


class ObjectStoreWithHead(Protocol):
    async def get_json(self, ref: str) -> Any: ...

    async def head_object(self, ref: str) -> bool: ...

    async def put_object(self, ref: str, body: bytes, *, content_type: str = "application/json") -> None: ...

    async def media_uri_for(self, chunk: Any) -> str: ...

    async def get_bytes(self, ref: str) -> bytes: ...


class WorkerSqsClient(Protocol):
    async def delete(self, receipt_handle: str) -> None: ...

    async def change_visibility(self, receipt_handle: str, visibility_timeout: int) -> None: ...

    async def send(self, body: str) -> None: ...


class LoggerFactory(Protocol):
    def __call__(self, review_id: str) -> Any: ...


@dataclass(frozen=True, slots=True)
class WorkerContext:
    stage: str
    organization_id: str
    media_store: ObjectStoreWithHead
    request_store: ObjectStoreWithHead
    result_store: ObjectStoreWithHead
    request_queue: WorkerSqsClient
    result_queue: WorkerSqsClient
    gemini: Any
    limiter: FairGeminiLimiter
    task_protection: TaskProtection
    make_logger: LoggerFactory
    heartbeat_interval_seconds: int
    visibility_timeout_seconds: int


def result_object_key(*, stage: str, org_assess_id: str, attempt_user_id: str, review_id: str) -> str:
    return RESULT_KEY_TEMPLATE.format(
        stage=stage,
        org_assess_id=org_assess_id,
        attempt_user_id=attempt_user_id,
        review_id=review_id,
    )


async def _publish_result(
    ctx: WorkerContext, *, review_id: str, review_status: str, error_message: str | None
) -> None:
    body = json.dumps(
        {
            "message_type": RESULT_MESSAGE_TYPE,
            "review_id": review_id,
            "review_status": review_status,
            "error_message": error_message,
        }
    )
    await ctx.result_queue.send(body)


async def _run_and_publish(
    ctx: WorkerContext,
    payload: StagedReviewPayload,
    result_key: str,
    receipt_handle: str,
    logger: Any,
) -> None:
    await ctx.task_protection.acquire()
    try:
        async with VisibilityHeartbeat(
            ctx.request_queue,
            receipt_handle,
            interval_seconds=ctx.heartbeat_interval_seconds,
            visibility_timeout_seconds=ctx.visibility_timeout_seconds,
            logger=logger,
        ):
            try:
                review_request = build_review_request(payload)
                deps = PipelineDeps(
                    object_store=ctx.media_store,
                    gemini=ctx.gemini,
                    cache=InMemoryPerceptionCache(),
                    limiter=ctx.limiter.for_review(),
                    logger=logger,
                    media_uri_provider=ctx.media_store,
                    organization_id=ctx.organization_id,
                )
                bundle = await run_integrity_review(review_request, deps)
            except Exception as error:  # noqa: BLE001 - publish FAILURE for any pipeline error
                logger.error("message-processor: review failed", error=str(error))
                try:
                    await _publish_result(
                        ctx,
                        review_id=payload.review_id,
                        review_status="FAILURE",
                        error_message=str(error),
                    )
                except Exception as publish_error:  # noqa: BLE001
                    logger.error(
                        "message-processor: failure publish failed, retaining message",
                        error=str(publish_error),
                    )
                    return
                await ctx.request_queue.delete(receipt_handle)
                return

            body = bundle.model_dump_json(by_alias=True, exclude_none=True).encode("utf-8")
            try:
                await ctx.result_store.put_object(result_key, body)
                await _publish_result(
                    ctx,
                    review_id=payload.review_id,
                    review_status="SUCCESS",
                    error_message=None,
                )
            except Exception as error:  # noqa: BLE001 - leave message for redelivery
                logger.error(
                    "message-processor: publish failed after successful run, retaining message",
                    error=str(error),
                )
                return
            await ctx.request_queue.delete(receipt_handle)
    finally:
        await ctx.task_protection.release()


async def _process_envelope(
    ctx: WorkerContext, envelope: SqsRequestEnvelope, receipt_handle: str, logger: Any
) -> None:
    raw_payload = await ctx.request_store.get_json(envelope.payload_s3_key)
    payload = StagedReviewPayload.model_validate(raw_payload)

    if payload.review_id != envelope.review_id:
        logger.error(
            "message-processor: review_id mismatch, dropping",
            envelope_review_id=envelope.review_id,
            payload_review_id=payload.review_id,
        )
        await ctx.request_queue.delete(receipt_handle)
        return

    result_key = result_object_key(
        stage=ctx.stage,
        org_assess_id=payload.org_assess_id,
        attempt_user_id=payload.attempt_user_id,
        review_id=payload.review_id,
    )

    if await ctx.result_store.head_object(result_key):
        logger.info("message-processor: result already exists, republishing without Gemini")
        try:
            await _publish_result(
                ctx, review_id=payload.review_id, review_status="SUCCESS", error_message=None
            )
        except Exception as error:  # noqa: BLE001 - leave message for redelivery
            logger.error("message-processor: republish failed, retaining message", error=str(error))
            return
        await ctx.request_queue.delete(receipt_handle)
        return

    await _run_and_publish(ctx, payload, result_key, receipt_handle, logger)


async def handle_message(ctx: WorkerContext, message: dict[str, Any]) -> None:
    receipt_handle = message["ReceiptHandle"]
    try:
        envelope = SqsRequestEnvelope.model_validate(json.loads(message.get("Body", "")))
    except (json.JSONDecodeError, ValueError) as error:
        ctx.make_logger("unparsed").error(
            "message-processor: malformed envelope, dropping", error=str(error)
        )
        await ctx.request_queue.delete(receipt_handle)
        return

    logger = ctx.make_logger(envelope.review_id)
    if envelope.message_type != REQUEST_MESSAGE_TYPE:
        logger.error(
            "message-processor: unsupported message_type, dropping",
            message_type=envelope.message_type,
        )
        await ctx.request_queue.delete(receipt_handle)
        return

    try:
        await _process_envelope(ctx, envelope, receipt_handle, logger)
    except Exception as error:  # noqa: BLE001 - worker safety net; message stays for redelivery
        logger.error("message-processor: unhandled error, retaining message", error=str(error))
