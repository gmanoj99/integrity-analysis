"""Worker entrypoint: two-slot long-poll loop over the request queue with SIGTERM drain."""

from __future__ import annotations

import asyncio
import os
import signal
import time

from ..adapters import (
    CloudWatchAiUsageLoggerFactory,
    GoogleGeminiClient,
    S3ObjectStore,
    SqsClient,
    StructuredLogger,
    configure_logging,
)
from ..integrity_review_pipeline.adapters import S3ObjectStore as SebLogS3Store
from .config import WorkerConfig
from .fair_limiter import FairGeminiLimiter
from .message_processor import WorkerContext, handle_message
from .task_protection import TaskProtection

# Worker-scope logger: same single-line JSON shape as the per-review loggers,
# without a reviewId, so container startup and poll-loop health are searchable
# in the same log group as the reviews themselves.
_LOG = StructuredLogger(scope="worker")


def _build_context(config: WorkerConfig) -> WorkerContext:
    store = S3ObjectStore(
        config.storage_bucket,
        region_name=config.aws_region,
        presign_expires_in_seconds=config.presign_expires_in_seconds,
    )
    seb_log_store = SebLogS3Store(
        config.storage_bucket,
        region_name=config.aws_region,
    )
    return WorkerContext(
        s3_media_prefix=config.s3_media_prefix,
        organization_id=config.organization_id,
        media_store=store,
        request_store=store,
        result_store=store,
        request_queue=SqsClient(config.request_queue_url, region_name=config.aws_region),
        response_queue=SqsClient(config.response_queue_url, region_name=config.aws_region),
        gemini=GoogleGeminiClient(config.gemini_api_key),
        limiter=FairGeminiLimiter(
            global_limit=config.gemini_task_limit,
            per_review_limit=config.gemini_per_review_limit,
        ),
        task_protection=TaskProtection(logger=StructuredLogger(scope="task-protection")),
        make_logger=StructuredLogger,
        make_ai_usage_logger=CloudWatchAiUsageLoggerFactory(
            log_group_name=config.custom_ai_logs_group_name,
            log_stream_name=config.custom_ai_logs_stream_name,
            region_name=config.aws_region,
        ),
        heartbeat_interval_seconds=config.heartbeat_interval_seconds,
        visibility_timeout_seconds=config.visibility_timeout_seconds,
        seb_log_object_store=seb_log_store,
        max_receive_count=config.max_receive_count,
    )


async def _prune(in_flight: set[asyncio.Task[None]]) -> set[asyncio.Task[None]]:
    """Drop finished tasks, surfacing any that raised despite the processor's own safety net."""

    if not in_flight:
        return in_flight
    done, pending = await asyncio.wait(in_flight, timeout=0)
    for task in done:
        if task.cancelled():
            continue
        error = task.exception()
        if error is not None:
            _LOG.error("worker: message task raised unexpectedly", error=error, exc_info=True)
    return pending


async def _poll_loop(ctx: WorkerContext, config: WorkerConfig, stopping: asyncio.Event) -> None:
    in_flight: set[asyncio.Task[None]] = set()
    polls = 0
    received_total = 0
    last_idle_report = time.monotonic()

    while not stopping.is_set():
        in_flight = await _prune(in_flight)
        free_slots = config.max_concurrent_reviews - len(in_flight)
        if free_slots <= 0:
            _LOG.info(
                "worker: all review slots busy, waiting for one to finish",
                in_flight=len(in_flight),
                max_concurrent_reviews=config.max_concurrent_reviews,
            )
            _, in_flight = await asyncio.wait(in_flight, return_when=asyncio.FIRST_COMPLETED)
            continue

        try:
            messages = await ctx.request_queue.receive(
                max_messages=free_slots,
                wait_time_seconds=config.poll_wait_time_seconds,
                visibility_timeout=config.visibility_timeout_seconds,
            )
        except Exception as error:  # noqa: BLE001 - keep polling through transient SQS errors
            _LOG.error("worker: receive failed, retrying", error=error, exc_info=True)
            await asyncio.sleep(min(config.poll_wait_time_seconds, 10))
            continue

        polls += 1
        if messages:
            received_total += len(messages)
            _LOG.info(
                "worker: messages received",
                count=len(messages),
                in_flight=len(in_flight),
                free_slots=free_slots,
                received_total=received_total,
            )
            for message in messages:
                in_flight.add(asyncio.ensure_future(handle_message(ctx, message)))
            continue

        # An empty long-poll is the normal idle state; report it periodically so
        # the log shows a live, healthy consumer instead of going silent.
        now = time.monotonic()
        if now - last_idle_report >= 300:
            _LOG.info(
                "worker: idle",
                polls=polls,
                in_flight=len(in_flight),
                received_total=received_total,
            )
            last_idle_report = now

    if in_flight:
        _LOG.info("worker: draining in-flight reviews before exit", in_flight=len(in_flight))
        await asyncio.wait(in_flight)
        _LOG.info("worker: drain complete")


async def run(config: WorkerConfig | None = None) -> None:
    config = config or WorkerConfig.from_env()
    _LOG.info("worker: configuration resolved", **config.as_log_fields())
    ctx = _build_context(config)

    stopping = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _on_signal(sig, stopping))

    _LOG.info("worker: starting poll loop")
    started = time.monotonic()
    try:
        await _poll_loop(ctx, config, stopping)
    finally:
        _LOG.info("worker: stopped", uptime_seconds=round(time.monotonic() - started, 1))


def _on_signal(sig: signal.Signals, stopping: asyncio.Event):
    def handler() -> None:
        _LOG.info("worker: shutdown signal received, draining", signal=sig.name)
        stopping.set()

    return handler


def main() -> None:
    # Logging is configured before anything else so even a bad environment
    # produces one clear line instead of an unformatted traceback.
    configure_logging(os.environ.get("LOG_LEVEL", "INFO"))
    try:
        config = WorkerConfig.from_env()
    except RuntimeError as error:
        _LOG.error("worker: invalid configuration, exiting", error=error)
        raise
    asyncio.run(run(config))


if __name__ == "__main__":
    main()
