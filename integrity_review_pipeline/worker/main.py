"""Worker entrypoint: two-slot long-poll loop over the request queue with SIGTERM drain."""

from __future__ import annotations

import asyncio
import logging
import os
import signal

from ..adapters import GoogleGeminiClient, S3ObjectStore, SqsClient, StructuredLogger
from .config import WorkerConfig
from .fair_limiter import FairGeminiLimiter
from .message_processor import WorkerContext, handle_message
from .task_protection import TaskProtection

_LOGGER = logging.getLogger("integrity_review_pipeline.worker")


def _build_context(config: WorkerConfig) -> WorkerContext:
    return WorkerContext(
        stage=config.stage,
        organization_id=config.organization_id,
        media_store=S3ObjectStore(
            config.media_bucket,
            region_name=config.aws_region,
            presign_expires_in_seconds=config.presign_expires_in_seconds,
        ),
        request_store=S3ObjectStore(config.request_bucket, region_name=config.aws_region),
        result_store=S3ObjectStore(config.result_bucket, region_name=config.aws_region),
        request_queue=SqsClient(config.request_queue_url, region_name=config.aws_region),
        result_queue=SqsClient(config.result_queue_url, region_name=config.aws_region),
        gemini=GoogleGeminiClient(config.gemini_api_key),
        limiter=FairGeminiLimiter(
            global_limit=config.gemini_task_limit,
            max_concurrent_reviews=config.max_concurrent_reviews,
        ),
        task_protection=TaskProtection(logger=StructuredLogger("task-protection")),
        make_logger=StructuredLogger,
        heartbeat_interval_seconds=config.heartbeat_interval_seconds,
        visibility_timeout_seconds=config.visibility_timeout_seconds,
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
            _LOGGER.error("worker: message task raised unexpectedly", exc_info=error)
    return pending


async def _poll_loop(ctx: WorkerContext, config: WorkerConfig, stopping: asyncio.Event) -> None:
    in_flight: set[asyncio.Task[None]] = set()
    while not stopping.is_set():
        in_flight = await _prune(in_flight)
        free_slots = config.max_concurrent_reviews - len(in_flight)
        if free_slots <= 0:
            _, in_flight = await asyncio.wait(in_flight, return_when=asyncio.FIRST_COMPLETED)
            continue

        messages = await ctx.request_queue.receive(
            max_messages=free_slots,
            wait_time_seconds=config.poll_wait_time_seconds,
            visibility_timeout=config.visibility_timeout_seconds,
        )
        for message in messages:
            in_flight.add(asyncio.ensure_future(handle_message(ctx, message)))

    if in_flight:
        _LOGGER.info("worker: draining %d in-flight review(s) before exit", len(in_flight))
        await asyncio.wait(in_flight)


async def run(config: WorkerConfig | None = None) -> None:
    config = config or WorkerConfig.from_env()
    ctx = _build_context(config)

    stopping = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stopping.set)

    _LOGGER.info("worker: starting poll loop")
    await _poll_loop(ctx, config, stopping)
    _LOGGER.info("worker: stopped")


def main() -> None:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    asyncio.run(run())


if __name__ == "__main__":
    main()
