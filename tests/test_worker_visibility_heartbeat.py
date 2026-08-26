"""Tests for the SQS visibility-timeout heartbeat."""

from __future__ import annotations

import asyncio

import pytest

from integrity_review_pipeline.worker.visibility_heartbeat import VisibilityHeartbeat

from .worker_fakes import FailingChangeVisibilitySqsClient, FakeSqsClient, RecordingLogger

RECEIPT_HANDLE = "receipt-handle-1"


@pytest.mark.asyncio
async def test_extends_visibility_on_every_interval() -> None:
    sqs = FakeSqsClient()
    async with VisibilityHeartbeat(
        sqs,
        RECEIPT_HANDLE,
        interval_seconds=0.01,
        visibility_timeout_seconds=300,
        logger=RecordingLogger(),
    ):
        await asyncio.sleep(0.035)

    assert len(sqs.visibility_changes) >= 2
    assert all(call == (RECEIPT_HANDLE, 300) for call in sqs.visibility_changes)


@pytest.mark.asyncio
async def test_stops_extending_once_the_context_exits() -> None:
    sqs = FakeSqsClient()
    async with VisibilityHeartbeat(
        sqs,
        RECEIPT_HANDLE,
        interval_seconds=0.01,
        visibility_timeout_seconds=300,
        logger=RecordingLogger(),
    ):
        await asyncio.sleep(0.015)

    calls_at_exit = len(sqs.visibility_changes)
    await asyncio.sleep(0.03)

    assert len(sqs.visibility_changes) == calls_at_exit


@pytest.mark.asyncio
async def test_extend_failure_is_logged_and_does_not_raise() -> None:
    sqs = FailingChangeVisibilitySqsClient()
    logger = RecordingLogger()
    async with VisibilityHeartbeat(
        sqs,
        RECEIPT_HANDLE,
        interval_seconds=0.01,
        visibility_timeout_seconds=300,
        logger=logger,
    ):
        await asyncio.sleep(0.025)

    assert logger.warnings
