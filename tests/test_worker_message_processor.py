"""Tests for the idempotent single-message worker flow.

Covers: result deduplication, publish-before-delete ordering, redelivery on
publish failure, handled pipeline failure, and envelope validation.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from integrity_review_pipeline.worker import message_processor
from integrity_review_pipeline.worker.message_processor import (
    REQUEST_MESSAGE_TYPE,
    RESULT_MESSAGE_TYPE,
    WorkerContext,
    handle_message,
    result_object_key,
)

from .worker_fakes import (
    FakeFairLimiter,
    FakeObjectStore,
    FakeSqsClient,
    FakeTaskProtection,
    RecordingLogger,
)

STAGE = "test"
REVIEW_ID = "review-1"
ORG_ASSESS_ID = "assessment-1"
ATTEMPT_USER_ID = "candidate-1"
PAYLOAD_KEY = f"{STAGE}/media/ai_integrity_review_requests/{REVIEW_ID}.json"
RECEIPT_HANDLE = "receipt-handle-1"


def _raw_payload(*, review_id: str = REVIEW_ID) -> dict[str, Any]:
    return {
        "review_id": review_id,
        "org_assess_id": ORG_ASSESS_ID,
        "attempt_user_id": ATTEMPT_USER_ID,
        "manifest": {
            "chunks": [
                {
                    "chunk_id": "camera-1",
                    "media_type": "CAMERA_VIDEO",
                    "exam_attempt_id": "attempt-1",
                    "s3_key": "media/camera/attempt-1/1778001319000__60000.webm",
                    "epoch_ms": 1_778_001_319_000,
                    "duration_ms": 60_000,
                }
            ]
        },
        "activity_timeline": {
            "activity_logs": [
                {
                    "order": 0,
                    "activity_type": "ASSESSMENT_STARTED",
                    "creation_datetime": "2026-05-05 22:44:19",
                }
            ],
            "sections": [
                {
                    "section_id": "mcq",
                    "exam_id": "exam-1",
                    "order": 1,
                    "exam_attempt_id": "attempt-1",
                    "start_datetime": "2026-05-05 22:44:19",
                    "end_datetime": None,
                }
            ],
        },
    }


def _envelope_message(
    *, message_type: str = REQUEST_MESSAGE_TYPE, review_id: str = REVIEW_ID
) -> dict[str, Any]:
    body = {
        "message_type": message_type,
        "review_id": review_id,
        "payload_s3_key": PAYLOAD_KEY,
    }
    return {"Body": json.dumps(body), "ReceiptHandle": RECEIPT_HANDLE}


def _make_ctx(
    *,
    request_store: FakeObjectStore,
    result_store: FakeObjectStore,
    request_queue: FakeSqsClient,
    result_queue: FakeSqsClient,
    task_protection: FakeTaskProtection,
) -> WorkerContext:
    return WorkerContext(
        stage=STAGE,
        organization_id="test-org",
        media_store=FakeObjectStore(),
        request_store=request_store,
        result_store=result_store,
        request_queue=request_queue,
        result_queue=result_queue,
        gemini=None,
        limiter=FakeFairLimiter(),
        task_protection=task_protection,
        make_logger=lambda review_id: RecordingLogger(),
        heartbeat_interval_seconds=9_999,
        visibility_timeout_seconds=300,
    )


@pytest.mark.asyncio
async def test_existing_result_republishes_without_running_pipeline(monkeypatch) -> None:
    async def _fail_if_called(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("run_integrity_review must not run when a result already exists")

    monkeypatch.setattr(message_processor, "run_integrity_review", _fail_if_called)

    request_store = FakeObjectStore(json_by_ref={PAYLOAD_KEY: _raw_payload()})
    result_store = FakeObjectStore(existing_result=True)
    request_queue = FakeSqsClient()
    result_queue = FakeSqsClient()
    task_protection = FakeTaskProtection()
    ctx = _make_ctx(
        request_store=request_store,
        result_store=result_store,
        request_queue=request_queue,
        result_queue=result_queue,
        task_protection=task_protection,
    )

    await handle_message(ctx, _envelope_message())

    assert len(result_queue.sent) == 1
    sent = json.loads(result_queue.sent[0])
    assert sent == {
        "message_type": RESULT_MESSAGE_TYPE,
        "review_id": REVIEW_ID,
        "review_status": "SUCCESS",
        "error_message": None,
    }
    assert request_queue.deleted == [RECEIPT_HANDLE]
    assert task_protection.acquired == 0


@pytest.mark.asyncio
async def test_success_writes_result_then_publishes_then_deletes(monkeypatch) -> None:
    from .worker_fakes import FakeBundle

    async def _fake_run(*_args: Any, **_kwargs: Any) -> FakeBundle:
        return FakeBundle({"reviewId": REVIEW_ID})

    monkeypatch.setattr(message_processor, "run_integrity_review", _fake_run)

    request_store = FakeObjectStore(json_by_ref={PAYLOAD_KEY: _raw_payload()})
    result_store = FakeObjectStore(existing_result=False)
    request_queue = FakeSqsClient()
    result_queue = FakeSqsClient()
    task_protection = FakeTaskProtection()
    ctx = _make_ctx(
        request_store=request_store,
        result_store=result_store,
        request_queue=request_queue,
        result_queue=result_queue,
        task_protection=task_protection,
    )

    await handle_message(ctx, _envelope_message())

    expected_key = result_object_key(
        stage=STAGE,
        org_assess_id=ORG_ASSESS_ID,
        attempt_user_id=ATTEMPT_USER_ID,
        review_id=REVIEW_ID,
    )
    assert [ref for ref, _, _ in result_store.put_calls] == [expected_key]
    assert len(result_queue.sent) == 1
    assert json.loads(result_queue.sent[0])["review_status"] == "SUCCESS"
    assert request_queue.deleted == [RECEIPT_HANDLE]
    assert task_protection.acquired == 1
    assert task_protection.released == 1


@pytest.mark.asyncio
async def test_publish_failure_after_success_retains_request(monkeypatch) -> None:
    from .worker_fakes import FakeBundle

    async def _fake_run(*_args: Any, **_kwargs: Any) -> FakeBundle:
        return FakeBundle()

    monkeypatch.setattr(message_processor, "run_integrity_review", _fake_run)

    request_store = FakeObjectStore(json_by_ref={PAYLOAD_KEY: _raw_payload()})
    result_store = FakeObjectStore(existing_result=False)
    request_queue = FakeSqsClient()
    result_queue = FakeSqsClient(send_raises=True)
    task_protection = FakeTaskProtection()
    ctx = _make_ctx(
        request_store=request_store,
        result_store=result_store,
        request_queue=request_queue,
        result_queue=result_queue,
        task_protection=task_protection,
    )

    await handle_message(ctx, _envelope_message())

    assert len(result_store.put_calls) == 1
    assert request_queue.deleted == []
    assert task_protection.acquired == 1
    assert task_protection.released == 1


@pytest.mark.asyncio
async def test_pipeline_failure_publishes_failure_and_deletes(monkeypatch) -> None:
    async def _fake_run(*_args: Any, **_kwargs: Any) -> Any:
        raise ValueError("pipeline exploded")

    monkeypatch.setattr(message_processor, "run_integrity_review", _fake_run)

    request_store = FakeObjectStore(json_by_ref={PAYLOAD_KEY: _raw_payload()})
    result_store = FakeObjectStore(existing_result=False)
    request_queue = FakeSqsClient()
    result_queue = FakeSqsClient()
    task_protection = FakeTaskProtection()
    ctx = _make_ctx(
        request_store=request_store,
        result_store=result_store,
        request_queue=request_queue,
        result_queue=result_queue,
        task_protection=task_protection,
    )

    await handle_message(ctx, _envelope_message())

    assert result_store.put_calls == []
    assert len(result_queue.sent) == 1
    sent = json.loads(result_queue.sent[0])
    assert sent["review_status"] == "FAILURE"
    assert sent["error_message"] == "pipeline exploded"
    assert request_queue.deleted == [RECEIPT_HANDLE]


@pytest.mark.asyncio
async def test_failure_publish_failure_retains_request(monkeypatch) -> None:
    async def _fake_run(*_args: Any, **_kwargs: Any) -> Any:
        raise ValueError("pipeline exploded")

    monkeypatch.setattr(message_processor, "run_integrity_review", _fake_run)

    request_store = FakeObjectStore(json_by_ref={PAYLOAD_KEY: _raw_payload()})
    result_store = FakeObjectStore(existing_result=False)
    request_queue = FakeSqsClient()
    result_queue = FakeSqsClient(send_raises=True)
    task_protection = FakeTaskProtection()
    ctx = _make_ctx(
        request_store=request_store,
        result_store=result_store,
        request_queue=request_queue,
        result_queue=result_queue,
        task_protection=task_protection,
    )

    await handle_message(ctx, _envelope_message())

    assert request_queue.deleted == []


@pytest.mark.asyncio
async def test_review_id_mismatch_deletes_without_running_pipeline(monkeypatch) -> None:
    async def _fail_if_called(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("run_integrity_review must not run on a mismatched review_id")

    monkeypatch.setattr(message_processor, "run_integrity_review", _fail_if_called)

    request_store = FakeObjectStore(json_by_ref={PAYLOAD_KEY: _raw_payload(review_id="other-review")})
    result_store = FakeObjectStore()
    request_queue = FakeSqsClient()
    result_queue = FakeSqsClient()
    task_protection = FakeTaskProtection()
    ctx = _make_ctx(
        request_store=request_store,
        result_store=result_store,
        request_queue=request_queue,
        result_queue=result_queue,
        task_protection=task_protection,
    )

    await handle_message(ctx, _envelope_message())

    assert request_queue.deleted == [RECEIPT_HANDLE]
    assert result_queue.sent == []


@pytest.mark.asyncio
async def test_malformed_envelope_is_deleted() -> None:
    request_store = FakeObjectStore()
    result_store = FakeObjectStore()
    request_queue = FakeSqsClient()
    result_queue = FakeSqsClient()
    task_protection = FakeTaskProtection()
    ctx = _make_ctx(
        request_store=request_store,
        result_store=result_store,
        request_queue=request_queue,
        result_queue=result_queue,
        task_protection=task_protection,
    )
    message = {"Body": "not-json", "ReceiptHandle": RECEIPT_HANDLE}

    await handle_message(ctx, message)

    assert request_queue.deleted == [RECEIPT_HANDLE]


@pytest.mark.asyncio
async def test_unsupported_message_type_is_deleted(monkeypatch) -> None:
    async def _fail_if_called(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("run_integrity_review must not run for an unsupported message_type")

    monkeypatch.setattr(message_processor, "run_integrity_review", _fail_if_called)

    request_store = FakeObjectStore()
    result_store = FakeObjectStore()
    request_queue = FakeSqsClient()
    result_queue = FakeSqsClient()
    task_protection = FakeTaskProtection()
    ctx = _make_ctx(
        request_store=request_store,
        result_store=result_store,
        request_queue=request_queue,
        result_queue=result_queue,
        task_protection=task_protection,
    )

    await handle_message(ctx, _envelope_message(message_type="SOME_OTHER_TYPE"))

    assert request_queue.deleted == [RECEIPT_HANDLE]
    assert result_queue.sent == []
