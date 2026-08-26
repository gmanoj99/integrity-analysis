"""Worker message-processor tests with fake S3/SQS: idempotency, publish ordering, retention."""

from __future__ import annotations

import json
from typing import Any

import pytest

from integrity_review_pipeline.worker import message_processor as mp
from integrity_review_pipeline.worker.fair_limiter import FairGeminiLimiter
from integrity_review_pipeline.worker.task_protection import TaskProtection


class FakeLogger:
    def info(self, message: str, **fields: Any) -> None:
        pass

    def error(self, message: str, **fields: Any) -> None:
        pass

    def warning(self, message: str, **fields: Any) -> None:
        pass

    def debug(self, message: str, **fields: Any) -> None:
        pass


class FakeObjectStore:
    def __init__(self, json_data: Any = None, *, has_result: bool = False, fail_put: bool = False) -> None:
        self._json_data = json_data
        self._has_result = has_result
        self._fail_put = fail_put
        self.head_calls: list[str] = []
        self.put_calls: list[str] = []

    async def get_json(self, ref: str) -> Any:
        return self._json_data

    async def head_object(self, ref: str) -> bool:
        self.head_calls.append(ref)
        return self._has_result

    async def put_object(self, ref: str, body: bytes, *, content_type: str = "application/json") -> None:
        if self._fail_put:
            raise RuntimeError("simulated S3 put failure")
        self.put_calls.append(ref)

    async def media_uri_for(self, chunk: Any) -> str:
        return "https://example.test/signed"

    async def get_bytes(self, ref: str) -> bytes:
        return b"{}"


class FakeSqsClient:
    def __init__(self, *, fail_send: bool = False) -> None:
        self._fail_send = fail_send
        self.sent: list[str] = []
        self.deleted: list[str] = []
        self.visibility_changes: list[tuple[str, int]] = []

    async def delete(self, receipt_handle: str) -> None:
        self.deleted.append(receipt_handle)

    async def change_visibility(self, receipt_handle: str, visibility_timeout: int) -> None:
        self.visibility_changes.append((receipt_handle, visibility_timeout))

    async def send(self, body: str) -> None:
        if self._fail_send:
            raise RuntimeError("simulated SQS send failure")
        self.sent.append(body)


def _payload_json(review_id: str = "review-1") -> dict[str, Any]:
    return {
        "review_id": review_id,
        "org_assess_id": "assess-1",
        "attempt_user_id": "candidate-1",
        "manifest": {"chunks": []},
        "activity_timeline": {"activity_logs": [], "sections": []},
    }


def _envelope_body(review_id: str = "review-1", payload_s3_key: str = "requests/review-1.json") -> str:
    return json.dumps(
        {
            "message_type": "VIDEO_ANALYSIS_REQUEST",
            "review_id": review_id,
            "payload_s3_key": payload_s3_key,
        }
    )


def _make_ctx(
    *,
    request_store: FakeObjectStore,
    result_store: FakeObjectStore,
    request_queue: FakeSqsClient,
    result_queue: FakeSqsClient,
) -> mp.WorkerContext:
    return mp.WorkerContext(
        stage="test",
        media_store=FakeObjectStore(),
        request_store=request_store,
        result_store=result_store,
        request_queue=request_queue,
        result_queue=result_queue,
        gemini=object(),
        limiter=FairGeminiLimiter(global_limit=2, max_concurrent_reviews=2),
        task_protection=TaskProtection(),
        make_logger=lambda review_id: FakeLogger(),
        heartbeat_interval_seconds=300,
        visibility_timeout_seconds=60,
    )


@pytest.mark.asyncio
async def test_existing_result_republishes_success_without_gemini(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _unexpected_run(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("run_integrity_review must not be called when a result already exists")

    monkeypatch.setattr(mp, "run_integrity_review", _unexpected_run)

    request_store = FakeObjectStore(json_data=_payload_json())
    result_store = FakeObjectStore(has_result=True)
    request_queue = FakeSqsClient()
    result_queue = FakeSqsClient()
    ctx = _make_ctx(
        request_store=request_store,
        result_store=result_store,
        request_queue=request_queue,
        result_queue=result_queue,
    )

    message = {"ReceiptHandle": "rh-1", "Body": _envelope_body()}
    await mp.handle_message(ctx, message)

    assert result_store.head_calls
    assert len(result_queue.sent) == 1
    published = json.loads(result_queue.sent[0])
    assert published == {
        "message_type": "VIDEO_ANALYSIS_RESPONSE",
        "review_id": "review-1",
        "review_status": "SUCCESS",
        "error_message": None,
    }
    assert request_queue.deleted == ["rh-1"]


@pytest.mark.asyncio
async def test_publish_then_delete_ordering_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeBundle:
        def model_dump_json(self, *, by_alias: bool, exclude_none: bool) -> str:
            return "{}"

    async def _fake_run(*args: Any, **kwargs: Any) -> Any:
        return FakeBundle()

    monkeypatch.setattr(mp, "run_integrity_review", _fake_run)

    request_store = FakeObjectStore(json_data=_payload_json())
    result_store = FakeObjectStore(has_result=False)
    request_queue = FakeSqsClient()
    result_queue = FakeSqsClient()
    ctx = _make_ctx(
        request_store=request_store,
        result_store=result_store,
        request_queue=request_queue,
        result_queue=result_queue,
    )

    message = {"ReceiptHandle": "rh-2", "Body": _envelope_body()}
    await mp.handle_message(ctx, message)

    assert len(result_store.put_calls) == 1
    assert len(result_queue.sent) == 1
    assert request_queue.deleted == ["rh-2"]

    published = json.loads(result_queue.sent[0])
    assert published["review_status"] == "SUCCESS"


@pytest.mark.asyncio
async def test_message_retained_when_publish_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeBundle:
        def model_dump_json(self, *, by_alias: bool, exclude_none: bool) -> str:
            return "{}"

    async def _fake_run(*args: Any, **kwargs: Any) -> Any:
        return FakeBundle()

    monkeypatch.setattr(mp, "run_integrity_review", _fake_run)

    request_store = FakeObjectStore(json_data=_payload_json())
    result_store = FakeObjectStore(has_result=False, fail_put=True)
    request_queue = FakeSqsClient()
    result_queue = FakeSqsClient()
    ctx = _make_ctx(
        request_store=request_store,
        result_store=result_store,
        request_queue=request_queue,
        result_queue=result_queue,
    )

    message = {"ReceiptHandle": "rh-3", "Body": _envelope_body()}
    await mp.handle_message(ctx, message)

    assert result_store.put_calls == []
    assert result_queue.sent == []
    assert request_queue.deleted == []


@pytest.mark.asyncio
async def test_review_id_mismatch_drops_message_without_running(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _unexpected_run(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("run_integrity_review must not be called on a mismatched review_id")

    monkeypatch.setattr(mp, "run_integrity_review", _unexpected_run)

    request_store = FakeObjectStore(json_data=_payload_json(review_id="review-mismatch"))
    result_store = FakeObjectStore(has_result=False)
    request_queue = FakeSqsClient()
    result_queue = FakeSqsClient()
    ctx = _make_ctx(
        request_store=request_store,
        result_store=result_store,
        request_queue=request_queue,
        result_queue=result_queue,
    )

    message = {"ReceiptHandle": "rh-4", "Body": _envelope_body(review_id="review-1")}
    await mp.handle_message(ctx, message)

    assert request_queue.deleted == ["rh-4"]
    assert result_queue.sent == []
