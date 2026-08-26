"""Shared in-memory fakes for worker-layer tests."""

from __future__ import annotations

from typing import Any


class NoopLogger:
    def debug(self, message: str, **fields: Any) -> None: ...

    def info(self, message: str, **fields: Any) -> None: ...

    def warning(self, message: str, **fields: Any) -> None: ...

    def error(self, message: str, **fields: Any) -> None: ...


class RecordingLogger(NoopLogger):
    def __init__(self) -> None:
        self.errors: list[tuple[str, dict[str, Any]]] = []
        self.warnings: list[tuple[str, dict[str, Any]]] = []

    def warning(self, message: str, **fields: Any) -> None:
        self.warnings.append((message, fields))

    def error(self, message: str, **fields: Any) -> None:
        self.errors.append((message, fields))


class FakeObjectStore:
    """Fake ``ObjectStoreWithHead`` covering the request/result/media roles."""

    def __init__(
        self,
        *,
        json_by_ref: dict[str, Any] | None = None,
        existing_result: bool = False,
        put_raises: bool = False,
    ) -> None:
        self._json_by_ref = json_by_ref or {}
        self._existing_result = existing_result
        self._put_raises = put_raises
        self.put_calls: list[tuple[str, bytes, str]] = []

    async def get_json(self, ref: str) -> Any:
        return self._json_by_ref[ref]

    async def get_bytes(self, ref: str) -> bytes:
        return b"[]"

    async def head_object(self, ref: str) -> bool:
        return self._existing_result

    async def put_object(
        self, ref: str, body: bytes, *, content_type: str = "application/json"
    ) -> None:
        if self._put_raises:
            raise RuntimeError("s3 put_object failed")
        self.put_calls.append((ref, body, content_type))

    async def media_uri_for(self, chunk: Any) -> str:
        return f"https://media.example/{getattr(chunk, 'source_ref', 'unknown')}"


class FakeSqsClient:
    """Fake ``WorkerSqsClient`` for both the request and result queues."""

    def __init__(self, *, send_raises: bool = False) -> None:
        self._send_raises = send_raises
        self.deleted: list[str] = []
        self.sent: list[str] = []
        self.visibility_changes: list[tuple[str, int]] = []

    async def delete(self, receipt_handle: str) -> None:
        self.deleted.append(receipt_handle)

    async def change_visibility(self, receipt_handle: str, visibility_timeout: int) -> None:
        self.visibility_changes.append((receipt_handle, visibility_timeout))

    async def send(self, body: str) -> None:
        if self._send_raises:
            raise RuntimeError("sqs send_message failed")
        self.sent.append(body)


class FailingChangeVisibilitySqsClient(FakeSqsClient):
    async def change_visibility(self, receipt_handle: str, visibility_timeout: int) -> None:
        raise RuntimeError("sqs change_message_visibility failed")


class FakeReviewLimiter:
    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, *_args: object) -> None:
        return None


class FakeFairLimiter:
    def for_review(self) -> FakeReviewLimiter:
        return FakeReviewLimiter()


class FakeTaskProtection:
    def __init__(self) -> None:
        self.acquired = 0
        self.released = 0

    async def acquire(self) -> None:
        self.acquired += 1

    async def release(self) -> None:
        self.released += 1


class FakeBundle:
    """Stand-in for ``EvidenceBundle`` exposing only what the processor calls."""

    def __init__(self, payload: dict[str, Any] | None = None) -> None:
        self._payload = payload or {"ok": True}

    def model_dump_json(self, *, by_alias: bool, exclude_none: bool) -> str:
        import json

        return json.dumps(self._payload)
