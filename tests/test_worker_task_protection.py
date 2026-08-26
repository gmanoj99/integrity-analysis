"""Tests for refcounted ``ecs:UpdateTaskProtection`` handling."""

from __future__ import annotations

from typing import Any

import pytest

from integrity_review_pipeline.worker.task_protection import TaskProtection

from .worker_fakes import RecordingLogger


class FakeEcsClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def update_task_protection(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)


class FakeMetadataResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, str]:
        return {"TaskARN": "arn:aws:ecs:ap-south-1:123456789012:task/ecs-test/abc123"}


@pytest.mark.asyncio
async def test_noops_without_cluster_configured() -> None:
    protection = TaskProtection(cluster=None, logger=RecordingLogger())

    await protection.acquire()
    await protection.acquire()
    await protection.release()
    await protection.release()

    # No cluster means _set_protected returns before touching any boto3 client.


@pytest.mark.asyncio
async def test_enables_once_and_disables_after_last_release(monkeypatch) -> None:
    monkeypatch.setenv("ECS_CONTAINER_METADATA_URI_V4", "http://169.254.170.2/v4/abc123")
    monkeypatch.setattr(
        "integrity_review_pipeline.worker.task_protection.httpx.get",
        lambda *_args, **_kwargs: FakeMetadataResponse(),
    )
    client = FakeEcsClient()
    protection = TaskProtection(
        cluster="ecs-test",
        client=client,
        protection_duration_minutes=15,
        logger=RecordingLogger(),
    )

    await protection.acquire()
    await protection.acquire()
    await protection.release()

    assert len(client.calls) == 1
    assert client.calls[0]["protectionEnabled"] is True
    assert client.calls[0]["expiresInMinutes"] == 15

    await protection.release()

    assert len(client.calls) == 2
    assert client.calls[1]["protectionEnabled"] is False
    assert "expiresInMinutes" not in client.calls[1]


@pytest.mark.asyncio
async def test_update_failure_is_logged_and_swallowed(monkeypatch) -> None:
    monkeypatch.setenv("ECS_CONTAINER_METADATA_URI_V4", "http://169.254.170.2/v4/abc123")
    monkeypatch.setattr(
        "integrity_review_pipeline.worker.task_protection.httpx.get",
        lambda *_args, **_kwargs: FakeMetadataResponse(),
    )

    class RaisingEcsClient:
        def update_task_protection(self, **_kwargs: Any) -> None:
            raise RuntimeError("ecs:UpdateTaskProtection denied")

    logger = RecordingLogger()
    protection = TaskProtection(cluster="ecs-test", client=RaisingEcsClient(), logger=logger)

    await protection.acquire()

    assert logger.warnings
