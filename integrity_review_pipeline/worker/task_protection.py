"""Refcounted ``ecs:UpdateTaskProtection`` so scale-in never kills an in-flight review.

Protection is enabled while at least one review is running and disabled once the
last one finishes. Resolves the task ARN from the ECS Task Metadata Endpoint v4
and no-ops (with a debug log) outside ECS, e.g. in local runs and tests.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Protocol

import httpx


class ProtectionLogger(Protocol):
    def debug(self, message: str, **fields: Any) -> None: ...

    def warning(self, message: str, **fields: Any) -> None: ...


def _resolve_task_arn() -> str | None:
    metadata_uri = os.environ.get("ECS_CONTAINER_METADATA_URI_V4")
    if not metadata_uri:
        return None
    try:
        response = httpx.get(f"{metadata_uri}/task", timeout=5.0)
        response.raise_for_status()
        task_arn = response.json().get("TaskARN")
        return task_arn if isinstance(task_arn, str) else None
    except Exception:  # noqa: BLE001 - metadata endpoint is best-effort
        return None


class TaskProtection:
    def __init__(
        self,
        *,
        cluster: str | None = None,
        client: Any | None = None,
        protection_duration_minutes: int = 60,
        logger: ProtectionLogger | None = None,
    ) -> None:
        self._cluster = cluster or os.environ.get("ECS_CLUSTER")
        self._client = client
        self._protection_duration_minutes = protection_duration_minutes
        self._logger = logger
        self._count = 0
        self._lock = asyncio.Lock()
        self._task_arn: str | None = None

    async def acquire(self) -> None:
        async with self._lock:
            self._count += 1
            if self._count == 1:
                await self._set_protected(True)

    async def release(self) -> None:
        async with self._lock:
            self._count = max(0, self._count - 1)
            if self._count == 0:
                await self._set_protected(False)

    async def _set_protected(self, protected: bool) -> None:
        if not self._cluster:
            if self._logger:
                self._logger.debug("task-protection: ECS_CLUSTER unset, skipping")
            return
        try:
            await asyncio.to_thread(self._update_sync, protected)
        except Exception as error:  # noqa: BLE001 - never let this block review processing
            if self._logger:
                self._logger.warning(
                    "task-protection: update failed", protected=protected, error=str(error)
                )

    def _update_sync(self, protected: bool) -> None:
        if self._task_arn is None:
            self._task_arn = _resolve_task_arn()
        if not self._task_arn:
            return
        client = self._client or self._boto3_client()
        kwargs: dict[str, Any] = {
            "cluster": self._cluster,
            "tasks": [self._task_arn],
            "protectionEnabled": protected,
        }
        if protected:
            kwargs["expiresInMinutes"] = self._protection_duration_minutes
        client.update_task_protection(**kwargs)

    @staticmethod
    def _boto3_client() -> Any:
        import boto3

        return boto3.client("ecs")
