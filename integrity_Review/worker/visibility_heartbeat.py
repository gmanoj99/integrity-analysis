"""Keep one in-flight SQS message's visibility timeout extended while it is processed."""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import Any, Protocol


class VisibilitySqsClient(Protocol):
    async def change_visibility(self, receipt_handle: str, visibility_timeout: int) -> None: ...


class HeartbeatLogger(Protocol):
    def info(self, message: str, **fields: Any) -> None: ...

    def warning(self, message: str, **fields: Any) -> None: ...


class VisibilityHeartbeat:
    """Async context manager: extends visibility every ``interval_seconds`` while open."""

    def __init__(
        self,
        sqs: VisibilitySqsClient,
        receipt_handle: str,
        *,
        interval_seconds: int,
        visibility_timeout_seconds: int,
        logger: HeartbeatLogger,
    ) -> None:
        self._sqs = sqs
        self._receipt_handle = receipt_handle
        self._interval_seconds = interval_seconds
        self._visibility_timeout_seconds = visibility_timeout_seconds
        self._logger = logger
        self._task: asyncio.Task[None] | None = None

    async def __aenter__(self) -> VisibilityHeartbeat:
        self._task = asyncio.ensure_future(self._loop())
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: Any,
    ) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    async def _loop(self) -> None:
        started = time.monotonic()
        extensions = 0
        while True:
            await asyncio.sleep(self._interval_seconds)
            try:
                await self._sqs.change_visibility(
                    self._receipt_handle, self._visibility_timeout_seconds
                )
            except Exception as error:  # noqa: BLE001 - best-effort heartbeat
                self._logger.warning("visibility-heartbeat: extend failed", error=error)
                continue
            extensions += 1
            # One line per interval is the only in-flight liveness signal for a
            # long review; without it a slow review looks like a hung task.
            self._logger.info(
                "visibility-heartbeat: visibility extended",
                extensions=extensions,
                elapsed_seconds=round(time.monotonic() - started, 1),
                visibility_timeout_seconds=self._visibility_timeout_seconds,
            )
