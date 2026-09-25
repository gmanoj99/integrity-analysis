from __future__ import annotations

import asyncio
from typing import Any


class ReviewGeminiLimiter:
    def __init__(self, global_semaphore: asyncio.Semaphore, review_semaphore: asyncio.Semaphore) -> None:
        self._global = global_semaphore
        self._review = review_semaphore

    async def __aenter__(self) -> None:
        await self._review.acquire()
        try:
            await self._global.acquire()
        except BaseException:
            self._review.release()
            raise

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: Any,
    ) -> None:
        self._global.release()
        self._review.release()


class FairGeminiLimiter:
    def __init__(self, *, global_limit: int, per_review_limit: int) -> None:
        self._global = asyncio.Semaphore(global_limit)
        self._per_review_limit = per_review_limit

    def for_review(self) -> ReviewGeminiLimiter:
        return ReviewGeminiLimiter(self._global, asyncio.Semaphore(self._per_review_limit))
