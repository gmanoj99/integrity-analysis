"""Task-global Gemini concurrency limit with a fair per-review sub-cap.

A single review with many chunks must not starve the other concurrent review of
Gemini slots, so each review gets its own sub-semaphore capped at a fair share of
the task-global limit, in addition to acquiring a task-global slot.
"""

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
    def __init__(self, *, global_limit: int, max_concurrent_reviews: int) -> None:
        self._global = asyncio.Semaphore(global_limit)
        self._per_review_limit = max(1, global_limit // max(1, max_concurrent_reviews))

    def for_review(self) -> ReviewGeminiLimiter:
        return ReviewGeminiLimiter(self._global, asyncio.Semaphore(self._per_review_limit))
