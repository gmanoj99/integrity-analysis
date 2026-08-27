"""Google Gemini client adapter."""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Mapping, Sequence
from typing import Any

logger = logging.getLogger(__name__)

# The concurrency limiter caps how many Gemini calls run at once, not the
# provider's requests-per-minute quota, so 429/RESOURCE_EXHAUSTED can still
# happen under load. Retry a bounded number of times with jittered backoff
# instead of failing the whole review on the first rate-limit response.
_RATE_LIMIT_STATUS = "RESOURCE_EXHAUSTED"
_RATE_LIMIT_HTTP_CODE = 429
_MAX_RETRY_ATTEMPTS = 3
_BASE_BACKOFF_SECONDS = 1.0
_MAX_BACKOFF_SECONDS = 4.0


def _is_rate_limited(error: Exception) -> bool:
    code = getattr(error, "code", None)
    status = getattr(error, "status", None)
    return code == _RATE_LIMIT_HTTP_CODE or status == _RATE_LIMIT_STATUS


def _backoff_seconds(attempt: int) -> float:
    capped_delay = min(_BASE_BACKOFF_SECONDS * (2**attempt), _MAX_BACKOFF_SECONDS)
    return random.uniform(0, capped_delay)


class GoogleGeminiClient:
    def __init__(self, api_key: str) -> None:
        try:
            from google import genai
        except ImportError as error:  # pragma: no cover - dependency error
            raise RuntimeError("google-genai is required for live analysis") from error
        self._client = genai.Client(api_key=api_key)

    async def generate(
        self,
        *,
        model: str,
        contents: Sequence[Mapping[str, Any] | Any],
        config: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        attempt = 0
        while True:
            try:
                response = await self._client.aio.models.generate_content(
                    model=model,
                    contents=list(contents),
                    config=dict(config),
                )
            except Exception as error:  # noqa: BLE001 - inspected below, re-raised if not rate-limited
                if not _is_rate_limited(error) or attempt == _MAX_RETRY_ATTEMPTS:
                    raise
                delay = _backoff_seconds(attempt)
                logger.warning(
                    "gemini rate-limited, retrying: model=%s attempt=%d delay=%.2fs",
                    model,
                    attempt + 1,
                    delay,
                )
                await asyncio.sleep(delay)
                attempt += 1
                continue
            text = response.text or "{}"
            return {"text": text}
