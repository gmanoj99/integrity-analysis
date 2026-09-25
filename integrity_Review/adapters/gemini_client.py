from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Mapping, Sequence
from typing import Any

logger = logging.getLogger(__name__)

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


def _incomplete_reason(response: Any) -> str | None:
    blocked = getattr(getattr(response, "prompt_feedback", None), "block_reason", None)
    if blocked is not None:
        return f"prompt_blocked:{getattr(blocked, 'name', blocked)}"
    candidates = getattr(response, "candidates", None) or []
    finish = getattr(candidates[0], "finish_reason", None) if candidates else None
    if finish is None:
        return None
    name = str(getattr(finish, "name", finish))
    return None if name == "STOP" else name


def _usage_from_response(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage_metadata", None)
    return {
        "prompt_tk": getattr(usage, "prompt_token_count", None) or 0,
        "completion_tk": getattr(usage, "candidates_token_count", None) or 0,
        "cache_tk": getattr(usage, "cached_content_token_count", None) or 0,
        "reasoning_tk": getattr(usage, "thoughts_token_count", None) or 0,
    }


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
            usage = _usage_from_response(response)
            reason = _incomplete_reason(response)
            if reason is not None:
                logger.warning(
                    "gemini stopped early: model=%s finishReason=%s completion_tk=%d",
                    model,
                    reason,
                    usage["completion_tk"],
                )
            text = response.text or "{}"
            return {"text": text, "usage": usage}
