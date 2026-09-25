from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from .adapters.ai_usage_logger import AI_USAGE_STATUS_FAILURE, AI_USAGE_STATUS_SUCCESS


class GeminiClient(Protocol):
    async def generate(
        self,
        *,
        model: str,
        contents: Sequence[Mapping[str, Any] | Any],
        config: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...


class GeminiLimiter(Protocol):
    async def __aenter__(self) -> None: ...
    async def __aexit__(self, *args: Any) -> None: ...


class Logger(Protocol):
    def warning(self, message: str, **fields: Any) -> None: ...


class AiUsageLogger(Protocol):
    def record(
        self,
        *,
        step: str,
        model_name: str,
        model_version: str,
        status: str,
        latency_seconds: float,
        usage: Mapping[str, int] | None,
        extra_meta: Mapping[str, Any],
    ) -> None: ...


class AiDeps(Protocol):
    gemini: GeminiClient
    limiter: GeminiLimiter
    logger: Logger
    ai_usage_logger: AiUsageLogger


def _record_usage(
    deps: AiDeps,
    *,
    step: str,
    model: str,
    model_version: str,
    status: str,
    latency_seconds: float,
    usage: Mapping[str, int] | None,
    extra_meta: Mapping[str, Any],
) -> None:
    try:
        deps.ai_usage_logger.record(
            step=step,
            model_name=model,
            model_version=model_version,
            status=status,
            latency_seconds=latency_seconds,
            usage=usage,
            extra_meta=extra_meta,
        )
    except Exception as error:  # noqa: BLE001 - usage logging must never fail a review
        deps.logger.warning("gemini-call: ai usage log publish failed", error=str(error))


async def generate_and_log(
    deps: AiDeps,
    *,
    step: str,
    model: str,
    model_version: str,
    contents: Sequence[Mapping[str, Any] | Any],
    config: Mapping[str, Any],
    extra_meta: Mapping[str, Any],
) -> Mapping[str, Any]:
    start = time.perf_counter()
    try:
        async with deps.limiter:
            response = await deps.gemini.generate(model=model, contents=contents, config=config)
    except Exception:
        _record_usage(
            deps,
            step=step,
            model=model,
            model_version=model_version,
            status=AI_USAGE_STATUS_FAILURE,
            latency_seconds=time.perf_counter() - start,
            usage=None,
            extra_meta=extra_meta,
        )
        raise

    _record_usage(
        deps,
        step=step,
        model=model,
        model_version=model_version,
        status=AI_USAGE_STATUS_SUCCESS,
        latency_seconds=time.perf_counter() - start,
        usage=response.get("usage"),
        extra_meta=extra_meta,
    )
    return response

