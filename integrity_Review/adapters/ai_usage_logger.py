from __future__ import annotations

import json
import time
from collections.abc import Mapping
from typing import Any

import boto3

AI_USAGE_PROJECT_NAME = "TOPIN_BACKEND"
AI_USAGE_TEAM_NAME = "TECH"
AI_USAGE_GROUP_NAME = "TOPIN"
AI_USAGE_FEATURE_NAME = "ASSESSMENT_AI_ANALYSIS"

AI_USAGE_MODEL_PROVIDER = "GOOGLE_GEMINI"
AI_USAGE_MODEL_REGION = "GLOBAL"

AI_USAGE_STATUS_SUCCESS = "SUCCESS"
AI_USAGE_STATUS_FAILURE = "FAILURE"

STEP_CAMERA_PERCEPTION = "CAMERA_PERCEPTION"
STEP_SCREEN_PERCEPTION = "SCREEN_PERCEPTION"
STEP_SCREEN_CAMERA_PERCEPTION = "SCREEN_CAMERA_PERCEPTION"
STEP_DELIBERATION = "DELIBERATION"
STEP_SEB_LOG_ANALYSIS = "SEB_LOG_ANALYSIS"

_EMPTY_USAGE: Mapping[str, int] = {
    "prompt_tk": 0,
    "completion_tk": 0,
    "cache_tk": 0,
    "reasoning_tk": 0,
}


def _usage_event(
    *,
    step: str,
    model_name: str,
    model_version: str,
    status: str,
    latency_seconds: float,
    usage: Mapping[str, int] | None,
    meta: Mapping[str, Any],
) -> dict[str, Any]:
    resolved_usage = usage or _EMPTY_USAGE
    return {
        "event": "AI_USAGE_LOG",
        "body": {
            "model": {
                "provider": AI_USAGE_MODEL_PROVIDER,
                "name": model_name,
                "version": model_version,
                "region": AI_USAGE_MODEL_REGION,
            },
            "project": {
                "name": AI_USAGE_PROJECT_NAME,
                "feature": AI_USAGE_FEATURE_NAME,
                "step": step,
                "team": AI_USAGE_TEAM_NAME,
                "group": AI_USAGE_GROUP_NAME,
            },
            "response": {"status": status, "time": round(latency_seconds, 5)},
            "usage": {
                "llm": {
                    "prompt_tk": resolved_usage.get("prompt_tk", 0),
                    "completion_tk": resolved_usage.get("completion_tk", 0),
                },
                "misc": {
                    "cache_tk": resolved_usage.get("cache_tk", 0),
                    "reasoning_tk": resolved_usage.get("reasoning_tk", 0),
                },
            },
            "meta": dict(meta),
        },
    }


class CloudWatchAiUsageLogger:
    def __init__(
        self,
        *,
        review_meta: Mapping[str, Any],
        ensure_stream: Any,
        put_events: Any,
    ) -> None:
        self._review_meta = dict(review_meta)
        self._ensure_stream = ensure_stream
        self._put_events = put_events

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
    ) -> None:
        event = _usage_event(
            step=step,
            model_name=model_name,
            model_version=model_version,
            status=status,
            latency_seconds=latency_seconds,
            usage=usage,
            meta={**self._review_meta, **extra_meta},
        )
        self._ensure_stream()
        self._put_events(json.dumps(event, default=str))


class CloudWatchAiUsageLoggerFactory:
    def __init__(
        self,
        *,
        log_group_name: str,
        log_stream_name: str,
        region_name: str | None,
        client: Any | None = None,
    ) -> None:
        self._log_group_name = log_group_name
        self._log_stream_name = log_stream_name
        self._client = client or boto3.client("logs", region_name=region_name)
        self._stream_ready = False
        self._sequence_token: str | None = None

    def _ensure_stream(self) -> None:
        if self._stream_ready:
            return
        try:
            self._client.create_log_stream(
                logGroupName=self._log_group_name, logStreamName=self._log_stream_name
            )
        except self._client.exceptions.ResourceAlreadyExistsException:
            pass
        self._stream_ready = True

    def _put_events(self, message: str) -> None:
        log_events = [{"timestamp": int(time.time() * 1000), "message": message}]
        for _ in range(5):
            kwargs: dict[str, Any] = {
                "logGroupName": self._log_group_name,
                "logStreamName": self._log_stream_name,
                "logEvents": log_events,
            }
            if self._sequence_token is not None:
                kwargs["sequenceToken"] = self._sequence_token
            try:
                response = self._client.put_log_events(**kwargs)
            except self._client.exceptions.InvalidSequenceTokenException as error:
                self._sequence_token = error.response["expectedSequenceToken"]
                continue
            except self._client.exceptions.DataAlreadyAcceptedException as error:
                self._sequence_token = error.response.get("expectedSequenceToken")
                return
            self._sequence_token = response.get("nextSequenceToken")
            return
        raise RuntimeError("PutLogEvents failed after sequence-token retries")

    def __call__(
        self, *, review_id: str, org_assess_id: str, attempt_user_id: str
    ) -> CloudWatchAiUsageLogger:
        return CloudWatchAiUsageLogger(
            review_meta={
                "review_id": review_id,
                "org_assess_id": org_assess_id,
                "attempt_user_id": attempt_user_id,
            },
            ensure_stream=self._ensure_stream,
            put_events=self._put_events,
        )


class NullAiUsageLogger:
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
    ) -> None:
        return None
