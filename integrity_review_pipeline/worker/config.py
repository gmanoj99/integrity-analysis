"""Worker environment configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is required")
    return value


@dataclass(frozen=True, slots=True)
class WorkerConfig:
    media_bucket: str
    request_bucket: str
    result_bucket: str
    request_queue_url: str
    result_queue_url: str
    aws_region: str
    gemini_api_key: str
    organization_id: str
    stage: str = "beta"
    max_concurrent_reviews: int = 2
    gemini_task_limit: int = 12
    visibility_timeout_seconds: int = 300
    heartbeat_interval_seconds: int = 90
    poll_wait_time_seconds: int = 20
    presign_expires_in_seconds: int = 3_600

    @classmethod
    def from_env(cls) -> WorkerConfig:
        return cls(
            media_bucket=_require_env("MEDIA_BUCKET"),
            request_bucket=_require_env("REQUEST_BUCKET"),
            result_bucket=_require_env("RESULT_BUCKET"),
            request_queue_url=_require_env("REQUEST_QUEUE_URL"),
            result_queue_url=_require_env("RESULT_QUEUE_URL"),
            aws_region=os.environ.get("AWS_REGION", "us-east-1"),
            gemini_api_key=_require_env("GEMINI_API_KEY"),
            organization_id=os.environ.get("ORGANIZATION_ID", "local"),
            stage=os.environ.get("STAGE", "prod"),
            max_concurrent_reviews=int(os.environ.get("MAX_CONCURRENT_REVIEWS", "2")),
            gemini_task_limit=int(os.environ.get("GEMINI_TASK_LIMIT", "12")),
            visibility_timeout_seconds=int(os.environ.get("VISIBILITY_TIMEOUT_SECONDS", "300")),
            heartbeat_interval_seconds=int(os.environ.get("HEARTBEAT_INTERVAL_SECONDS", "90")),
            poll_wait_time_seconds=int(os.environ.get("POLL_WAIT_TIME_SECONDS", "20")),
            presign_expires_in_seconds=int(
                os.environ.get("PRESIGN_EXPIRES_IN_SECONDS", "3600")
            ),
        )
