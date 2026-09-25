from __future__ import annotations

import os
from dataclasses import dataclass
from enum import StrEnum


class Stage(StrEnum):
    BETA = "beta"
    PROD = "prod"


STAGE_TO_S3_MEDIA_PREFIX = {
    Stage.BETA: "topin_beta",
    Stage.PROD: "topin_prod",
}


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is required")
    return value


def _require_stage() -> Stage:
    raw_stage = _require_env("STAGE")
    try:
        return Stage(raw_stage)
    except ValueError as error:
        allowed = ", ".join(stage.value for stage in Stage)
        raise RuntimeError(f"STAGE must be one of: {allowed}") from error


@dataclass(frozen=True, slots=True)
class WorkerConfig:
    storage_bucket: str
    request_queue_url: str
    response_queue_url: str
    aws_region: str
    gemini_api_key: str
    organization_id: str
    stage: Stage
    s3_media_prefix: str
    max_concurrent_reviews: int = 2
    gemini_task_limit: int = 24
    gemini_per_review_limit: int = 12
    visibility_timeout_seconds: int = 300
    heartbeat_interval_seconds: int = 90
    poll_wait_time_seconds: int = 10
    presign_expires_in_seconds: int = 3_600
    custom_ai_logs_group_name: str = "custom-ai-logs"
    custom_ai_logs_stream_name: str = Stage.BETA.value
    max_receive_count: int = 5
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> WorkerConfig:
        stage = _require_stage()
        s3_media_prefix = STAGE_TO_S3_MEDIA_PREFIX[stage]
        return cls(
            storage_bucket=_require_env("AWS_STORAGE_BUCKET_NAME"),
            request_queue_url=_require_env("REQUEST_QUEUE_URL"),
            response_queue_url=_require_env("RESPONSE_QUEUE_URL"),
            aws_region=os.environ.get("AWS_REGION", "ap-south-1"),
            gemini_api_key=_require_env("GEMINI_API_KEY"),
            organization_id=os.environ.get("ORGANIZATION_ID", "local"),
            stage=stage,
            s3_media_prefix=s3_media_prefix,
            max_concurrent_reviews=int(os.environ.get("MAX_CONCURRENT_REVIEWS", "2")),
            gemini_task_limit=int(os.environ.get("GEMINI_TASK_LIMIT", "24")),
            gemini_per_review_limit=int(os.environ.get("GEMINI_PER_REVIEW_LIMIT", "12")),
            visibility_timeout_seconds=int(os.environ.get("VISIBILITY_TIMEOUT_SECONDS", "300")),
            heartbeat_interval_seconds=int(os.environ.get("HEARTBEAT_INTERVAL_SECONDS", "90")),
            poll_wait_time_seconds=int(os.environ.get("POLL_WAIT_TIME_SECONDS", "10")),
            presign_expires_in_seconds=int(
                os.environ.get("PRESIGN_EXPIRES_IN_SECONDS", "3600")
            ),
            custom_ai_logs_group_name=os.environ.get(
                "CUSTOM_AI_LOGS_GROUP_NAME", "custom-ai-logs"
            ),
            custom_ai_logs_stream_name=os.environ.get(
                "CUSTOM_AI_LOGS_STREAM_NAME", stage.value
            ),
            max_receive_count=int(os.environ.get("MAX_RECEIVE_COUNT", "5")),
            log_level=os.environ.get("LOG_LEVEL", "INFO"),
        )

    def as_log_fields(self) -> dict[str, object]:
        return {
            "stage": self.stage.value,
            "aws_region": self.aws_region,
            "storage_bucket": self.storage_bucket,
            "s3_media_prefix": self.s3_media_prefix,
            "request_queue_url": self.request_queue_url,
            "response_queue_url": self.response_queue_url,
            "organization_id": self.organization_id,
            "max_concurrent_reviews": self.max_concurrent_reviews,
            "gemini_task_limit": self.gemini_task_limit,
            "gemini_per_review_limit": self.gemini_per_review_limit,
            "visibility_timeout_seconds": self.visibility_timeout_seconds,
            "heartbeat_interval_seconds": self.heartbeat_interval_seconds,
            "poll_wait_time_seconds": self.poll_wait_time_seconds,
            "max_receive_count": self.max_receive_count,
            "custom_ai_logs_group_name": self.custom_ai_logs_group_name,
            "custom_ai_logs_stream_name": self.custom_ai_logs_stream_name,
            "gemini_api_key_present": bool(self.gemini_api_key),
            "log_level": self.log_level,
        }
