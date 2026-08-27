"""Non-secret environment configuration for the isolated ECS worker stack.

Values live in a JSON file per environment (see
``infra/aws/config/beta.json``); secrets such as the Gemini API key are
never read from this file.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ALLOWED_ENVIRONMENTS = frozenset({"beta"})
DEFAULT_REGION = "ap-south-1"


class ConfigValidationError(ValueError):
    """Raised when the loaded configuration is missing fields or targets prod."""


@dataclass(frozen=True, slots=True)
class NetworkConfig:
    vpc_cidr_block: str
    public_subnet_cidrs: tuple[str, ...]
    private_subnet_cidrs: tuple[str, ...]
    availability_zones: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SizingConfig:
    task_cpu: str
    task_memory: str
    ephemeral_storage_gib: int
    desired_count: int
    max_concurrent_reviews: int
    gemini_task_limit: int
    gemini_per_review_limit: int
    min_task_count: int
    max_task_count: int
    scale_in_idle_periods: int


@dataclass(frozen=True, slots=True)
class RetentionConfig:
    log_retention_days: int
    bucket_expiration_days: int
    queue_message_retention_seconds: int
    ecr_max_tagged_images: int


@dataclass(frozen=True, slots=True)
class EnvironmentConfig:
    account_id: str
    region: str
    environment: str
    project: str
    owner: str
    cost_center: str
    stage: str
    organization_id: str
    storage_bucket_name: str
    storage_kms_key_arn: str | None
    network: NetworkConfig
    sizing: SizingConfig
    retention: RetentionConfig
    tags: dict[str, str]

    @property
    def resource_prefix(self) -> str:
        return f"{self.project}-{self.environment}"

    def resolved_tags(self) -> dict[str, str]:
        return {
            "Project": self.project,
            "Environment": self.environment,
            "Owner": self.owner,
            "CostCenter": self.cost_center,
            "ManagedBy": "infra.aws",
            **self.tags,
        }


def _require(data: dict[str, Any], key: str) -> Any:
    if key not in data:
        raise ConfigValidationError(f"config is missing required field: {key}")
    return data[key]


def _build_network(data: dict[str, Any]) -> NetworkConfig:
    return NetworkConfig(
        vpc_cidr_block=_require(data, "vpc_cidr_block"),
        public_subnet_cidrs=tuple(_require(data, "public_subnet_cidrs")),
        private_subnet_cidrs=tuple(_require(data, "private_subnet_cidrs")),
        availability_zones=tuple(_require(data, "availability_zones")),
    )


def _build_sizing(data: dict[str, Any]) -> SizingConfig:
    return SizingConfig(
        task_cpu=str(_require(data, "task_cpu")),
        task_memory=str(_require(data, "task_memory")),
        ephemeral_storage_gib=int(_require(data, "ephemeral_storage_gib")),
        desired_count=int(_require(data, "desired_count")),
        max_concurrent_reviews=int(_require(data, "max_concurrent_reviews")),
        gemini_task_limit=int(_require(data, "gemini_task_limit")),
        gemini_per_review_limit=int(_require(data, "gemini_per_review_limit")),
        min_task_count=int(_require(data, "min_task_count")),
        max_task_count=int(_require(data, "max_task_count")),
        scale_in_idle_periods=int(data.get("scale_in_idle_periods", 5)),
    )


def _build_retention(data: dict[str, Any]) -> RetentionConfig:
    return RetentionConfig(
        log_retention_days=int(_require(data, "log_retention_days")),
        bucket_expiration_days=int(_require(data, "bucket_expiration_days")),
        queue_message_retention_seconds=int(_require(data, "queue_message_retention_seconds")),
        ecr_max_tagged_images=int(_require(data, "ecr_max_tagged_images")),
    )


def _validate_not_production(config: EnvironmentConfig, known_prod_account_ids: frozenset[str]) -> None:
    if config.environment not in ALLOWED_ENVIRONMENTS:
        raise ConfigValidationError(
            f"this provisioner only manages environments {sorted(ALLOWED_ENVIRONMENTS)!r}, "
            f"got {config.environment!r}"
        )
    if config.stage.lower() == "prod":
        raise ConfigValidationError("stage must not be 'prod' in the isolated test stack")
    if config.account_id in known_prod_account_ids:
        raise ConfigValidationError(
            f"account_id {config.account_id!r} is a known production account"
        )


def load_environment_config(
    config_path: str | Path,
    *,
    overrides: dict[str, Any] | None = None,
    known_prod_account_ids: frozenset[str] = frozenset(),
) -> EnvironmentConfig:
    """Load and validate the JSON config, applying CLI overrides such as --account-id."""

    raw = json.loads(Path(config_path).read_text())
    raw.update(overrides or {})

    config = EnvironmentConfig(
        account_id=str(_require(raw, "account_id")),
        region=raw.get("region", DEFAULT_REGION),
        environment=raw.get("environment", "beta"),
        project=_require(raw, "project"),
        owner=_require(raw, "owner"),
        cost_center=_require(raw, "cost_center"),
        stage=raw.get("stage", "beta"),
        organization_id=raw.get("organization_id", "local"),
        storage_bucket_name=_require(raw, "storage_bucket_name"),
        storage_kms_key_arn=raw.get("storage_kms_key_arn"),
        network=_build_network(_require(raw, "network")),
        sizing=_build_sizing(_require(raw, "sizing")),
        retention=_build_retention(_require(raw, "retention")),
        tags=dict(raw.get("tags", {})),
    )
    _validate_not_production(config, known_prod_account_ids)
    return config
