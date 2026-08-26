"""Config loading and the prod/environment guard rails."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from infra.aws.config import ConfigValidationError, load_environment_config

_BASE_CONFIG = {
    "account_id": "111111111111",
    "project": "integrity-review",
    "owner": "platform-team",
    "cost_center": "eng-test",
    "network": {
        "vpc_cidr_block": "10.90.0.0/16",
        "public_subnet_cidrs": ["10.90.0.0/24", "10.90.1.0/24"],
        "private_subnet_cidrs": ["10.90.10.0/24", "10.90.11.0/24"],
        "availability_zones": ["ap-south-1a", "ap-south-1b"],
    },
    "sizing": {
        "task_cpu": "2048",
        "task_memory": "4096",
        "ephemeral_storage_gib": 20,
        "desired_count": 0,
        "max_concurrent_reviews": 2,
        "gemini_task_limit": 24,
        "min_task_count": 0,
        "max_task_count": 2,
        "scale_in_idle_periods": 5,
    },
    "retention": {
        "log_retention_days": 14,
        "bucket_expiration_days": 7,
        "queue_message_retention_seconds": 345600,
        "ecr_max_tagged_images": 10,
    },
}


def _write_config(tmp_path: Path, overrides: dict | None = None) -> Path:
    data = {**_BASE_CONFIG, **(overrides or {})}
    config_path = tmp_path / "ecs-test.json"
    config_path.write_text(json.dumps(data))
    return config_path


def test_loads_valid_config_with_defaults(tmp_path: Path) -> None:
    config = load_environment_config(_write_config(tmp_path))

    assert config.region == "ap-south-1"
    assert config.environment == "ecs-test"
    assert config.resource_prefix == "integrity-review-ecs-test"
    assert config.resolved_tags()["Owner"] == "platform-team"


def test_cli_overrides_take_precedence(tmp_path: Path) -> None:
    config = load_environment_config(
        _write_config(tmp_path), overrides={"account_id": "222222222222", "region": "us-west-2"}
    )

    assert config.account_id == "222222222222"
    assert config.region == "us-west-2"


def test_rejects_unknown_environment(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, {"environment": "staging"})

    with pytest.raises(ConfigValidationError):
        load_environment_config(config_path)


def test_accepts_beta_environment(tmp_path: Path) -> None:
    config = load_environment_config(
        _write_config(tmp_path, {"environment": "beta", "stage": "beta"})
    )

    assert config.environment == "beta"
    assert config.resource_prefix == "integrity-review-beta"


def test_rejects_prod_stage(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, {"stage": "prod"})

    with pytest.raises(ConfigValidationError):
        load_environment_config(config_path)


def test_rejects_known_production_account(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)

    with pytest.raises(ConfigValidationError):
        load_environment_config(
            config_path, known_prod_account_ids=frozenset({"111111111111"})
        )


def test_missing_required_field_raises(tmp_path: Path) -> None:
    data = dict(_BASE_CONFIG)
    del data["owner"]
    config_path = tmp_path / "ecs-test.json"
    config_path.write_text(json.dumps(data))

    with pytest.raises(ConfigValidationError):
        load_environment_config(config_path)
