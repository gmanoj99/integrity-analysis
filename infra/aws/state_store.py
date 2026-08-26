"""Encrypted S3 deployment manifest for the isolated ECS test stack.

The state manifest records every resource this provisioner created (IDs,
ARNs, a config hash, and status) so ``apply``/``destroy`` never depend only
on local files, and so ``destroy`` only removes what this tool created.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any

from .utils.s3_utils import get_json_object, put_json_object

STATE_OBJECT_KEY_TEMPLATE = "state/{resource_prefix}/manifest.json"


@dataclass
class ResourceRecord:
    resource_type: str
    identifier: str
    arn: str | None
    config_hash: str
    status: str  # "created" | "failed" | "destroyed"
    managed_by_tool: bool  # True if this provisioner created it (eligible for destroy)


@dataclass
class DeploymentManifest:
    resource_prefix: str
    config_hash: str
    resources: dict[str, ResourceRecord] = field(default_factory=dict)
    updated_at_epoch_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "resource_prefix": self.resource_prefix,
            "config_hash": self.config_hash,
            "updated_at_epoch_seconds": self.updated_at_epoch_seconds,
            "resources": {
                key: {
                    "resource_type": record.resource_type,
                    "identifier": record.identifier,
                    "arn": record.arn,
                    "config_hash": record.config_hash,
                    "status": record.status,
                    "managed_by_tool": record.managed_by_tool,
                }
                for key, record in self.resources.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DeploymentManifest":
        return cls(
            resource_prefix=data["resource_prefix"],
            config_hash=data["config_hash"],
            updated_at_epoch_seconds=data.get("updated_at_epoch_seconds", 0.0),
            resources={
                key: ResourceRecord(**record) for key, record in data.get("resources", {}).items()
            },
        )


def config_hash(config_dict: dict[str, Any]) -> str:
    canonical = json.dumps(config_dict, sort_keys=True).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()[:16]


class StateStore:
    def __init__(self, *, s3_client: Any, state_bucket: str, kms_key_arn: str) -> None:
        self._s3 = s3_client
        self._state_bucket = state_bucket
        self._kms_key_arn = kms_key_arn

    def _state_key(self, resource_prefix: str) -> str:
        return STATE_OBJECT_KEY_TEMPLATE.format(resource_prefix=resource_prefix)

    def load(self, resource_prefix: str, *, config_hash_value: str) -> DeploymentManifest:
        raw = get_json_object(self._s3, bucket=self._state_bucket, key=self._state_key(resource_prefix))
        if raw is None:
            return DeploymentManifest(resource_prefix=resource_prefix, config_hash=config_hash_value)
        return DeploymentManifest.from_dict(raw)

    def save(self, manifest: DeploymentManifest) -> None:
        manifest.updated_at_epoch_seconds = time.time()
        put_json_object(
            self._s3,
            bucket=self._state_bucket,
            key=self._state_key(manifest.resource_prefix),
            body=manifest.to_dict(),
            kms_key_arn=self._kms_key_arn,
        )
