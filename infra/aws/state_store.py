"""In-memory deployment manifest for the isolated ECS test stack.

The state manifest records every resource this provisioner created (IDs,
ARNs, a config hash, and status) so a single ``apply`` can roll back only
what *this* run created. It is not written to S3.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


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
    def from_dict(cls, data: dict[str, Any]) -> DeploymentManifest:
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
