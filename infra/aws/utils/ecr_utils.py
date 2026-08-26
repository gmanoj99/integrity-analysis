"""ECR repository provisioning: immutable tags, KMS encryption, scan-on-push."""

from __future__ import annotations

import json
from typing import Any

from botocore.exceptions import ClientError

from ..specs import RepositorySpec


def _describe_repository(client: Any, name: str) -> dict[str, Any] | None:
    try:
        return client.describe_repositories(repositoryNames=[name])["repositories"][0]
    except ClientError as error:
        if error.response.get("Error", {}).get("Code") == "RepositoryNotFoundException":
            return None
        raise


def repository_exists(client: Any, name: str) -> bool:
    return _describe_repository(client, name) is not None


def ensure_repository(client: Any, spec: RepositorySpec, tags: dict[str, str]) -> str:
    """Create (if missing) an immutable, KMS-encrypted, scan-on-push repository."""

    repository = _describe_repository(client, spec.name)
    if repository is None:
        repository = client.create_repository(
            repositoryName=spec.name,
            imageTagMutability="IMMUTABLE",
            imageScanningConfiguration={"scanOnPush": True},
            encryptionConfiguration={"encryptionType": "KMS", "kmsKey": spec.kms_key_arn},
            tags=[{"Key": key, "Value": value} for key, value in tags.items()],
        )["repository"]

    client.put_lifecycle_policy(
        repositoryName=spec.name,
        lifecyclePolicyText=json.dumps(
            {
                "rules": [
                    {
                        "rulePriority": 1,
                        "description": "Retain only the most recent tagged test images",
                        "selection": {
                            "tagStatus": "tagged",
                            "tagPrefixList": [""],
                            "countType": "imageCountMoreThan",
                            "countNumber": spec.max_tagged_images,
                        },
                        "action": {"type": "expire"},
                    }
                ]
            }
        ),
    )
    return repository["repositoryArn"]


def get_latest_image_scan_findings(client: Any, *, repository_name: str, image_tag: str) -> dict[str, Any]:
    return client.describe_image_scan_findings(
        repositoryName=repository_name, imageId={"imageTag": image_tag}
    )
