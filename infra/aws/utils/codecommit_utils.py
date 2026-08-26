"""CodeCommit source repository provisioning (the CI/CD pipeline's ``git push`` target)."""

from __future__ import annotations

from typing import Any

from botocore.exceptions import ClientError

from ..specs import CodeCommitRepositorySpec


def _get_repository(client: Any, name: str) -> dict[str, Any] | None:
    try:
        return client.get_repository(repositoryName=name)["repositoryMetadata"]
    except ClientError as error:
        if error.response.get("Error", {}).get("Code") == "RepositoryDoesNotExistException":
            return None
        raise


def repository_exists(client: Any, name: str) -> bool:
    return _get_repository(client, name) is not None


def ensure_repository(client: Any, spec: CodeCommitRepositorySpec, tags: dict[str, str]) -> str:
    """Create the repository if missing. Returns the repository ARN."""

    existing = _get_repository(client, spec.name)
    if existing is None:
        created = client.create_repository(
            repositoryName=spec.name,
            repositoryDescription=spec.description,
            tags=tags,
        )["repositoryMetadata"]
        return created["Arn"]

    client.update_repository_description(
        repositoryName=spec.name, repositoryDescription=spec.description
    )
    client.tag_resource(resourceArn=existing["Arn"], tags=tags)
    return existing["Arn"]


def delete_repository(client: Any, name: str) -> None:
    client.delete_repository(repositoryName=name)
