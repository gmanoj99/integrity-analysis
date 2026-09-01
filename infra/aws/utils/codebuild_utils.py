"""Read-only lookups against CodeBuild projects this tool does not provision."""

from __future__ import annotations

from typing import Any


class CodeBuildProjectNotFoundError(RuntimeError):
    """Raised when the named CodeBuild project does not exist in this account."""


def get_project_service_role_arn(client: Any, project_name: str) -> str:
    """Return the execution (service) role ARN of an already-provisioned CodeBuild project.

    The project and its role are created and owned outside this repo (see
    ``cicd/buildspec.manual.yml``); this only reads the role that already
    exists so a deploy policy can be attached to it.
    """

    projects = client.batch_get_projects(names=[project_name])["projects"]
    if not projects:
        raise CodeBuildProjectNotFoundError(f"CodeBuild project not found: {project_name!r}")
    return projects[0]["serviceRole"]
