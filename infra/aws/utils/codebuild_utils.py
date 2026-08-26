"""CodeBuild project provisioning for the image-build and provisioner-deploy stages."""

from __future__ import annotations

from typing import Any

from ..specs import CodeBuildProjectSpec


def _describe_project(client: Any, name: str) -> dict[str, Any] | None:
    projects = client.batch_get_projects(names=[name])["projects"]
    return projects[0] if projects else None


def project_exists(client: Any, name: str) -> bool:
    return _describe_project(client, name) is not None


def _project_kwargs(spec: CodeBuildProjectSpec) -> dict[str, Any]:
    return {
        "name": spec.name,
        "description": spec.description,
        "source": {"type": "CODEPIPELINE", "buildspec": spec.buildspec_path},
        "artifacts": {"type": "CODEPIPELINE"},
        "environment": {
            "type": "LINUX_CONTAINER",
            "image": spec.image,
            "computeType": spec.compute_type,
            "privilegedMode": spec.privileged_mode,
            "environmentVariables": [
                {"name": key, "value": value, "type": "PLAINTEXT"}
                for key, value in sorted(spec.environment_variables.items())
            ],
        },
        "serviceRole": spec.service_role_arn,
        "logsConfig": {
            "cloudWatchLogs": {"status": "ENABLED", "groupName": spec.log_group_name}
        },
    }


def ensure_project(client: Any, spec: CodeBuildProjectSpec, tags: dict[str, str]) -> str:
    """Create or update the CodeBuild project. Returns the project ARN."""

    kwargs = _project_kwargs(spec)
    tag_list = [{"key": key, "value": value} for key, value in tags.items()]

    if _describe_project(client, spec.name) is None:
        created = client.create_project(tags=tag_list, **kwargs)["project"]
        return created["arn"]

    updated = client.update_project(**kwargs)["project"]
    return updated["arn"]


def delete_project(client: Any, name: str) -> None:
    client.delete_project(name=name)


def latest_build_status(client: Any, project_name: str) -> str | None:
    build_ids = client.list_builds_for_project(projectName=project_name, sortOrder="DESCENDING")[
        "ids"
    ]
    if not build_ids:
        return None
    builds = client.batch_get_builds(ids=[build_ids[0]])["builds"]
    return builds[0]["buildStatus"] if builds else None
