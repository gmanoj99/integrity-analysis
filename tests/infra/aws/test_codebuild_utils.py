"""botocore Stubber coverage for CodeBuild project provisioning."""

from __future__ import annotations

from infra.aws.specs import CodeBuildProjectSpec
from infra.aws.utils import codebuild_utils

from .aws_stub import stubbed_client

PROJECT_NAME = "integrity-review-beta-build"
PROJECT_ARN = "arn:aws:codebuild:ap-south-1:111111111111:project/integrity-review-beta-build"


def _spec() -> CodeBuildProjectSpec:
    return CodeBuildProjectSpec(
        name=PROJECT_NAME,
        description="Build and push the worker image to ECR",
        service_role_arn="arn:aws:iam::111111111111:role/integrity-review-beta-codebuild",
        buildspec_path="infra/aws/cicd/buildspec.build.yml",
        image="aws/codebuild/amazonlinux2-x86_64-standard:5.0",
        compute_type="BUILD_GENERAL1_MEDIUM",
        privileged_mode=True,
        log_group_name="/aws/codebuild/integrity-review-beta-build",
        environment_variables={"AWS_REGION": "ap-south-1"},
    )


def test_project_exists_false_when_missing() -> None:
    client, stubber = stubbed_client("codebuild")
    stubber.add_response("batch_get_projects", {"projects": []}, {"names": [PROJECT_NAME]})
    with stubber:
        assert codebuild_utils.project_exists(client, PROJECT_NAME) is False


def test_ensure_project_creates_when_missing() -> None:
    client, stubber = stubbed_client("codebuild")
    stubber.add_response("batch_get_projects", {"projects": []})
    stubber.add_response("create_project", {"project": {"name": PROJECT_NAME, "arn": PROJECT_ARN}})

    with stubber:
        arn = codebuild_utils.ensure_project(client, _spec(), {"Project": "integrity-review"})

    assert arn == PROJECT_ARN
    stubber.assert_no_pending_responses()


def test_ensure_project_updates_when_existing() -> None:
    client, stubber = stubbed_client("codebuild")
    stubber.add_response(
        "batch_get_projects", {"projects": [{"name": PROJECT_NAME, "arn": PROJECT_ARN}]}
    )
    stubber.add_response("update_project", {"project": {"name": PROJECT_NAME, "arn": PROJECT_ARN}})

    with stubber:
        arn = codebuild_utils.ensure_project(client, _spec(), {"Project": "integrity-review"})

    assert arn == PROJECT_ARN
    stubber.assert_no_pending_responses()


def test_latest_build_status_returns_status_of_most_recent_build() -> None:
    client, stubber = stubbed_client("codebuild")
    build_id = f"{PROJECT_NAME}:abc123"
    stubber.add_response(
        "list_builds_for_project",
        {"ids": [build_id]},
        {"projectName": PROJECT_NAME, "sortOrder": "DESCENDING"},
    )
    stubber.add_response(
        "batch_get_builds",
        {"builds": [{"id": build_id, "buildStatus": "SUCCEEDED"}]},
        {"ids": [build_id]},
    )
    with stubber:
        assert codebuild_utils.latest_build_status(client, PROJECT_NAME) == "SUCCEEDED"
