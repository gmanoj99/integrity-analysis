"""botocore Stubber coverage for ECR repository existence checks."""

from __future__ import annotations

from botocore.stub import ANY

from infra.aws.utils import ecr_utils

from .aws_stub import stubbed_client

REPOSITORY_NAME = "integrity-review-ecs-test-worker"


def test_repository_exists_true() -> None:
    client, stubber = stubbed_client("ecr")
    stubber.add_response(
        "describe_repositories",
        {"repositories": [{"repositoryName": REPOSITORY_NAME, "repositoryArn": "arn:aws:ecr:x"}]},
        ANY,
    )
    with stubber:
        assert ecr_utils.repository_exists(client, REPOSITORY_NAME) is True


def test_repository_exists_false_when_not_found() -> None:
    client, stubber = stubbed_client("ecr")
    stubber.add_client_error(
        "describe_repositories", service_error_code="RepositoryNotFoundException", http_status_code=400
    )
    with stubber:
        assert ecr_utils.repository_exists(client, REPOSITORY_NAME) is False


def test_get_latest_image_scan_findings_passes_through_tag() -> None:
    client, stubber = stubbed_client("ecr")
    stubber.add_response(
        "describe_image_scan_findings",
        {"imageScanStatus": {"status": "COMPLETE"}, "imageScanFindings": {"findings": []}},
        {"repositoryName": REPOSITORY_NAME, "imageId": {"imageTag": "abc123"}},
    )
    with stubber:
        result = ecr_utils.get_latest_image_scan_findings(
            client, repository_name=REPOSITORY_NAME, image_tag="abc123"
        )
    assert result["imageScanStatus"]["status"] == "COMPLETE"
