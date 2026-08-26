"""botocore Stubber coverage for the CodeCommit source repository provisioning."""

from __future__ import annotations

from infra.aws.specs import CodeCommitRepositorySpec
from infra.aws.utils import codecommit_utils

from .aws_stub import stubbed_client

REPOSITORY_NAME = "integrity-review-beta-source"
REPOSITORY_ARN = "arn:aws:codecommit:ap-south-1:111111111111:integrity-review-beta-source"


def _spec() -> CodeCommitRepositorySpec:
    return CodeCommitRepositorySpec(
        name=REPOSITORY_NAME,
        description="Integrity review beta worker: image source + provisioner config",
        default_branch="main",
    )


def test_repository_exists_false_when_not_found() -> None:
    client, stubber = stubbed_client("codecommit")
    stubber.add_client_error(
        "get_repository", service_error_code="RepositoryDoesNotExistException", http_status_code=400
    )
    with stubber:
        assert codecommit_utils.repository_exists(client, REPOSITORY_NAME) is False


def test_ensure_repository_creates_when_missing() -> None:
    client, stubber = stubbed_client("codecommit")
    stubber.add_client_error(
        "get_repository", service_error_code="RepositoryDoesNotExistException", http_status_code=400
    )
    stubber.add_response(
        "create_repository",
        {"repositoryMetadata": {"repositoryName": REPOSITORY_NAME, "Arn": REPOSITORY_ARN}},
        {
            "repositoryName": REPOSITORY_NAME,
            "repositoryDescription": _spec().description,
            "tags": {"Project": "integrity-review"},
        },
    )

    with stubber:
        arn = codecommit_utils.ensure_repository(client, _spec(), {"Project": "integrity-review"})

    assert arn == REPOSITORY_ARN
    stubber.assert_no_pending_responses()


def test_ensure_repository_updates_description_and_tags_when_existing() -> None:
    client, stubber = stubbed_client("codecommit")
    stubber.add_response(
        "get_repository",
        {"repositoryMetadata": {"repositoryName": REPOSITORY_NAME, "Arn": REPOSITORY_ARN}},
    )
    stubber.add_response(
        "update_repository_description",
        {},
        {"repositoryName": REPOSITORY_NAME, "repositoryDescription": _spec().description},
    )
    stubber.add_response(
        "tag_resource", {}, {"resourceArn": REPOSITORY_ARN, "tags": {"Project": "integrity-review"}}
    )

    with stubber:
        arn = codecommit_utils.ensure_repository(client, _spec(), {"Project": "integrity-review"})

    assert arn == REPOSITORY_ARN
    stubber.assert_no_pending_responses()
