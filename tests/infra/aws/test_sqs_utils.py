"""botocore Stubber coverage for idempotent SQS queue provisioning."""

from __future__ import annotations

from botocore.stub import ANY

from infra.aws.specs import QueueSpec
from infra.aws.utils import sqs_utils

from .aws_stub import stubbed_client

QUEUE_URL = "https://sqs.ap-south-1.amazonaws.com/111111111111/integrity-review-beta-request"
QUEUE_ARN = "arn:aws:sqs:ap-south-1:111111111111:integrity-review-beta-request"


def test_ensure_queue_creates_new_queue_when_missing() -> None:
    client, stubber = stubbed_client("sqs")
    spec = QueueSpec(
        name="integrity-review-beta-request",
        kms_key_id="alias/integrity-review-beta-data",
        visibility_timeout_seconds=300,
        message_retention_seconds=345600,
        dlq_arn=None,
        max_receive_count=0,
        allowed_role_arns=(),
    )

    stubber.add_client_error(
        "get_queue_url",
        service_error_code="AWS.SimpleQueueService.NonExistentQueue",
        http_status_code=400,
    )
    stubber.add_response("create_queue", {"QueueUrl": QUEUE_URL}, ANY)

    with stubber:
        queue_url = sqs_utils.ensure_queue(client, spec, {"Project": "integrity-review"})

    assert queue_url == QUEUE_URL
    stubber.assert_no_pending_responses()


def test_ensure_queue_updates_existing_queue_and_applies_policy() -> None:
    client, stubber = stubbed_client("sqs")
    spec = QueueSpec(
        name="integrity-review-beta-request",
        kms_key_id="alias/integrity-review-beta-data",
        visibility_timeout_seconds=300,
        message_retention_seconds=345600,
        dlq_arn="arn:aws:sqs:ap-south-1:111111111111:integrity-review-beta-request-dlq",
        max_receive_count=5,
        allowed_role_arns=("arn:aws:iam::111111111111:role/integrity-review-beta-ecs-task",),
    )

    stubber.add_response("get_queue_url", {"QueueUrl": QUEUE_URL}, ANY)
    stubber.add_response("set_queue_attributes", {}, ANY)
    stubber.add_response("tag_queue", {}, ANY)
    stubber.add_response("get_queue_attributes", {"Attributes": {"QueueArn": QUEUE_ARN}}, ANY)
    stubber.add_response("set_queue_attributes", {}, ANY)

    with stubber:
        queue_url = sqs_utils.ensure_queue(client, spec, {"Project": "integrity-review"})

    assert queue_url == QUEUE_URL
    stubber.assert_no_pending_responses()


def test_queue_arn_reads_queue_arn_attribute() -> None:
    client, stubber = stubbed_client("sqs")
    stubber.add_response(
        "get_queue_attributes", {"Attributes": {"QueueArn": QUEUE_ARN}}, {"QueueUrl": QUEUE_URL, "AttributeNames": ["QueueArn"]}
    )
    with stubber:
        assert sqs_utils.queue_arn(client, QUEUE_URL) == QUEUE_ARN
