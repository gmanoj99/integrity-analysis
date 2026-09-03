"""Unencrypted SQS queues with redrive-to-DLQ and an approved-principal policy."""

from __future__ import annotations

import json
from typing import Any

from botocore.exceptions import ClientError

from ..specs import QueueSpec

LONG_POLL_WAIT_TIME_SECONDS = "10"


def _queue_url(client: Any, name: str) -> str | None:
    try:
        return client.get_queue_url(QueueName=name)["QueueUrl"]
    except ClientError as error:
        if error.response.get("Error", {}).get("Code") == "AWS.SimpleQueueService.NonExistentQueue":
            return None
        raise


def queue_arn(client: Any, queue_url: str) -> str:
    return client.get_queue_attributes(QueueUrl=queue_url, AttributeNames=["QueueArn"])[
        "Attributes"
    ]["QueueArn"]


def ensure_queue(client: Any, spec: QueueSpec, tags: dict[str, str]) -> str:
    """Create (if missing) and configure the queue. Returns the queue URL.

    ``KmsMasterKeyId`` is deliberately handled differently per branch:
    ``CreateQueue`` rejects an empty string for it (``InvalidAttributeValue``)
    -- an unencrypted new queue must simply omit the attribute. An
    *existing* queue previously encrypted with a customer KMS key, on the
    other hand, only drops that encryption when ``SetQueueAttributes`` is
    called with ``KmsMasterKeyId=""`` explicitly; omitting it there would
    leave the old key in place, since SQS only touches attributes it is
    given.
    """

    attributes: dict[str, str] = {
        "VisibilityTimeout": str(spec.visibility_timeout_seconds),
        "MessageRetentionPeriod": str(spec.message_retention_seconds),
        "ReceiveMessageWaitTimeSeconds": LONG_POLL_WAIT_TIME_SECONDS,
    }
    if spec.dlq_arn:
        attributes["RedrivePolicy"] = json.dumps(
            {"deadLetterTargetArn": spec.dlq_arn, "maxReceiveCount": spec.max_receive_count}
        )

    queue_url = _queue_url(client, spec.name)
    if queue_url is None:
        queue_url = client.create_queue(
            QueueName=spec.name,
            Attributes=attributes,
            tags=tags,
        )["QueueUrl"]
    else:
        client.set_queue_attributes(
            QueueUrl=queue_url, Attributes={**attributes, "KmsMasterKeyId": ""}
        )
        client.tag_queue(QueueUrl=queue_url, Tags=tags)

    if spec.allowed_role_arns:
        from ..policies import sqs_queue_policy

        arn = queue_arn(client, queue_url)
        policy = sqs_queue_policy(queue_arn=arn, allowed_role_arns=list(spec.allowed_role_arns))
        client.set_queue_attributes(QueueUrl=queue_url, Attributes={"Policy": json.dumps(policy)})
    return queue_url
