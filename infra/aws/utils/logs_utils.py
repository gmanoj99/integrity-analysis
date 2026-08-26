"""Encrypted CloudWatch Logs group provisioning and worker log inspection."""

from __future__ import annotations

from typing import Any

from botocore.exceptions import ClientError

from ..specs import LogGroupSpec


def ensure_log_group(client: Any, spec: LogGroupSpec) -> str:
    """Create the log group if missing and (re)apply retention.

    Deliberately untagged: ``CreateLogGroup`` only requires
    ``logs:TagResource`` in the caller's IAM policy when a non-empty
    ``tags`` map is supplied, and the state manifest (not AWS-side tags)
    tracks resource ownership for ``status``/``destroy``.
    """

    try:
        client.create_log_group(logGroupName=spec.name, kmsKeyId=spec.kms_key_arn)
    except ClientError as error:
        if error.response.get("Error", {}).get("Code") != "ResourceAlreadyExistsException":
            raise
    client.put_retention_policy(logGroupName=spec.name, retentionInDays=spec.retention_days)
    described = client.describe_log_groups(logGroupNamePrefix=spec.name)["logGroups"]
    match = next(group for group in described if group["logGroupName"] == spec.name)
    return match["arn"].removesuffix(":*")


def tail_review_logs(client: Any, *, log_group_name: str, review_id: str, start_time_ms: int) -> list[str]:
    """Fetch the ordered log lines mentioning ``review_id`` since ``start_time_ms``."""

    query_id = client.start_query(
        logGroupName=log_group_name,
        startTime=start_time_ms,
        endTime=start_time_ms + 24 * 60 * 60 * 1000,
        queryString=(
            f'fields @timestamp, @message | filter @message like /{review_id}/ | sort @timestamp asc'
        ),
    )["queryId"]
    results = client.get_query_results(queryId=query_id)
    return [
        next(field["value"] for field in row if field["field"] == "@message")
        for row in results.get("results", [])
    ]
