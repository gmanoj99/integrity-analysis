"""botocore Stubber coverage for KMS key/alias idempotency."""

from __future__ import annotations

from botocore.stub import ANY

from infra.aws.specs import KmsKeySpec
from infra.aws.utils import kms_utils

from .aws_stub import stubbed_client

ALIAS = "integrity-review-ecs-test-data"
KEY_ID = "1234abcd-12ab-34cd-56ef-1234567890ab"
KEY_ARN = f"arn:aws:kms:ap-south-1:111111111111:key/{KEY_ID}"


def test_key_arn_for_alias_returns_none_when_missing() -> None:
    client, stubber = stubbed_client("kms")
    stubber.add_response("list_aliases", {"Aliases": [], "Truncated": False}, ANY)
    with stubber:
        assert kms_utils.key_arn_for_alias(client, ALIAS) is None


def test_key_arn_for_alias_resolves_existing_alias() -> None:
    client, stubber = stubbed_client("kms")
    stubber.add_response(
        "list_aliases",
        {"Aliases": [{"AliasName": f"alias/{ALIAS}", "TargetKeyId": KEY_ID}], "Truncated": False},
        ANY,
    )
    stubber.add_response("describe_key", {"KeyMetadata": {"KeyId": KEY_ID, "Arn": KEY_ARN}}, ANY)
    with stubber:
        assert kms_utils.key_arn_for_alias(client, ALIAS) == KEY_ARN


def test_ensure_key_creates_key_and_alias_when_missing() -> None:
    client, stubber = stubbed_client("kms")
    spec = KmsKeySpec(alias=ALIAS, description="test key", key_policy={"Version": "2012-10-17", "Statement": []})

    stubber.add_response("list_aliases", {"Aliases": [], "Truncated": False}, ANY)
    stubber.add_response("create_key", {"KeyMetadata": {"KeyId": KEY_ID, "Arn": KEY_ARN}}, ANY)
    stubber.add_response("create_alias", {}, {"AliasName": f"alias/{ALIAS}", "TargetKeyId": KEY_ID})
    stubber.add_response("describe_key", {"KeyMetadata": {"KeyId": KEY_ID, "Arn": KEY_ARN}}, ANY)

    with stubber:
        arn = kms_utils.ensure_key(client, spec)

    assert arn == KEY_ARN
    stubber.assert_no_pending_responses()


def test_ensure_key_updates_policy_when_alias_exists() -> None:
    client, stubber = stubbed_client("kms")
    spec = KmsKeySpec(alias=ALIAS, description="test key", key_policy={"Version": "2012-10-17", "Statement": []})

    stubber.add_response(
        "list_aliases",
        {"Aliases": [{"AliasName": f"alias/{ALIAS}", "TargetKeyId": KEY_ID}], "Truncated": False},
        ANY,
    )
    stubber.add_response("put_key_policy", {}, ANY)
    stubber.add_response("describe_key", {"KeyMetadata": {"KeyId": KEY_ID, "Arn": KEY_ARN}}, ANY)

    with stubber:
        arn = kms_utils.ensure_key(client, spec)

    assert arn == KEY_ARN
    stubber.assert_no_pending_responses()
