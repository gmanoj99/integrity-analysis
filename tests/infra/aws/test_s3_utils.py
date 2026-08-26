"""botocore Stubber coverage for the S3 idempotency and hardening helpers."""

from __future__ import annotations

from botocore.stub import ANY

from infra.aws.specs import BucketSpec
from infra.aws.utils import s3_utils

from .aws_stub import stubbed_client

BUCKET_NAME = "integrity-review-ecs-test-media-111111111111"
KMS_KEY_ARN = "arn:aws:kms:ap-south-1:111111111111:key/abcd1234"


def test_bucket_exists_true_on_success() -> None:
    client, stubber = stubbed_client("s3")
    stubber.add_response("head_bucket", {}, {"Bucket": BUCKET_NAME})
    with stubber:
        assert s3_utils.bucket_exists(client, BUCKET_NAME) is True


def test_bucket_exists_false_on_404() -> None:
    client, stubber = stubbed_client("s3")
    stubber.add_client_error("head_bucket", service_error_code="404", http_status_code=404)
    with stubber:
        assert s3_utils.bucket_exists(client, BUCKET_NAME) is False


def test_ensure_bucket_is_idempotent_and_reapplies_hardening() -> None:
    """When the bucket already exists, ensure_bucket must not call create_bucket."""

    client, stubber = stubbed_client("s3")
    spec = BucketSpec(
        name=BUCKET_NAME,
        region="ap-south-1",
        kms_key_arn=KMS_KEY_ARN,
        expiration_days=7,
        allowed_role_arns=("arn:aws:iam::111111111111:role/integrity-review-ecs-test-ecs-task",),
    )

    stubber.add_response("head_bucket", {}, {"Bucket": BUCKET_NAME})
    stubber.add_response("put_public_access_block", {}, ANY)
    stubber.add_response("put_bucket_ownership_controls", {}, ANY)
    stubber.add_response("put_bucket_encryption", {}, ANY)
    stubber.add_response("put_bucket_versioning", {}, ANY)
    stubber.add_response("put_bucket_policy", {}, ANY)
    stubber.add_response("put_bucket_lifecycle_configuration", {}, ANY)
    stubber.add_response("put_bucket_tagging", {}, ANY)

    with stubber:
        arn = s3_utils.ensure_bucket(client, spec, {"Project": "integrity-review"})

    assert arn == f"arn:aws:s3:::{BUCKET_NAME}"
    stubber.assert_no_pending_responses()


def test_put_json_object_uses_kms_encryption() -> None:
    client, stubber = stubbed_client("s3")
    body = {"resource_prefix": "integrity-review-ecs-test", "resources": {}}

    stubber.add_response("put_object", {}, ANY)
    with stubber:
        s3_utils.put_json_object(client, bucket=BUCKET_NAME, key="state/manifest.json", body=body, kms_key_arn=KMS_KEY_ARN)
    stubber.assert_no_pending_responses()


def test_get_json_object_returns_none_when_missing() -> None:
    client, stubber = stubbed_client("s3")
    stubber.add_client_error("get_object", service_error_code="NoSuchKey", http_status_code=404)
    with stubber:
        assert s3_utils.get_json_object(client, bucket=BUCKET_NAME, key="state/manifest.json") is None
