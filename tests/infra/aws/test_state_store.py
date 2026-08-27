"""Tests for the S3-backed deployment manifest."""

from __future__ import annotations

from botocore.stub import ANY

from infra.aws.state_store import (
    DeploymentManifest,
    ResourceRecord,
    StateStore,
    config_hash,
)

from .aws_stub import stubbed_client

RESOURCE_PREFIX = "integrity-review-beta"
STATE_BUCKET = f"{RESOURCE_PREFIX}-tf-state-111111111111"
KMS_KEY_ARN = "arn:aws:kms:ap-south-1:111111111111:key/abcd"


def _store(s3_client) -> StateStore:
    return StateStore(
        s3_client=s3_client,
        state_bucket=STATE_BUCKET,
        kms_key_arn=KMS_KEY_ARN,
    )


def test_config_hash_is_deterministic_and_order_independent() -> None:
    first = config_hash({"a": 1, "b": {"c": 2}})
    second = config_hash({"b": {"c": 2}, "a": 1})
    assert first == second
    assert len(first) == 16


def test_load_returns_empty_manifest_when_no_state_object_exists() -> None:
    s3_client, s3_stub = stubbed_client("s3")
    s3_stub.add_client_error("get_object", service_error_code="NoSuchKey", http_status_code=404)

    with s3_stub:
        manifest = _store(s3_client).load(RESOURCE_PREFIX, config_hash_value="deadbeef")

    assert manifest.resources == {}
    assert manifest.config_hash == "deadbeef"


def test_save_writes_kms_encrypted_state_object() -> None:
    s3_client, s3_stub = stubbed_client("s3")
    s3_stub.add_response("put_object", {}, ANY)
    manifest = DeploymentManifest(
        resource_prefix=RESOURCE_PREFIX,
        config_hash="deadbeef",
        resources={
            "media_bucket": ResourceRecord(
                resource_type="s3.bucket",
                identifier="integrity-review-beta-media",
                arn="arn:aws:s3:::integrity-review-beta-media",
                config_hash="deadbeef",
                status="created",
                managed_by_tool=True,
            )
        },
    )

    with s3_stub:
        _store(s3_client).save(manifest)

    s3_stub.assert_no_pending_responses()
