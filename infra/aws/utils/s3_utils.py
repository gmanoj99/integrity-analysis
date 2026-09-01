"""Private, encrypted S3 buckets: creation, hardening, and small object helpers.

Every bucket created here gets Block Public Access, bucket-owner-enforced
ownership, KMS encryption, a TLS-only/approved-role bucket policy, and a
short lifecycle expiration for test objects.
"""

from __future__ import annotations

import json
from typing import Any

from botocore.exceptions import ClientError

from ..specs import BucketSpec


def bucket_exists(client: Any, bucket_name: str) -> bool:
    try:
        client.head_bucket(Bucket=bucket_name)
        return True
    except ClientError as error:
        if error.response.get("Error", {}).get("Code") in {"404", "NoSuchBucket"}:
            return False
        raise


def _create_bucket(client: Any, bucket_name: str, region: str) -> None:
    kwargs: dict[str, Any] = {"Bucket": bucket_name}
    if region != "us-east-1":
        kwargs["CreateBucketConfiguration"] = {"LocationConstraint": region}
    client.create_bucket(**kwargs)
    client.get_waiter("bucket_exists").wait(Bucket=bucket_name)


def _harden_bucket(client: Any, spec: BucketSpec) -> None:
    client.put_public_access_block(
        Bucket=spec.name,
        PublicAccessBlockConfiguration={
            "BlockPublicAcls": True,
            "IgnorePublicAcls": True,
            "BlockPublicPolicy": True,
            "RestrictPublicBuckets": True,
        },
    )
    client.put_bucket_ownership_controls(
        Bucket=spec.name, OwnershipControls={"Rules": [{"ObjectOwnership": "BucketOwnerEnforced"}]}
    )
    client.put_bucket_encryption(
        Bucket=spec.name,
        ServerSideEncryptionConfiguration={
            "Rules": [
                {
                    "ApplyServerSideEncryptionByDefault": {
                        "SSEAlgorithm": "aws:kms",
                        "KMSMasterKeyID": spec.kms_key_arn,
                    },
                    "BucketKeyEnabled": True,
                }
            ]
        },
    )
    client.put_bucket_versioning(Bucket=spec.name, VersioningConfiguration={"Status": "Enabled"})


def _put_tls_only_policy(client: Any, spec: BucketSpec) -> None:
    from ..policies import s3_tls_only_and_public_deny_policy

    bucket_arn = f"arn:aws:s3:::{spec.name}"
    policy = s3_tls_only_and_public_deny_policy(
        bucket_arn=bucket_arn, allowed_role_arns=list(spec.allowed_role_arns)
    )
    client.put_bucket_policy(Bucket=spec.name, Policy=json.dumps(policy))


def _put_lifecycle_policy(client: Any, spec: BucketSpec) -> None:
    client.put_bucket_lifecycle_configuration(
        Bucket=spec.name,
        LifecycleConfiguration={
            "Rules": [
                {
                    "ID": "expire-test-objects",
                    "Status": "Enabled",
                    "Filter": {"Prefix": ""},
                    "Expiration": {"Days": spec.expiration_days},
                    "NoncurrentVersionExpiration": {"NoncurrentDays": spec.expiration_days},
                    "AbortIncompleteMultipartUpload": {"DaysAfterInitiation": 1},
                }
            ]
        },
    )


def ensure_bucket(client: Any, spec: BucketSpec, tags: dict[str, str]) -> str:
    """Create (if missing) and fully harden the bucket. Returns the bucket ARN."""

    if not bucket_exists(client, spec.name):
        raise RuntimeError(
        f"S3 bucket {spec.name!r} does not exist; this tool will not create buckets")
    _harden_bucket(client, spec)
    _put_tls_only_policy(client, spec)
    if spec.expiration_days > 0:
        _put_lifecycle_policy(client, spec)
    client.put_bucket_tagging(
        Bucket=spec.name, Tagging={"TagSet": [{"Key": k, "Value": v} for k, v in tags.items()]}
    )
    return f"arn:aws:s3:::{spec.name}"


def put_json_object(client: Any, *, bucket: str, key: str, body: dict[str, Any], kms_key_arn: str) -> None:
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(body, indent=2).encode("utf-8"),
        ContentType="application/json",
        ServerSideEncryption="aws:kms",
        SSEKMSKeyId=kms_key_arn,
    )


def get_json_object(client: Any, *, bucket: str, key: str) -> dict[str, Any] | None:
    try:
        response = client.get_object(Bucket=bucket, Key=key)
    except ClientError as error:
        if error.response.get("Error", {}).get("Code") in {"404", "NoSuchKey"}:
            return None
        raise
    return json.loads(response["Body"].read().decode("utf-8"))
