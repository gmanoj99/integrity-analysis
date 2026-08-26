"""KMS key + alias provisioning for S3, SQS, Secrets Manager, and Logs."""

from __future__ import annotations

import json
from typing import Any

from ..specs import KmsKeySpec


def _find_key_id_by_alias(client: Any, alias: str) -> str | None:
    alias_name = f"alias/{alias}"
    paginator = client.get_paginator("list_aliases")
    for page in paginator.paginate():
        for entry in page["Aliases"]:
            if entry["AliasName"] == alias_name:
                return entry.get("TargetKeyId")
    return None


def ensure_key(client: Any, spec: KmsKeySpec) -> str:
    """Return the key ARN, creating the key and alias if they do not exist.

    Deliberately untagged: ``CreateKey`` only requires ``kms:TagResource`` in
    the caller's IAM policy when a non-empty ``Tags`` list is supplied, and
    the state manifest (not AWS-side tags) tracks resource ownership for
    ``status``/``destroy``.
    """

    key_id = _find_key_id_by_alias(client, spec.alias)
    if key_id is None:
        created = client.create_key(
            Description=spec.description,
            KeyUsage="ENCRYPT_DECRYPT",
            Origin="AWS_KMS",
            Policy=json.dumps(spec.key_policy),
        )["KeyMetadata"]
        key_id = created["KeyId"]
        client.create_alias(AliasName=f"alias/{spec.alias}", TargetKeyId=key_id)
    else:
        client.put_key_policy(KeyId=key_id, PolicyName="default", Policy=json.dumps(spec.key_policy))
    return client.describe_key(KeyId=key_id)["KeyMetadata"]["Arn"]


def key_arn_for_alias(client: Any, alias: str) -> str | None:
    key_id = _find_key_id_by_alias(client, alias)
    if key_id is None:
        return None
    return client.describe_key(KeyId=key_id)["KeyMetadata"]["Arn"]
