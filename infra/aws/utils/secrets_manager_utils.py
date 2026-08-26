"""Secrets Manager container provisioning.

``ensure_secret`` never writes a secret value — the Gemini API key is
populated out-of-band (``set_secret_value``) so it never enters Terraform-
style state or CLI history managed by this package.
"""

from __future__ import annotations

import json
from typing import Any

from botocore.exceptions import ClientError

from ..specs import SecretSpec


def _describe_secret(client: Any, name: str) -> dict[str, Any] | None:
    try:
        return client.describe_secret(SecretId=name)
    except ClientError as error:
        if error.response.get("Error", {}).get("Code") == "ResourceNotFoundException":
            return None
        raise


def ensure_secret(client: Any, spec: SecretSpec, tags: dict[str, str]) -> str:
    """Create the secret container (empty placeholder value) if missing. Returns the ARN."""

    from ..policies import secrets_manager_resource_policy

    existing = _describe_secret(client, spec.name)
    if existing is None:
        created = client.create_secret(
            Name=spec.name,
            Description=spec.description,
            KmsKeyId=spec.kms_key_arn,
            SecretString=json.dumps({"placeholder": "set-out-of-band"}),
            Tags=[{"Key": key, "Value": value} for key, value in tags.items()],
        )
        secret_arn = created["ARN"]
    else:
        secret_arn = existing["ARN"]

    policy = secrets_manager_resource_policy(allowed_role_arns=list(spec.allowed_role_arns))
    client.put_resource_policy(SecretId=secret_arn, ResourcePolicy=json.dumps(policy))
    return secret_arn


def set_secret_value(client: Any, *, secret_id: str, value: str) -> None:
    """Populate the real secret value; call manually/out-of-band, never from CI logs."""

    client.put_secret_value(SecretId=secret_id, SecretString=value)


def secret_has_placeholder_value(client: Any, *, secret_id: str) -> bool:
    value = client.get_secret_value(SecretId=secret_id)["SecretString"]
    try:
        return json.loads(value).get("placeholder") == "set-out-of-band"
    except (json.JSONDecodeError, AttributeError):
        return False
