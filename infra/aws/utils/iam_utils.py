"""IAM role provisioning and least-privilege verification helpers."""

from __future__ import annotations

import json
from typing import Any

from botocore.exceptions import ClientError

from ..specs import RoleSpec


def _get_role(client: Any, name: str) -> dict[str, Any] | None:
    try:
        return client.get_role(RoleName=name)["Role"]
    except ClientError as error:
        if error.response.get("Error", {}).get("Code") == "NoSuchEntity":
            return None
        raise


def ensure_role(client: Any, spec: RoleSpec, tags: dict[str, str]) -> str:
    """Create/update the role, its trust policy, and its inline policies. Returns the ARN."""

    role = _get_role(client, spec.name)
    if role is None:
        role = client.create_role(
            RoleName=spec.name,
            AssumeRolePolicyDocument=json.dumps(spec.trust_policy),
            Description=spec.description,
            Tags=[{"Key": key, "Value": value} for key, value in tags.items()],
        )["Role"]
    else:
        client.update_assume_role_policy(
            RoleName=spec.name, PolicyDocument=json.dumps(spec.trust_policy)
        )

    for policy_name, policy_document in spec.inline_policies.items():
        client.put_role_policy(
            RoleName=spec.name, PolicyName=policy_name, PolicyDocument=json.dumps(policy_document)
        )
    return role["Arn"]


def validate_policy_document(access_analyzer_client: Any, policy_document: dict[str, Any]) -> list[dict[str, Any]]:
    """Run IAM Access Analyzer policy validation; returns any ERROR/SECURITY_WARNING findings."""

    response = access_analyzer_client.validate_policy(
        policyDocument=json.dumps(policy_document), policyType="IDENTITY_POLICY"
    )
    return [
        finding
        for finding in response.get("findings", [])
        if finding.get("findingType") in {"ERROR", "SECURITY_WARNING"}
    ]


def simulate_actions(
    client: Any, *, role_arn: str, actions: list[str], resource_arns: list[str]
) -> dict[str, str]:
    """Return ``{action: decision}`` (``allowed``/``implicitDeny``/``explicitDeny``)."""

    response = client.simulate_principal_policy(
        PolicySourceArn=role_arn, ActionNames=actions, ResourceArns=resource_arns
    )
    return {
        result["EvalActionName"]: result["EvalDecision"] for result in response["EvaluationResults"]
    }
