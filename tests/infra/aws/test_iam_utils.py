"""botocore Stubber coverage for role idempotency and access-denied verification."""

from __future__ import annotations

from botocore.stub import ANY

from infra.aws import policies
from infra.aws.specs import RoleSpec
from infra.aws.utils import iam_utils

from .aws_stub import stubbed_client

ROLE_NAME = "integrity-review-beta-ecs-task"
ROLE_ARN = f"arn:aws:iam::111111111111:role/{ROLE_NAME}"


def _spec() -> RoleSpec:
    return RoleSpec(
        name=ROLE_NAME,
        trust_policy=policies.ecs_task_trust_policy(),
        description="ECS task role",
        inline_policies={"s3": {"Version": "2012-10-17", "Statement": []}},
    )


def test_ensure_role_creates_role_when_missing() -> None:
    client, stubber = stubbed_client("iam")
    stubber.add_client_error("get_role", service_error_code="NoSuchEntity", http_status_code=404)
    stubber.add_response("create_role", {"Role": {"RoleName": ROLE_NAME, "Arn": ROLE_ARN}}, ANY)
    stubber.add_response("put_role_policy", {}, ANY)

    with stubber:
        arn = iam_utils.ensure_role(client, _spec(), {"Project": "integrity-review"})

    assert arn == ROLE_ARN
    stubber.assert_no_pending_responses()


def test_ensure_role_updates_trust_policy_when_existing() -> None:
    client, stubber = stubbed_client("iam")
    stubber.add_response("get_role", {"Role": {"RoleName": ROLE_NAME, "Arn": ROLE_ARN}}, ANY)
    stubber.add_response("update_assume_role_policy", {}, ANY)
    stubber.add_response("put_role_policy", {}, ANY)

    with stubber:
        arn = iam_utils.ensure_role(client, _spec(), {"Project": "integrity-review"})

    assert arn == ROLE_ARN
    stubber.assert_no_pending_responses()


def test_simulate_actions_flags_denied_actions() -> None:
    client, stubber = stubbed_client("iam")
    stubber.add_response(
        "simulate_principal_policy",
        {
            "EvaluationResults": [
                {"EvalActionName": "secretsmanager:GetSecretValue", "EvalDecision": "implicitDeny"},
                {"EvalActionName": "sqs:ReceiveMessage", "EvalDecision": "allowed"},
            ]
        },
        ANY,
    )

    with stubber:
        decisions = iam_utils.simulate_actions(
            client,
            role_arn=ROLE_ARN,
            actions=["secretsmanager:GetSecretValue", "sqs:ReceiveMessage"],
            resource_arns=["*"],
        )

    assert decisions["secretsmanager:GetSecretValue"] == "implicitDeny"
    assert decisions["sqs:ReceiveMessage"] == "allowed"


def test_validate_policy_document_surfaces_only_errors_and_warnings() -> None:
    client, stubber = stubbed_client("accessanalyzer")
    stubber.add_response(
        "validate_policy",
        {
            "findings": [
                {"findingType": "ERROR", "findingDetails": "malformed statement"},
                {"findingType": "SUGGESTION", "findingDetails": "use a narrower resource"},
            ]
        },
        ANY,
    )

    with stubber:
        findings = iam_utils.validate_policy_document(client, {"Version": "2012-10-17", "Statement": []})

    assert len(findings) == 1
    assert findings[0]["findingType"] == "ERROR"
