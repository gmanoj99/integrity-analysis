"""botocore Stubber coverage for the Secrets Manager container provisioning.

``ensure_secret`` must never write the real Gemini API key value.
"""

from __future__ import annotations

from botocore.stub import ANY

from infra.aws.specs import SecretSpec
from infra.aws.utils import secrets_manager_utils

from .aws_stub import stubbed_client

SECRET_NAME = "integrity-review-beta/gemini-api-key"
SECRET_ARN = "arn:aws:secretsmanager:ap-south-1:111111111111:secret:integrity-review-beta/gemini-api-key-abc123"


def test_ensure_secret_creates_placeholder_when_missing() -> None:
    client, stubber = stubbed_client("secretsmanager")
    spec = SecretSpec(
        name=SECRET_NAME,
        kms_key_arn="arn:aws:kms:ap-south-1:111111111111:key/abcd",
        description="Gemini API key",
        allowed_role_arns=("arn:aws:iam::111111111111:role/integrity-review-beta-ecs-execution",),
    )

    stubber.add_client_error("describe_secret", service_error_code="ResourceNotFoundException", http_status_code=400)
    stubber.add_response("create_secret", {"ARN": SECRET_ARN, "Name": SECRET_NAME}, ANY)
    stubber.add_response("put_resource_policy", {}, ANY)

    with stubber:
        arn = secrets_manager_utils.ensure_secret(client, spec, {"Project": "integrity-review"})

    assert arn == SECRET_ARN
    stubber.assert_no_pending_responses()


def test_ensure_secret_is_idempotent_when_already_present() -> None:
    client, stubber = stubbed_client("secretsmanager")
    spec = SecretSpec(
        name=SECRET_NAME,
        kms_key_arn="arn:aws:kms:ap-south-1:111111111111:key/abcd",
        description="Gemini API key",
        allowed_role_arns=("arn:aws:iam::111111111111:role/integrity-review-beta-ecs-execution",),
    )

    stubber.add_response("describe_secret", {"ARN": SECRET_ARN, "Name": SECRET_NAME}, ANY)
    stubber.add_response("put_resource_policy", {}, ANY)

    with stubber:
        arn = secrets_manager_utils.ensure_secret(client, spec, {"Project": "integrity-review"})

    assert arn == SECRET_ARN
    stubber.assert_no_pending_responses()


def test_secret_has_placeholder_value_detects_out_of_band_gap() -> None:
    client, stubber = stubbed_client("secretsmanager")
    stubber.add_response(
        "get_secret_value",
        {"SecretString": '{"placeholder": "set-out-of-band"}'},
        ANY,
    )
    with stubber:
        assert secrets_manager_utils.secret_has_placeholder_value(client, secret_id=SECRET_ARN) is True


def test_secret_has_placeholder_value_false_once_populated() -> None:
    client, stubber = stubbed_client("secretsmanager")
    stubber.add_response("get_secret_value", {"SecretString": "the-real-gemini-key"}, ANY)
    with stubber:
        assert secrets_manager_utils.secret_has_placeholder_value(client, secret_id=SECRET_ARN) is False
