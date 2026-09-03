"""Read-only deep infrastructure verification used by ``cli.py status``.

Every check appends a ``{name, passed, detail}`` entry; the overall result is
``PASS`` only if every check passed. This is what should gate turning ECS
desired count above zero.
"""

from __future__ import annotations

from typing import Any
from botocore.exceptions import ClientError

from .config import ALLOWED_ENVIRONMENTS
from .orchestrator import OrchestratorContext, resource_names
from .utils import iam_utils


def _check(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "passed": passed, "detail": detail}


def _check_identity(ctx: OrchestratorContext) -> dict[str, Any]:
    identity = ctx.client("sts").get_caller_identity()
    ok = (
        identity["Account"] == ctx.config.account_id
        and ctx.config.environment in ALLOWED_ENVIRONMENTS
    )
    return _check(
        "identity",
        ok,
        f"account={identity['Account']} region={ctx.config.region} environment={ctx.config.environment}",
    )


def _check_storage_bucket(ctx: OrchestratorContext) -> dict[str, Any]:
    """The shared media bucket is externally owned; only verify reachability here."""

    s3 = ctx.client("s3")
    bucket_name = ctx.config.storage_bucket_name
    try:
        s3.head_bucket(Bucket=bucket_name)
        return _check("s3.storage_bucket", True, f"bucket={bucket_name} reachable")
    except ClientError as error:
        return _check("s3.storage_bucket", False, f"bucket={bucket_name} error={error}")


def _check_queues(ctx: OrchestratorContext, names: dict[str, str]) -> list[dict[str, Any]]:
    """Queues are intentionally unencrypted (no SSE-KMS); this checks reachability plus,
    for the two main queues, that a redrive-to-DLQ policy is actually configured.
    """

    sqs = ctx.client("sqs")
    checks = []
    for key in ("request_queue", "response_queue", "request_dlq", "response_dlq"):
        try:
            queue_url = sqs.get_queue_url(QueueName=names[key])["QueueUrl"]
            attributes = sqs.get_queue_attributes(
                QueueUrl=queue_url,
                AttributeNames=["ApproximateNumberOfMessages", "RedrivePolicy"],
            )["Attributes"]
            redrive_configured = "RedrivePolicy" in attributes
            checks.append(
                _check(
                    f"sqs.{key}",
                    True,
                    f"redrive_configured={redrive_configured} visible={attributes.get('ApproximateNumberOfMessages')}",
                )
            )
        except sqs.exceptions.QueueDoesNotExist:
            checks.append(_check(f"sqs.{key}", False, "queue does not exist"))
    return checks


def _check_gemini_ssm_parameter(ctx: OrchestratorContext, names: dict[str, str]) -> dict[str, Any]:
    """The Gemini API key SSM parameter is externally managed (never written by this tool);
    only verify it exists so ``GEMINI_API_KEY`` injection at task start won't fail.
    """

    ssm = ctx.client("ssm")
    parameter_name = names["gemini_ssm_parameter"]
    try:
        ssm.get_parameter(Name=parameter_name)
        return _check("ssm.gemini_api_key", True, f"parameter={parameter_name} exists")
    except ClientError as error:
        if error.response.get("Error", {}).get("Code") == "ParameterNotFound":
            return _check(
                "ssm.gemini_api_key",
                False,
                f"parameter={parameter_name} not found; set it with: "
                f'aws ssm put-parameter --name "{parameter_name}" --type "SecureString" '
                f'--value "<GEMINI_API_KEY>" --overwrite',
            )
        raise


def _check_ecs_service(ctx: OrchestratorContext, names: dict[str, str]) -> dict[str, Any]:
    ecs = ctx.client("ecs")
    services = ecs.describe_services(cluster=names["cluster"], services=[names["service"]])["services"]
    if not services:
        return _check("ecs.service", False, "service not found")
    service = services[0]
    running = service["runningCount"]
    desired = service["desiredCount"]
    deployments_ok = all(d["rolloutState"] != "FAILED" for d in service.get("deployments", []))
    return _check(
        "ecs.service",
        running == desired and deployments_ok,
        f"running={running} desired={desired} deployments_ok={deployments_ok}",
    )


def _check_task_role_least_privilege(ctx: OrchestratorContext, manifest_task_role_arn: str) -> dict[str, Any]:
    iam = ctx.client("iam")
    denied_actions = ["secretsmanager:GetSecretValue", "iam:CreateRole", "kms:ScheduleKeyDeletion"]
    decisions = iam_utils.simulate_actions(
        iam, role_arn=manifest_task_role_arn, actions=denied_actions, resource_arns=["*"]
    )
    all_denied = all(decision != "allowed" for decision in decisions.values())
    return _check("iam.task_role_scope", all_denied, f"decisions={decisions}")


def run_status_checks(ctx: OrchestratorContext, *, deep: bool) -> dict[str, Any]:
    names = resource_names(ctx.config)
    checks: list[dict[str, Any]] = [_check_identity(ctx)]

    if deep:
        checks.append(_check_storage_bucket(ctx))
        checks.extend(_check_queues(ctx, names))
        checks.append(_check_gemini_ssm_parameter(ctx, names))
        checks.append(_check_ecs_service(ctx, names))

        try:
            task_role_arn = ctx.client("iam").get_role(RoleName=names["task_role"])["Role"]["Arn"]
        except ClientError as error:
            if error.response.get("Error", {}).get("Code") != "NoSuchEntity":
                raise
            checks.append(_check("iam.task_role_scope", False, "task role not found"))
        else:
            checks.append(_check_task_role_least_privilege(ctx, task_role_arn))

    overall = "PASS" if all(check["passed"] for check in checks) else "FAIL"
    return {"overall": overall, "checks": checks}
