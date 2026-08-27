"""Read-only deep infrastructure verification used by ``cli.py status``.

Every check appends a ``{name, passed, detail}`` entry; the overall result is
``PASS`` only if every check passed. This is what should gate turning ECS
desired count above zero.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from botocore.exceptions import ClientError

from .config import ALLOWED_ENVIRONMENTS
from .orchestrator import OrchestratorContext, resource_names
from .state_store import StateStore, config_hash
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
    sqs = ctx.client("sqs")
    checks = []
    for key in ("request_queue", "result_queue", "request_dlq", "result_dlq"):
        try:
            queue_url = sqs.get_queue_url(QueueName=names[key])["QueueUrl"]
            attributes = sqs.get_queue_attributes(
                QueueUrl=queue_url,
                AttributeNames=["KmsMasterKeyId", "ApproximateNumberOfMessages", "RedrivePolicy"],
            )["Attributes"]
            encrypted = "KmsMasterKeyId" in attributes
            checks.append(
                _check(
                    f"sqs.{key}",
                    encrypted,
                    f"encrypted={encrypted} visible={attributes.get('ApproximateNumberOfMessages')}",
                )
            )
        except sqs.exceptions.QueueDoesNotExist:
            checks.append(_check(f"sqs.{key}", False, "queue does not exist"))
    return checks


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
        checks.append(_check_ecs_service(ctx, names))

        state_store = StateStore(
            s3_client=ctx.client("s3"),
            state_bucket=ctx.state_bucket,
            kms_key_arn=ctx.state_kms_key_arn,
        )
        manifest = state_store.load(
            ctx.config.resource_prefix, config_hash_value=config_hash(asdict(ctx.config))
        )
        task_role = manifest.resources.get("task_role")
        if task_role is not None:
            checks.append(_check_task_role_least_privilege(ctx, task_role.arn))

    overall = "PASS" if all(check["passed"] for check in checks) else "FAIL"
    return {"overall": overall, "checks": checks}
