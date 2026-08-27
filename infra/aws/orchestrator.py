"""Ordered, idempotent provisioning of the isolated ECS worker test stack.

``bootstrap`` creates the one-time state backend (S3 state bucket).
``apply`` provisions every application resource in dependency order, saving
the manifest after each step and rolling back only what *this* apply() call
created if a later step fails. ``destroy`` deletes everything this tool has
ever created, in reverse order. ``plan`` and ``status`` are read-only.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Callable

import boto3

from .config import EnvironmentConfig
from .policies import (
    ecs_task_execution_role_policy,
    ecs_task_kms_policy,
    ecs_task_protection_policy,
    ecs_task_s3_policy,
    ecs_task_sqs_policy,
    ecs_task_trust_policy,
)
from .session import client as make_client
from .specs import (
    AlarmSpec,
    BucketSpec,
    ClusterSpec,
    ContainerSpec,
    InternetGatewaySpec,
    KmsKeySpec,
    NatGatewaySpec,
    QueueSpec,
    RepositorySpec,
    RoleSpec,
    RouteTableSpec,
    ScalableTargetSpec,
    SecretSpec,
    SecurityGroupSpec,
    ServiceSpec,
    S3GatewayEndpointSpec,
    StepAdjustment,
    StepScalingPolicySpec,
    SubnetSpec,
    TaskDefinitionSpec,
    LogGroupSpec,
    VpcSpec,
)
from .state_store import DeploymentManifest, ResourceRecord, StateStore, config_hash
from .utils import (
    autoscaling_utils,
    cloudwatch_utils,
    ec2_utils,
    ecr_utils,
    ecs_utils,
    iam_utils,
    kms_utils,
    logs_utils,
    s3_utils,
    secrets_manager_utils,
    sqs_utils,
)

_LOGGER = logging.getLogger("infra.aws.orchestrator")
_ECS_SERVICE_NAMESPACE = "ecs"
_ECS_SCALABLE_DIMENSION = "ecs:service:DesiredCount"
_STEP_ORDER = [
    "network",
    "kms",
    "data_plane",
    "observability_prereqs",
    "ecr",
    "secret",
    "roles",
    "compute",
    "alarms",
    "autoscaling",
]


@dataclass(frozen=True, slots=True)
class OrchestratorContext:
    config: EnvironmentConfig
    session: boto3.Session
    state_bucket: str
    state_kms_key_arn: str
    image_tag: str

    def client(self, service_name: str) -> Any:
        return make_client(self.session, service_name)


def resource_names(config: EnvironmentConfig) -> dict[str, str]:
    prefix = config.resource_prefix
    return {
        "vpc": f"{prefix}-vpc",
        "public_subnet_0": f"{prefix}-public-0",
        "public_subnet_1": f"{prefix}-public-1",
        "private_subnet_0": f"{prefix}-private-0",
        "private_subnet_1": f"{prefix}-private-1",
        "internet_gateway": f"{prefix}-igw",
        "nat_gateway": f"{prefix}-nat",
        "public_route_table": f"{prefix}-public-rt",
        "private_route_table": f"{prefix}-private-rt",
        "security_group": f"{prefix}-worker-sg",
        "kms_data": f"{prefix}-data",
        "kms_secrets": f"{prefix}-secrets",
        "kms_logs": f"{prefix}-logs",
        "request_queue": f"{prefix}-request",
        "request_dlq": f"{prefix}-request-dlq",
        "result_queue": f"{prefix}-result",
        "result_dlq": f"{prefix}-result-dlq",
        "log_group": f"/ecs/{prefix}-worker",
        "ecr_repository": f"{prefix}-worker",
        "gemini_secret": f"{prefix}/gemini-api-key",
        "execution_role": f"{prefix}-ecs-execution",
        "task_role": f"{prefix}-ecs-task",
        "cluster": f"{prefix}-cluster",
        "task_family": f"{prefix}-worker",
        "service": f"{prefix}-worker-service",
    }


def _record(
    manifest: DeploymentManifest,
    key: str,
    *,
    resource_type: str,
    identifier: str,
    arn: str | None,
) -> None:
    manifest.resources[key] = ResourceRecord(
        resource_type=resource_type,
        identifier=identifier,
        arn=arn,
        config_hash=manifest.config_hash,
        status="created",
        managed_by_tool=True,
    )


def _ensure_network(ctx: OrchestratorContext, manifest: DeploymentManifest, names: dict[str, str]) -> None:
    ec2 = ctx.client("ec2")
    tags = ctx.config.resolved_tags()
    network = ctx.config.network

    vpc_id = ec2_utils.ensure_vpc(ec2, VpcSpec(names["vpc"], network.vpc_cidr_block), tags)
    _record(manifest, "vpc", resource_type="ec2.vpc", identifier=vpc_id, arn=None)

    public_subnet_ids = [
        ec2_utils.ensure_subnet(
            ec2,
            SubnetSpec(names[f"public_subnet_{i}"], vpc_id, cidr, network.availability_zones[i], True),
            tags,
        )
        for i, cidr in enumerate(network.public_subnet_cidrs)
    ]
    private_subnet_ids = [
        ec2_utils.ensure_subnet(
            ec2,
            SubnetSpec(names[f"private_subnet_{i}"], vpc_id, cidr, network.availability_zones[i], False),
            tags,
        )
        for i, cidr in enumerate(network.private_subnet_cidrs)
    ]
    for i, subnet_id in enumerate(public_subnet_ids):
        _record(manifest, f"public_subnet_{i}", resource_type="ec2.subnet", identifier=subnet_id, arn=None)
    for i, subnet_id in enumerate(private_subnet_ids):
        _record(manifest, f"private_subnet_{i}", resource_type="ec2.subnet", identifier=subnet_id, arn=None)

    igw_id = ec2_utils.ensure_internet_gateway(ec2, InternetGatewaySpec(names["internet_gateway"], vpc_id), tags)
    _record(manifest, "internet_gateway", resource_type="ec2.internet_gateway", identifier=igw_id, arn=None)

    nat_gateway_id = ec2_utils.ensure_nat_gateway(
        ec2, NatGatewaySpec(names["nat_gateway"], public_subnet_ids[0]), tags
    )
    _record(manifest, "nat_gateway", resource_type="ec2.nat_gateway", identifier=nat_gateway_id, arn=None)

    public_rt_id = ec2_utils.ensure_route_table(
        ec2,
        RouteTableSpec(
            names["public_route_table"], vpc_id, tuple(public_subnet_ids), internet_gateway_id=igw_id
        ),
        tags,
    )
    private_rt_id = ec2_utils.ensure_route_table(
        ec2,
        RouteTableSpec(
            names["private_route_table"], vpc_id, tuple(private_subnet_ids), nat_gateway_id=nat_gateway_id
        ),
        tags,
    )
    _record(manifest, "public_route_table", resource_type="ec2.route_table", identifier=public_rt_id, arn=None)
    _record(manifest, "private_route_table", resource_type="ec2.route_table", identifier=private_rt_id, arn=None)

    security_group_id = ec2_utils.ensure_worker_security_group(
        ec2,
        SecurityGroupSpec(names["security_group"], vpc_id, "ECS worker: no inbound, HTTPS egress only"),
        tags,
    )
    _record(manifest, "security_group", resource_type="ec2.security_group", identifier=security_group_id, arn=None)

    endpoint_id = ec2_utils.ensure_s3_gateway_endpoint(
        ec2,
        S3GatewayEndpointSpec(vpc_id, ctx.config.region, (public_rt_id, private_rt_id)),
        tags,
    )
    _record(manifest, "s3_gateway_endpoint", resource_type="ec2.vpc_endpoint", identifier=endpoint_id, arn=None)


def _admin_principal_arn(ctx: OrchestratorContext) -> str:
    """Resolve the IAM principal actually running the provisioner.

    KMS key policies only accept concrete IAM user/role ARNs, not the
    ``assumed-role`` STS ARN shape returned for role-based sessions, so
    assumed-role callers are rewritten to their underlying role ARN.
    """

    caller_arn = ctx.client("sts").get_caller_identity()["Arn"]
    if ":assumed-role/" in caller_arn:
        role_name = caller_arn.split(":assumed-role/", 1)[1].split("/", 1)[0]
        return f"arn:aws:iam::{ctx.config.account_id}:role/{role_name}"
    return caller_arn


def _kms_key_policy_for(ctx: OrchestratorContext, user_role_arns: list[str]) -> dict[str, Any]:
    from .policies import kms_key_policy

    return kms_key_policy(
        account_id=ctx.config.account_id,
        admin_role_arn=_admin_principal_arn(ctx),
        user_role_arns=user_role_arns,
    )


def _ensure_kms(ctx: OrchestratorContext, manifest: DeploymentManifest, names: dict[str, str]) -> None:
    kms = ctx.client("kms")
    task_role_arn = f"arn:aws:iam::{ctx.config.account_id}:role/{names['task_role']}"
    execution_role_arn = f"arn:aws:iam::{ctx.config.account_id}:role/{names['execution_role']}"

    for key_name, alias in (("kms_data", names["kms_data"]), ("kms_secrets", names["kms_secrets"]), ("kms_logs", names["kms_logs"])):
        user_roles = [task_role_arn] if key_name == "kms_data" else [execution_role_arn]
        arn = kms_utils.ensure_key(
            kms,
            KmsKeySpec(alias, f"{alias} - ECS test stack", _kms_key_policy_for(ctx, user_roles)),
        )
        _record(manifest, key_name, resource_type="kms.key", identifier=alias, arn=arn)


def _ensure_data_plane(ctx: OrchestratorContext, manifest: DeploymentManifest, names: dict[str, str]) -> None:
    sqs = ctx.client("sqs")
    tags = ctx.config.resolved_tags()
    data_kms_arn = manifest.resources["kms_data"].arn
    task_role_arn = f"arn:aws:iam::{ctx.config.account_id}:role/{names['task_role']}"

    request_dlq_url = sqs_utils.ensure_queue(
        sqs,
        QueueSpec(
            names["request_dlq"], data_kms_arn, 300, ctx.config.retention.queue_message_retention_seconds, None, 0, ()
        ),
        tags,
    )
    request_dlq_arn = sqs_utils.queue_arn(sqs, request_dlq_url)
    result_dlq_url = sqs_utils.ensure_queue(
        sqs,
        QueueSpec(
            names["result_dlq"], data_kms_arn, 300, ctx.config.retention.queue_message_retention_seconds, None, 0, ()
        ),
        tags,
    )
    result_dlq_arn = sqs_utils.queue_arn(sqs, result_dlq_url)
    _record(manifest, "request_dlq", resource_type="sqs.queue", identifier=request_dlq_url, arn=request_dlq_arn)
    _record(manifest, "result_dlq", resource_type="sqs.queue", identifier=result_dlq_url, arn=result_dlq_arn)

    request_queue_url = sqs_utils.ensure_queue(
        sqs,
        QueueSpec(
            names["request_queue"],
            data_kms_arn,
            300,
            ctx.config.retention.queue_message_retention_seconds,
            request_dlq_arn,
            5,
            (task_role_arn,),
        ),
        tags,
    )
    result_queue_url = sqs_utils.ensure_queue(
        sqs,
        QueueSpec(
            names["result_queue"],
            data_kms_arn,
            300,
            ctx.config.retention.queue_message_retention_seconds,
            result_dlq_arn,
            5,
            (task_role_arn,),
        ),
        tags,
    )
    _record(
        manifest,
        "request_queue",
        resource_type="sqs.queue",
        identifier=request_queue_url,
        arn=sqs_utils.queue_arn(sqs, request_queue_url),
    )
    _record(
        manifest,
        "result_queue",
        resource_type="sqs.queue",
        identifier=result_queue_url,
        arn=sqs_utils.queue_arn(sqs, result_queue_url),
    )


def _ensure_observability_prereqs(ctx: OrchestratorContext, manifest: DeploymentManifest, names: dict[str, str]) -> None:
    logs = ctx.client("logs")
    arn = logs_utils.ensure_log_group(
        logs,
        LogGroupSpec(names["log_group"], manifest.resources["kms_logs"].arn, ctx.config.retention.log_retention_days),
    )
    _record(manifest, "log_group", resource_type="logs.log_group", identifier=names["log_group"], arn=arn)


def _ensure_ecr(ctx: OrchestratorContext, manifest: DeploymentManifest, names: dict[str, str]) -> None:
    ecr = ctx.client("ecr")
    arn = ecr_utils.ensure_repository(
        ecr,
        RepositorySpec(names["ecr_repository"], manifest.resources["kms_data"].arn, ctx.config.retention.ecr_max_tagged_images),
        ctx.config.resolved_tags(),
    )
    _record(manifest, "ecr_repository", resource_type="ecr.repository", identifier=names["ecr_repository"], arn=arn)


def _ensure_secret(ctx: OrchestratorContext, manifest: DeploymentManifest, names: dict[str, str]) -> None:
    secrets_manager = ctx.client("secretsmanager")
    execution_role_arn = f"arn:aws:iam::{ctx.config.account_id}:role/{names['execution_role']}"
    arn = secrets_manager_utils.ensure_secret(
        secrets_manager,
        SecretSpec(
            names["gemini_secret"],
            manifest.resources["kms_secrets"].arn,
            "Gemini API key for the ECS worker (value set out-of-band)",
            (execution_role_arn,),
        ),
        ctx.config.resolved_tags(),
    )
    _record(manifest, "gemini_secret", resource_type="secretsmanager.secret", identifier=names["gemini_secret"], arn=arn)


def _ensure_roles(ctx: OrchestratorContext, manifest: DeploymentManifest, names: dict[str, str]) -> None:
    iam = ctx.client("iam")
    tags = ctx.config.resolved_tags()
    account_id = ctx.config.account_id
    trust = ecs_task_trust_policy()

    ecr_repo_arn = manifest.resources["ecr_repository"].arn
    log_group_arn = manifest.resources["log_group"].arn
    gemini_secret_arn = manifest.resources["gemini_secret"].arn
    execution_role_arn = iam_utils.ensure_role(
        iam,
        RoleSpec(
            names["execution_role"],
            trust,
            "ECS execution role: image pull, logs, Gemini secret",
            {
                "execution": ecs_task_execution_role_policy(
                    repository_arn=ecr_repo_arn,
                    log_group_arn=log_group_arn,
                    gemini_secret_arn=gemini_secret_arn,
                    kms_key_arns=[manifest.resources["kms_secrets"].arn, manifest.resources["kms_logs"].arn],
                )
            },
        ),
        tags,
    )
    _record(manifest, "execution_role", resource_type="iam.role", identifier=names["execution_role"], arn=execution_role_arn)

    cluster_arn = f"arn:aws:ecs:{ctx.config.region}:{account_id}:cluster/{names['cluster']}"
    storage_bucket_arn = f"arn:aws:s3:::{ctx.config.storage_bucket_name}"
    kms_key_arns = [manifest.resources["kms_data"].arn]
    if ctx.config.storage_kms_key_arn:
        # The shared media bucket lives outside this repo and may be
        # encrypted with its own KMS key; kms_data only covers the
        # request/result SQS queues provisioned here.
        kms_key_arns.append(ctx.config.storage_kms_key_arn)
    task_role_arn = iam_utils.ensure_role(
        iam,
        RoleSpec(
            names["task_role"],
            trust,
            "ECS task role: S3/SQS/KMS/task-protection least privilege",
            {
                "s3": ecs_task_s3_policy(
                    storage_bucket_arn=storage_bucket_arn,
                    stage=ctx.config.stage,
                ),
                "sqs": ecs_task_sqs_policy(
                    request_queue_arn=manifest.resources["request_queue"].arn,
                    result_queue_arn=manifest.resources["result_queue"].arn,
                ),
                "kms": ecs_task_kms_policy(
                    read_key_arns=kms_key_arns,
                    write_key_arns=kms_key_arns,
                ),
                "task_protection": ecs_task_protection_policy(cluster_arn=cluster_arn),
            },
        ),
        tags,
    )
    _record(manifest, "task_role", resource_type="iam.role", identifier=names["task_role"], arn=task_role_arn)


def _worker_environment(ctx: OrchestratorContext, manifest: DeploymentManifest, names: dict[str, str]) -> dict[str, str]:
    sizing = ctx.config.sizing
    return {
        "AWS_STORAGE_BUCKET_NAME": ctx.config.storage_bucket_name,
        "REQUEST_QUEUE_URL": manifest.resources["request_queue"].identifier,
        "RESULT_QUEUE_URL": manifest.resources["result_queue"].identifier,
        "AWS_REGION": ctx.config.region,
        "STAGE": ctx.config.stage,
        "ORGANIZATION_ID": ctx.config.organization_id,
        "MAX_CONCURRENT_REVIEWS": str(sizing.max_concurrent_reviews),
        "GEMINI_TASK_LIMIT": str(sizing.gemini_task_limit),
        "GEMINI_PER_REVIEW_LIMIT": str(sizing.gemini_per_review_limit),
        "VISIBILITY_TIMEOUT_SECONDS": "300",
        "HEARTBEAT_INTERVAL_SECONDS": "90",
        "POLL_WAIT_TIME_SECONDS": "20",
        "PRESIGN_EXPIRES_IN_SECONDS": "3600",
        "ECS_CLUSTER": names["cluster"],
        "LOG_LEVEL": "INFO",
    }


def _ensure_compute(ctx: OrchestratorContext, manifest: DeploymentManifest, names: dict[str, str]) -> None:
    ecs = ctx.client("ecs")
    tags = ctx.config.resolved_tags()

    cluster_arn = ecs_utils.ensure_cluster(ecs, ClusterSpec(names["cluster"]), tags)
    _record(manifest, "cluster", resource_type="ecs.cluster", identifier=names["cluster"], arn=cluster_arn)

    account_id = ctx.config.account_id
    image = f"{account_id}.dkr.ecr.{ctx.config.region}.amazonaws.com/{names['ecr_repository']}:{ctx.image_tag}"
    container = ContainerSpec(
        name="worker",
        image=image,
        log_group=names["log_group"],
        region=ctx.config.region,
        environment=_worker_environment(ctx, manifest, names),
        secrets={"GEMINI_API_KEY": manifest.resources["gemini_secret"].arn},
    )
    task_definition_arn = ecs_utils.register_task_definition(
        ecs,
        TaskDefinitionSpec(
            family=names["task_family"],
            cpu=ctx.config.sizing.task_cpu,
            memory=ctx.config.sizing.task_memory,
            execution_role_arn=manifest.resources["execution_role"].arn,
            task_role_arn=manifest.resources["task_role"].arn,
            ephemeral_storage_gib=ctx.config.sizing.ephemeral_storage_gib,
            stop_timeout_seconds=120,
            container=container,
        ),
        tags,
    )
    _record(manifest, "task_definition", resource_type="ecs.task_definition", identifier=names["task_family"], arn=task_definition_arn)

    service_arn = ecs_utils.ensure_service(
        ecs,
        ServiceSpec(
            cluster=names["cluster"],
            name=names["service"],
            task_definition_arn=task_definition_arn,
            desired_count=ctx.config.sizing.desired_count,
            subnet_ids=(
                manifest.resources["private_subnet_0"].identifier,
                manifest.resources["private_subnet_1"].identifier,
            ),
            security_group_ids=(manifest.resources["security_group"].identifier,),
        ),
        tags,
    )
    _record(manifest, "service", resource_type="ecs.service", identifier=names["service"], arn=service_arn)


def _alarm_specs(ctx: OrchestratorContext, manifest: DeploymentManifest, names: dict[str, str]) -> list[AlarmSpec]:
    prefix = ctx.config.resource_prefix
    common_dimensions_service = {"ClusterName": names["cluster"], "ServiceName": names["service"]}
    return [
        AlarmSpec(
            f"{prefix}-ecs-running-below-desired",
            "ECS/ContainerInsights",
            "RunningTaskCount",
            common_dimensions_service,
            "LessThanThreshold",
            float(ctx.config.sizing.desired_count),
            3,
            60,
            "Average",
            (),
        ),
        AlarmSpec(
            f"{prefix}-request-dlq-not-empty",
            "AWS/SQS",
            "ApproximateNumberOfMessagesVisible",
            {"QueueName": names["request_dlq"]},
            "GreaterThanThreshold",
            0.0,
            1,
            300,
            "Maximum",
            (),
        ),
        AlarmSpec(
            f"{prefix}-result-dlq-not-empty",
            "AWS/SQS",
            "ApproximateNumberOfMessagesVisible",
            {"QueueName": names["result_dlq"]},
            "GreaterThanThreshold",
            0.0,
            1,
            300,
            "Maximum",
            (),
        ),
        AlarmSpec(
            f"{prefix}-request-queue-age",
            "AWS/SQS",
            "ApproximateAgeOfOldestMessage",
            {"QueueName": names["request_queue"]},
            "GreaterThanThreshold",
            900.0,
            2,
            300,
            "Maximum",
            (),
        ),
    ]


def _ensure_alarms(ctx: OrchestratorContext, manifest: DeploymentManifest, names: dict[str, str]) -> None:
    cloudwatch = ctx.client("cloudwatch")
    for spec in _alarm_specs(ctx, manifest, names):
        cloudwatch_utils.ensure_alarm(cloudwatch, spec)
        _record(manifest, f"alarm.{spec.name}", resource_type="cloudwatch.alarm", identifier=spec.name, arn=None)


def _autoscaling_resource_id(names: dict[str, str]) -> str:
    return f"service/{names['cluster']}/{names['service']}"


def _ensure_autoscaling(ctx: OrchestratorContext, manifest: DeploymentManifest, names: dict[str, str]) -> None:
    """SQS-depth-driven step scaling: 0 tasks when idle, 1 for a small backlog, 2 above it.

    Step scaling (not target tracking) is used because target tracking's
    backlog-per-task metric divides by the running task count, so it cannot
    scale from or to zero. ``ecs:UpdateTaskProtection`` (see
    ``ecs_task_protection_policy``) keeps scale-in from killing an in-flight
    review.
    """

    aas = ctx.client("application-autoscaling")
    cloudwatch = ctx.client("cloudwatch")
    sizing = ctx.config.sizing
    prefix = ctx.config.resource_prefix
    resource_id = _autoscaling_resource_id(names)

    autoscaling_utils.register_scalable_target(
        aas,
        ScalableTargetSpec(
            service_namespace=_ECS_SERVICE_NAMESPACE,
            resource_id=resource_id,
            scalable_dimension=_ECS_SCALABLE_DIMENSION,
            min_capacity=sizing.min_task_count,
            max_capacity=sizing.max_task_count,
        ),
    )
    _record(
        manifest,
        "scalable_target",
        resource_type="applicationautoscaling.scalable_target",
        identifier=resource_id,
        arn=None,
    )

    scale_out_name = f"{prefix}-scale-out-on-backlog"
    scale_out_policy_arn = autoscaling_utils.put_step_scaling_policy(
        aas,
        StepScalingPolicySpec(
            name=scale_out_name,
            service_namespace=_ECS_SERVICE_NAMESPACE,
            resource_id=resource_id,
            scalable_dimension=_ECS_SCALABLE_DIMENSION,
            adjustment_type="ExactCapacity",
            # Bounds are relative to the alarm threshold (>=1 message): [1,3) -> 1
            # task, >=3 -> 2 tasks. This scales task count by queue depth, not by
            # MAX_CONCURRENT_REVIEWS=4 (reviews per task); the two are independent.
            step_adjustments=(
                StepAdjustment(scaling_adjustment=1, metric_interval_lower_bound=0.0, metric_interval_upper_bound=2.0),
                StepAdjustment(scaling_adjustment=2, metric_interval_lower_bound=2.0),
            ),
        ),
    )
    _record(
        manifest,
        "scaling_policy.scale_out",
        resource_type="applicationautoscaling.scaling_policy",
        identifier=scale_out_name,
        arn=scale_out_policy_arn,
    )

    scale_in_name = f"{prefix}-scale-in-on-idle"
    scale_in_policy_arn = autoscaling_utils.put_step_scaling_policy(
        aas,
        StepScalingPolicySpec(
            name=scale_in_name,
            service_namespace=_ECS_SERVICE_NAMESPACE,
            resource_id=resource_id,
            scalable_dimension=_ECS_SCALABLE_DIMENSION,
            adjustment_type="ExactCapacity",
            step_adjustments=(StepAdjustment(scaling_adjustment=0, metric_interval_upper_bound=0.0),),
        ),
    )
    _record(
        manifest,
        "scaling_policy.scale_in",
        resource_type="applicationautoscaling.scaling_policy",
        identifier=scale_in_name,
        arn=scale_in_policy_arn,
    )

    request_queue_dimensions = {"QueueName": names["request_queue"]}
    cloudwatch_utils.ensure_alarm(
        cloudwatch,
        AlarmSpec(
            scale_out_name,
            "AWS/SQS",
            "ApproximateNumberOfMessagesVisible",
            request_queue_dimensions,
            "GreaterThanOrEqualToThreshold",
            1.0,
            1,
            60,
            "Maximum",
            (scale_out_policy_arn,),
        ),
    )
    _record(manifest, "alarm.scale-out-on-backlog", resource_type="cloudwatch.alarm", identifier=scale_out_name, arn=None)

    cloudwatch_utils.ensure_alarm(
        cloudwatch,
        AlarmSpec(
            scale_in_name,
            "AWS/SQS",
            "ApproximateNumberOfMessagesVisible",
            request_queue_dimensions,
            "LessThanOrEqualToThreshold",
            0.0,
            sizing.scale_in_idle_periods,
            60,
            "Maximum",
            (scale_in_policy_arn,),
        ),
    )
    _record(manifest, "alarm.scale-in-on-idle", resource_type="cloudwatch.alarm", identifier=scale_in_name, arn=None)


_STEPS: dict[str, Callable[[OrchestratorContext, DeploymentManifest, dict[str, str]], None]] = {
    "network": _ensure_network,
    "kms": _ensure_kms,
    "data_plane": _ensure_data_plane,
    "observability_prereqs": _ensure_observability_prereqs,
    "ecr": _ensure_ecr,
    "secret": _ensure_secret,
    "roles": _ensure_roles,
    "compute": _ensure_compute,
    "alarms": _ensure_alarms,
    "autoscaling": _ensure_autoscaling,
}


def _build_state_store(ctx: OrchestratorContext) -> StateStore:
    return StateStore(
        s3_client=ctx.client("s3"),
        state_bucket=ctx.state_bucket,
        kms_key_arn=ctx.state_kms_key_arn,
    )


def bootstrap(ctx: OrchestratorContext) -> None:
    """One-time setup: encrypted state bucket and state KMS key."""

    s3 = ctx.client("s3")
    kms = ctx.client("kms")
    tags = ctx.config.resolved_tags()

    state_kms_arn = kms_utils.ensure_key(
        kms,
        KmsKeySpec(
            f"{ctx.config.resource_prefix}-state",
            "Terraform-free provisioner state encryption",
            _kms_key_policy_for(ctx, []),
        ),
    )
    s3_utils.ensure_bucket(
        s3,
        BucketSpec(ctx.state_bucket, ctx.config.region, state_kms_arn, 0, ()),
        tags,
    )


def plan(ctx: OrchestratorContext) -> dict[str, str]:
    """Read-only summary: which top-level resources already exist."""

    names = resource_names(ctx.config)
    s3 = ctx.client("s3")
    ecs = ctx.client("ecs")
    ecr = ctx.client("ecr")

    summary = {
        "storage_bucket": (
            "exists (external)"
            if s3_utils.bucket_exists(s3, ctx.config.storage_bucket_name)
            else "missing (external, not provisioned by this tool)"
        ),
        "ecr_repository": "exists" if ecr_utils.repository_exists(ecr, names["ecr_repository"]) else "create",
        "cluster": (
            "exists"
            if any(c["status"] == "ACTIVE" for c in ecs.describe_clusters(clusters=[names["cluster"]])["clusters"])
            else "create"
        ),
    }
    return summary


def apply(ctx: OrchestratorContext) -> DeploymentManifest:
    state_store = _build_state_store(ctx)
    manifest = state_store.load(
        ctx.config.resource_prefix, config_hash_value=config_hash(asdict(ctx.config))
    )
    names = resource_names(ctx.config)
    keys_before_this_apply = set(manifest.resources)
    try:
        for step_name in _STEP_ORDER:
            _LOGGER.info("apply: running step %s", step_name)
            _STEPS[step_name](ctx, manifest, names)
            state_store.save(manifest)
    except Exception:
        _LOGGER.error("apply: step failed, rolling back resources created by this run", exc_info=True)
        _rollback(ctx, manifest, created_this_run=set(manifest.resources) - keys_before_this_apply)
        state_store.save(manifest)
        raise
    return manifest


def _rollback(ctx: OrchestratorContext, manifest: DeploymentManifest, *, created_this_run: set[str]) -> None:
    for key in sorted(created_this_run, key=lambda k: _STEP_ORDER_INDEX.get(k.split(".")[0], -1), reverse=True):
        record = manifest.resources.get(key)
        if record is None or record.status == "destroyed":
            continue
        try:
            _delete_resource(ctx, record)
            record.status = "destroyed"
        except Exception:  # noqa: BLE001 - best-effort rollback, surface via logs
            _LOGGER.error("rollback: failed to delete %s (%s)", key, record.identifier, exc_info=True)


_STEP_ORDER_INDEX = {
    "vpc": 0, "public_subnet_0": 0, "public_subnet_1": 0, "private_subnet_0": 0, "private_subnet_1": 0,
    "internet_gateway": 0, "nat_gateway": 0, "public_route_table": 0, "private_route_table": 0,
    "security_group": 0, "s3_gateway_endpoint": 0,
    "kms_data": 1, "kms_secrets": 1, "kms_logs": 1,
    "request_dlq": 2, "result_dlq": 2, "request_queue": 2, "result_queue": 2,
    "log_group": 3, "ecr_repository": 4, "gemini_secret": 5,
    "execution_role": 6, "task_role": 6,
    "cluster": 7, "task_definition": 7, "service": 7,
    "alarm": 9,
    "scalable_target": 10, "scaling_policy": 11,
}


def _delete_resource(ctx: OrchestratorContext, record: ResourceRecord) -> None:
    """Best-effort teardown, dispatched by resource type. Used by rollback and destroy."""

    if record.resource_type == "applicationautoscaling.scaling_policy":
        autoscaling_utils.delete_scaling_policy(
            ctx.client("application-autoscaling"),
            name=record.identifier,
            service_namespace=_ECS_SERVICE_NAMESPACE,
            resource_id=_autoscaling_resource_id(resource_names(ctx.config)),
            scalable_dimension=_ECS_SCALABLE_DIMENSION,
        )
    elif record.resource_type == "applicationautoscaling.scalable_target":
        autoscaling_utils.deregister_scalable_target(
            ctx.client("application-autoscaling"),
            service_namespace=_ECS_SERVICE_NAMESPACE,
            resource_id=record.identifier,
            scalable_dimension=_ECS_SCALABLE_DIMENSION,
        )
    elif record.resource_type == "ecs.service":
        ecs = ctx.client("ecs")
        cluster = resource_names(ctx.config)["cluster"]
        ecs.update_service(cluster=cluster, service=record.identifier, desiredCount=0)
        ecs.delete_service(cluster=cluster, service=record.identifier, force=True)
    elif record.resource_type == "ecs.cluster":
        ctx.client("ecs").delete_cluster(cluster=record.identifier)
    elif record.resource_type == "ecs.task_definition":
        ecs = ctx.client("ecs")
        for revision in ecs.list_task_definitions(familyPrefix=record.identifier)["taskDefinitionArns"]:
            ecs.deregister_task_definition(taskDefinition=revision)
    elif record.resource_type == "secretsmanager.secret":
        ctx.client("secretsmanager").delete_secret(SecretId=record.identifier, ForceDeleteWithoutRecovery=True)
    elif record.resource_type == "ecr.repository":
        ctx.client("ecr").delete_repository(repositoryName=record.identifier, force=True)
    elif record.resource_type == "logs.log_group":
        ctx.client("logs").delete_log_group(logGroupName=record.identifier)
    elif record.resource_type == "sqs.queue":
        ctx.client("sqs").delete_queue(QueueUrl=record.identifier)
    elif record.resource_type == "s3.bucket":
        _empty_and_delete_bucket(ctx.client("s3"), record.identifier)
    elif record.resource_type == "iam.role":
        _delete_role(ctx.client("iam"), record.identifier)
    elif record.resource_type == "cloudwatch.alarm":
        ctx.client("cloudwatch").delete_alarms(AlarmNames=[record.identifier])
    # KMS keys and VPC networking are deliberately not force-deleted here:
    # keys are scheduled for deletion (7-day minimum) and network teardown
    # requires strict dependency ordering handled by `destroy()` directly.


def _empty_and_delete_bucket(s3: Any, bucket_name: str) -> None:
    paginator = s3.get_paginator("list_object_versions")
    for page in paginator.paginate(Bucket=bucket_name):
        objects = [
            {"Key": v["Key"], "VersionId": v["VersionId"]}
            for v in page.get("Versions", []) + page.get("DeleteMarkers", [])
        ]
        if objects:
            s3.delete_objects(Bucket=bucket_name, Delete={"Objects": objects})
    s3.delete_bucket(Bucket=bucket_name)


def _delete_role(iam: Any, role_name: str) -> None:
    for policy_name in iam.list_role_policies(RoleName=role_name)["PolicyNames"]:
        iam.delete_role_policy(RoleName=role_name, PolicyName=policy_name)
    iam.delete_role(RoleName=role_name)


def _teardown_network(ctx: OrchestratorContext, manifest: DeploymentManifest) -> None:
    """Best-effort VPC teardown in strict dependency order (NAT gateways bill hourly)."""

    ec2 = ctx.client("ec2")

    def _identifier(key: str) -> str | None:
        record = manifest.resources.get(key)
        return record.identifier if record and record.status != "destroyed" else None

    endpoint_id = _identifier("s3_gateway_endpoint")
    if endpoint_id:
        ec2.delete_vpc_endpoints(VpcEndpointIds=[endpoint_id])

    nat_gateway_id = _identifier("nat_gateway")
    if nat_gateway_id:
        ec2.delete_nat_gateway(NatGatewayId=nat_gateway_id)
        ec2.get_waiter("nat_gateway_deleted").wait(NatGatewayIds=[nat_gateway_id])
        addresses = ec2.describe_addresses(Filters=[{"Name": "tag:Name", "Values": [f"{ctx.config.resource_prefix}-nat"]}])
        for address in addresses.get("Addresses", []):
            ec2.release_address(AllocationId=address["AllocationId"])

    for key in ("public_route_table", "private_route_table"):
        route_table_id = _identifier(key)
        if not route_table_id:
            continue
        route_table = ec2.describe_route_tables(RouteTableIds=[route_table_id])["RouteTables"][0]
        for association in route_table.get("Associations", []):
            if not association.get("Main"):
                ec2.disassociate_route_table(AssociationId=association["RouteTableAssociationId"])
        ec2.delete_route_table(RouteTableId=route_table_id)

    igw_id = _identifier("internet_gateway")
    vpc_id = _identifier("vpc")
    if igw_id and vpc_id:
        ec2.detach_internet_gateway(InternetGatewayId=igw_id, VpcId=vpc_id)
        ec2.delete_internet_gateway(InternetGatewayId=igw_id)

    for key in ("public_subnet_0", "public_subnet_1", "private_subnet_0", "private_subnet_1"):
        subnet_id = _identifier(key)
        if subnet_id:
            ec2.delete_subnet(SubnetId=subnet_id)

    security_group_id = _identifier("security_group")
    if security_group_id:
        ec2.delete_security_group(GroupId=security_group_id)

    if vpc_id:
        ec2.delete_vpc(VpcId=vpc_id)

    for key in (
        "s3_gateway_endpoint", "nat_gateway", "public_route_table", "private_route_table",
        "internet_gateway", "public_subnet_0", "public_subnet_1", "private_subnet_0",
        "private_subnet_1", "security_group", "vpc",
    ):
        if key in manifest.resources:
            manifest.resources[key].status = "destroyed"


def _teardown_kms(ctx: OrchestratorContext, manifest: DeploymentManifest) -> None:
    """Schedule (not force-delete) KMS keys: AWS enforces a >=7-day deletion window."""

    kms = ctx.client("kms")
    for key in ("kms_data", "kms_secrets", "kms_logs"):
        record = manifest.resources.get(key)
        if record is None or record.status == "destroyed":
            continue
        kms.disable_key(KeyId=record.arn)
        kms.schedule_key_deletion(KeyId=record.arn, PendingWindowInDays=7)
        record.status = "destroyed"


def destroy(ctx: OrchestratorContext) -> None:
    """Delete every tool-managed resource, in reverse dependency order."""

    state_store = _build_state_store(ctx)
    manifest = state_store.load(
        ctx.config.resource_prefix, config_hash_value=config_hash(asdict(ctx.config))
    )
    managed_keys = {
        key
        for key, record in manifest.resources.items()
        if record.managed_by_tool
        and record.status != "destroyed"
        and _STEP_ORDER_INDEX.get(key.split(".")[0], -1) not in {0, 1}
    }
    _rollback(ctx, manifest, created_this_run=managed_keys)
    state_store.save(manifest)
    _teardown_network(ctx, manifest)
    state_store.save(manifest)
    _teardown_kms(ctx, manifest)
    state_store.save(manifest)
