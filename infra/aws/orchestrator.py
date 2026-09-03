"""Ordered, idempotent provisioning of the isolated ECS worker test stack.

``apply`` provisions every application resource in dependency order and
keeps the inventory in memory for this run only. On a first-apply failure
it rolls back only what *this* apply() call created. ``destroy`` requires
that in-memory inventory and is a no-op without it. ``plan`` and
``status`` are read-only.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Callable

import boto3

from .config import EnvironmentConfig
from .policies import (
    backend_request_queue_send_policy,
    backend_response_queue_receive_policy,
    codebuild_manual_deploy_policy,
    ecs_task_ai_usage_logs_policy,
    ecs_task_execution_role_policy,
    ecs_task_protection_policy,
    ecs_task_s3_policy,
    ecs_task_sqs_policy,
    ecs_task_trust_policy,
)
from .session import client as make_client
from .specs import (
    AlarmSpec,
    ClusterSpec,
    ContainerSpec,
    InternetGatewaySpec,
    NatGatewaySpec,
    QueueSpec,
    RepositorySpec,
    RoleSpec,
    RouteTableSpec,
    ScalableTargetSpec,
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
from .state_store import DeploymentManifest, ResourceRecord, config_hash
from .utils import (
    autoscaling_utils,
    cloudwatch_utils,
    codebuild_utils,
    ec2_utils,
    ecr_utils,
    ecs_utils,
    iam_utils,
    lambda_utils,
    logs_utils,
    s3_utils,
    sqs_utils,
)

_LOGGER = logging.getLogger("infra.aws.orchestrator")
_ECS_SERVICE_NAMESPACE = "ecs"
_ECS_SCALABLE_DIMENSION = "ecs:service:DesiredCount"
_CODEBUILD_DEPLOY_POLICY_NAME = "manual-deploy-worker"
_BACKEND_SEND_POLICY_NAME = "AiAnalysisRequestQueueSend"
_BACKEND_RECEIVE_POLICY_NAME = "AiAnalysisResponseQueueConsume"
_BACKEND_RESPONSE_EVENT_SOURCE_BATCH_SIZE = 1
_STEP_ORDER = [
    "network",
    "iam_roles",
    "data_plane",
    "observability_prereqs",
    "ecr",
    "roles",
    "compute",
    "alarms",
    "autoscaling",
    "codebuild_deploy_policy",
    "backend_send_access",
    "backend_receive_access",
]


@dataclass(frozen=True, slots=True)
class OrchestratorContext:
    config: EnvironmentConfig
    session: boto3.Session
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
        "request_queue": f"{prefix}-request-queue",
        "request_dlq": f"{prefix}-request-dlq",
        "response_queue": f"{prefix}-response-queue",
        "response_dlq": f"{prefix}-response-dlq",
        "log_group": f"/ecs/{prefix}-worker",
        "ecr_repository": f"{prefix}-worker",
        # SSM Parameter Store parameter holding the Gemini API key. This
        # tool never creates or writes this parameter -- see the
        # `aws ssm put-parameter` command documented next to
        # `_gemini_ssm_parameter_arn` below. Its name must keep this exact
        # "{prefix}-gemini-api-key" shape: the execution role's IAM
        # permission (`_gemini_ssm_parameter_prefix_arn`) is scoped to any
        # parameter starting with "{prefix}-", so every secret for this
        # environment must use that same prefix to be readable by ECS.
        "gemini_ssm_parameter": f"{prefix}-gemini-api-key",
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


def _gemini_ssm_parameter_arn(ctx: OrchestratorContext, names: dict[str, str]) -> str:
    """ARN of the single Gemini API key SSM parameter used as the task's ``GEMINI_API_KEY``.

    This tool never creates or writes this parameter's value -- populate it
    out-of-band, once per environment, e.g.::

        aws ssm put-parameter \\
          --name "nw-assessments-ai-analysis-beta-gemini-api-key" \\
          --type "SecureString" \\
          --value "<GEMINI_API_KEY>" \\
          --overwrite

    It uses SSM's default (AWS-managed ``alias/aws/ssm``) key, so no
    ``kms:Decrypt`` grant is required for the execution role to read it.
    """

    return f"arn:aws:ssm:{ctx.config.region}:{ctx.config.account_id}:parameter/{names['gemini_ssm_parameter']}"


def _gemini_ssm_parameter_prefix_arn(ctx: OrchestratorContext) -> str:
    """Wildcard ARN scoping the execution role's SSM read access to this environment's prefix only.

    Any parameter named "{resource_prefix}-*" (e.g. a future
    "nw-assessments-ai-analysis-beta-some-other-key") becomes readable
    without another policy change; parameters outside this prefix stay
    inaccessible.
    """

    return f"arn:aws:ssm:{ctx.config.region}:{ctx.config.account_id}:parameter/{ctx.config.resource_prefix}-*"


def _ensure_iam_roles(
    ctx: OrchestratorContext, manifest: DeploymentManifest, names: dict[str, str]
) -> None:
    iam = ctx.client("iam")
    tags = ctx.config.resolved_tags()
    trust = ecs_task_trust_policy()
    role_specs = (
        ("execution_role", "ECS execution role: image pull, logs, Gemini SSM parameter"),
        ("task_role", "ECS task role: S3/SQS/task-protection/AI-usage-logs least privilege"),
    )
    for key, description in role_specs:
        arn = iam_utils.ensure_role(
            iam, RoleSpec(names[key], trust, description), tags
        )
        _record(manifest, key, resource_type="iam.role", identifier=names[key], arn=arn)


_DLQ_MAX_RECEIVE_COUNT = 5


def _ensure_data_plane(ctx: OrchestratorContext, manifest: DeploymentManifest, names: dict[str, str]) -> None:
    sqs = ctx.client("sqs")
    tags = ctx.config.resolved_tags()
    task_role_arn = manifest.resources["task_role"].arn
    retention_seconds = ctx.config.retention.queue_message_retention_seconds

    request_dlq_url = sqs_utils.ensure_queue(
        sqs,
        QueueSpec(names["request_dlq"], 300, retention_seconds, None, 0, ()),
        tags,
    )
    request_dlq_arn = sqs_utils.queue_arn(sqs, request_dlq_url)
    response_dlq_url = sqs_utils.ensure_queue(
        sqs,
        QueueSpec(names["response_dlq"], 300, retention_seconds, None, 0, ()),
        tags,
    )
    response_dlq_arn = sqs_utils.queue_arn(sqs, response_dlq_url)
    _record(manifest, "request_dlq", resource_type="sqs.queue", identifier=request_dlq_url, arn=request_dlq_arn)
    _record(manifest, "response_dlq", resource_type="sqs.queue", identifier=response_dlq_url, arn=response_dlq_arn)

    request_queue_url = sqs_utils.ensure_queue(
        sqs,
        QueueSpec(
            names["request_queue"],
            300,
            retention_seconds,
            request_dlq_arn,
            _DLQ_MAX_RECEIVE_COUNT,
            (task_role_arn,),
        ),
        tags,
    )
    response_queue_url = sqs_utils.ensure_queue(
        sqs,
        QueueSpec(
            names["response_queue"],
            300,
            retention_seconds,
            response_dlq_arn,
            _DLQ_MAX_RECEIVE_COUNT,
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
        "response_queue",
        resource_type="sqs.queue",
        identifier=response_queue_url,
        arn=sqs_utils.queue_arn(sqs, response_queue_url),
    )


def _ensure_observability_prereqs(ctx: OrchestratorContext, manifest: DeploymentManifest, names: dict[str, str]) -> None:
    logs = ctx.client("logs")
    arn = logs_utils.ensure_log_group(
        logs,
        LogGroupSpec(names["log_group"], ctx.config.retention.log_retention_days),
    )
    _record(manifest, "log_group", resource_type="logs.log_group", identifier=names["log_group"], arn=arn)


def _ensure_ecr(ctx: OrchestratorContext, manifest: DeploymentManifest, names: dict[str, str]) -> None:
    ecr = ctx.client("ecr")
    arn = ecr_utils.ensure_repository(
        ecr,
        RepositorySpec(names["ecr_repository"], ctx.config.retention.ecr_max_tagged_images),
        ctx.config.resolved_tags(),
    )
    _record(manifest, "ecr_repository", resource_type="ecr.repository", identifier=names["ecr_repository"], arn=arn)


def _ensure_roles(ctx: OrchestratorContext, manifest: DeploymentManifest, names: dict[str, str]) -> None:
    iam = ctx.client("iam")
    tags = ctx.config.resolved_tags()
    account_id = ctx.config.account_id
    trust = ecs_task_trust_policy()

    ecr_repo_arn = manifest.resources["ecr_repository"].arn
    log_group_arn = manifest.resources["log_group"].arn
    execution_role_arn = iam_utils.ensure_role(
        iam,
        RoleSpec(
            names["execution_role"],
            trust,
            "ECS execution role: image pull, logs, Gemini SSM parameter",
            {
                "execution": ecs_task_execution_role_policy(
                    repository_arn=ecr_repo_arn,
                    log_group_arn=log_group_arn,
                    gemini_ssm_parameter_prefix_arn=_gemini_ssm_parameter_prefix_arn(ctx),
                )
            },
        ),
        tags,
    )
    _record(manifest, "execution_role", resource_type="iam.role", identifier=names["execution_role"], arn=execution_role_arn)

    cluster_arn = f"arn:aws:ecs:{ctx.config.region}:{account_id}:cluster/{names['cluster']}"
    storage_bucket_arn = f"arn:aws:s3:::{ctx.config.storage_bucket_name}"
    # The task role needs no KMS grant at all: this tool no longer manages
    # any customer-managed KMS key (SQS/DLQ, CloudWatch Logs, ECR, and the
    # Gemini SSM parameter are all unencrypted or use AWS-owned/managed
    # keys), and the externally-owned media bucket (same account) is
    # unencrypted too.
    inline_policies = {
        # Scoped to s3_bucket_stage_name ("topin_beta"/"topin_prod"), the
        # real S3 key prefix Django/the worker use for recordings, staged
        # requests, and results -- not ctx.config.stage ("beta"/"prod"),
        # which only feeds the worker's own STAGE env var/enum.
        "s3": ecs_task_s3_policy(
            storage_bucket_arn=storage_bucket_arn,
            stage=ctx.config.s3_bucket_stage_name,
        ),
        "sqs": ecs_task_sqs_policy(
            request_queue_arn=manifest.resources["request_queue"].arn,
            response_queue_arn=manifest.resources["response_queue"].arn,
        ),
        "task_protection": ecs_task_protection_policy(cluster_arn=cluster_arn),
        "ai_usage_logs": ecs_task_ai_usage_logs_policy(
            log_group_arn=(
                f"arn:aws:logs:{ctx.config.region}:{account_id}:"
                f"log-group:{ctx.config.custom_ai_logs_group_name}:*"
            ),
        ),
    }
    task_role_arn = iam_utils.ensure_role(
        iam,
        RoleSpec(
            names["task_role"],
            trust,
            "ECS task role: S3/SQS/task-protection/AI-usage-logs least privilege",
            inline_policies,
        ),
        tags,
    )
    _record(manifest, "task_role", resource_type="iam.role", identifier=names["task_role"], arn=task_role_arn)


def _worker_environment(ctx: OrchestratorContext, manifest: DeploymentManifest, names: dict[str, str]) -> dict[str, str]:
    sizing = ctx.config.sizing
    return {
        "AWS_STORAGE_BUCKET_NAME": ctx.config.storage_bucket_name,
        "REQUEST_QUEUE_URL": manifest.resources["request_queue"].identifier,
        "RESPONSE_QUEUE_URL": manifest.resources["response_queue"].identifier,
        "AWS_REGION": ctx.config.region,
        "STAGE": ctx.config.stage,
        "MAX_CONCURRENT_REVIEWS": str(sizing.max_concurrent_reviews),
        "GEMINI_TASK_LIMIT": str(sizing.gemini_task_limit),
        "GEMINI_PER_REVIEW_LIMIT": str(sizing.gemini_per_review_limit),
        "VISIBILITY_TIMEOUT_SECONDS": "300",
        "HEARTBEAT_INTERVAL_SECONDS": "90",
        "POLL_WAIT_TIME_SECONDS": "20",
        "PRESIGN_EXPIRES_IN_SECONDS": "3600",
        "CUSTOM_AI_LOGS_GROUP_NAME": ctx.config.custom_ai_logs_group_name,
        "CUSTOM_AI_LOGS_STREAM_NAME": ctx.config.stage,
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
        secrets={"GEMINI_API_KEY": _gemini_ssm_parameter_arn(ctx, names)},
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
            f"{prefix}-response-dlq-not-empty",
            "AWS/SQS",
            "ApproximateNumberOfMessagesVisible",
            {"QueueName": names["response_dlq"]},
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
            # MAX_CONCURRENT_REVIEWS=2 (reviews per task); the two are independent.
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


def _ensure_codebuild_deploy_policy(
    ctx: OrchestratorContext, manifest: DeploymentManifest, names: dict[str, str]
) -> None:
    """Keep the deploy-only policy on the CI project's *existing* execution role in sync.

    The CodeBuild project named in config (``codebuild_project_name``) and its
    execution role are provisioned outside this tool -- KV's CodeBuild runs
    ``cicd/buildspec.manual.yml`` on every push. This step only looks that
    role up and re-applies ``codebuild_manual_deploy_policy`` against it, so
    the role never needs the broader infra-provisioning permissions this
    tool's own roles use. It is intentionally excluded from rollback/destroy:
    this tool does not own the CodeBuild role's lifecycle, only this one
    inline policy on it.
    """

    role_arn = codebuild_utils.get_project_service_role_arn(
        ctx.client("codebuild"), ctx.config.codebuild_project_name
    )
    role_name = role_arn.split("/")[-1]
    policy_document = codebuild_manual_deploy_policy(
        ecr_repository_arn=manifest.resources["ecr_repository"].arn,
        ecs_service_arn=manifest.resources["service"].arn,
        ecs_execution_role_arn=manifest.resources["execution_role"].arn,
        ecs_task_role_arn=manifest.resources["task_role"].arn,
    )
    iam_utils.put_inline_policy(
        ctx.client("iam"),
        role_name=role_name,
        policy_name=_CODEBUILD_DEPLOY_POLICY_NAME,
        policy_document=policy_document,
    )
    _record(
        manifest,
        "codebuild_deploy_policy",
        resource_type="iam.inline_policy",
        identifier=f"{role_name}/{_CODEBUILD_DEPLOY_POLICY_NAME}",
        arn=role_arn,
    )


def _ensure_backend_send_access(
    ctx: OrchestratorContext, manifest: DeploymentManifest, names: dict[str, str]
) -> None:
    """Let the backend's existing SQS-send IAM user enqueue onto this stack's request queue.

    That IAM user (the one the Django backend authenticates as via
    ``CUSTOM_AWS_ACCESS_KEY_ID``/``SECRET``) is provisioned outside this
    tool, so this only attaches a policy to it, never creating or deleting it.
    """

    account_id = ctx.config.account_id
    user_name = ctx.config.backend_iam_user_name
    policy_document = backend_request_queue_send_policy(
        request_queue_arn=manifest.resources["request_queue"].arn,
    )
    iam_utils.put_user_inline_policy(
        ctx.client("iam"),
        user_name=user_name,
        policy_name=_BACKEND_SEND_POLICY_NAME,
        policy_document=policy_document,
    )
    _record(
        manifest,
        "backend_send_policy",
        resource_type="iam.inline_policy",
        identifier=f"{user_name}/{_BACKEND_SEND_POLICY_NAME}",
        arn=f"arn:aws:iam::{account_id}:user/{user_name}",
    )


def _ensure_backend_receive_access(
    ctx: OrchestratorContext, manifest: DeploymentManifest, names: dict[str, str]
) -> None:
    """Let the backend's existing Lambda consume from this stack's response queue.

    The Lambda function and its execution role are provisioned outside this
    tool; this only attaches a policy to the role and wires up the queue
    trigger, never creating or deleting the function or role itself.
    """

    lambda_client = ctx.client("lambda")
    function_name = ctx.config.backend_lambda_function_name
    response_queue_arn = manifest.resources["response_queue"].arn

    role_arn = lambda_utils.get_function_execution_role_arn(lambda_client, function_name)
    role_name = role_arn.split("/")[-1]
    policy_document = backend_response_queue_receive_policy(
        response_queue_arn=response_queue_arn,
    )
    iam_utils.put_inline_policy(
        ctx.client("iam"),
        role_name=role_name,
        policy_name=_BACKEND_RECEIVE_POLICY_NAME,
        policy_document=policy_document,
    )
    _record(
        manifest,
        "backend_receive_policy",
        resource_type="iam.inline_policy",
        identifier=f"{role_name}/{_BACKEND_RECEIVE_POLICY_NAME}",
        arn=role_arn,
    )

    mapping_uuid = lambda_utils.ensure_event_source_mapping(
        lambda_client,
        function_name=function_name,
        event_source_arn=response_queue_arn,
        batch_size=_BACKEND_RESPONSE_EVENT_SOURCE_BATCH_SIZE,
    )
    _record(
        manifest,
        "backend_event_source_mapping",
        resource_type="lambda.event_source_mapping",
        identifier=mapping_uuid,
        arn=None,
    )


_STEPS: dict[str, Callable[[OrchestratorContext, DeploymentManifest, dict[str, str]], None]] = {
    "network": _ensure_network,
    "iam_roles": _ensure_iam_roles,
    "data_plane": _ensure_data_plane,
    "observability_prereqs": _ensure_observability_prereqs,
    "ecr": _ensure_ecr,
    "roles": _ensure_roles,
    "compute": _ensure_compute,
    "alarms": _ensure_alarms,
    "autoscaling": _ensure_autoscaling,
    "codebuild_deploy_policy": _ensure_codebuild_deploy_policy,
    "backend_send_access": _ensure_backend_send_access,
    "backend_receive_access": _ensure_backend_receive_access,
}


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


def build_outputs(ctx: OrchestratorContext, manifest: DeploymentManifest, names: dict[str, str]) -> dict[str, Any]:
    """Flattened, buildspec-ready summary of every resource this apply() created.

    Meant to be captured (e.g. to a file) before ``infra/aws`` is deleted from
    the repo, so the identifiers/ARNs needed to hand-write a manual,
    provisioner-free buildspec are not lost. ``environment`` reuses
    ``_worker_environment`` so it matches exactly what the real task
    definition was registered with; ``raw_manifest_resources`` keeps every
    other resource available for audit even though only the ``ecr``/``ecs``
    blocks are needed to redeploy without this provisioner.
    """

    account_id = ctx.config.account_id
    return {
        "account_id": account_id,
        "region": ctx.config.region,
        "resource_prefix": ctx.config.resource_prefix,
        "ecr": {
            "repository_name": names["ecr_repository"],
            "repository_uri": f"{account_id}.dkr.ecr.{ctx.config.region}.amazonaws.com/{names['ecr_repository']}",
        },
        "ecs": {
            "cluster_name": names["cluster"],
            "service_name": names["service"],
            "task_family": names["task_family"],
            "container_name": "worker",
            "execution_role_arn": manifest.resources["execution_role"].arn,
            "task_role_arn": manifest.resources["task_role"].arn,
            "log_group": names["log_group"],
            "cpu": ctx.config.sizing.task_cpu,
            "memory": ctx.config.sizing.task_memory,
            "ephemeral_storage_gib": ctx.config.sizing.ephemeral_storage_gib,
            "subnet_ids": [
                manifest.resources["private_subnet_0"].identifier,
                manifest.resources["private_subnet_1"].identifier,
            ],
            "security_group_ids": [manifest.resources["security_group"].identifier],
        },
        "secrets": {"GEMINI_API_KEY": _gemini_ssm_parameter_arn(ctx, names)},
        "environment": _worker_environment(ctx, manifest, names),
        "raw_manifest_resources": manifest.to_dict()["resources"],
    }


def apply(ctx: OrchestratorContext) -> DeploymentManifest:
    names = resource_names(ctx.config)
    manifest = DeploymentManifest(
        resource_prefix=ctx.config.resource_prefix,
        config_hash=config_hash(asdict(ctx.config)),
    )
    keys_before_this_apply = set(manifest.resources)
    cluster_already_exists = any(
        cluster["status"] == "ACTIVE"
        for cluster in ctx.client("ecs").describe_clusters(clusters=[names["cluster"]])["clusters"]
    )
    try:
        for step_name in _STEP_ORDER:
            _LOGGER.info("apply: running step %s", step_name)
            _STEPS[step_name](ctx, manifest, names)
    except Exception:
        _LOGGER.error("apply: step failed, rolling back resources created by this run", exc_info=True)
        if not cluster_already_exists:
            _rollback(ctx, manifest, created_this_run=set(manifest.resources) - keys_before_this_apply)
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
    "request_dlq": 2, "response_dlq": 2, "request_queue": 2, "response_queue": 2,
    "log_group": 3, "ecr_repository": 4,
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
    # VPC networking is deliberately not force-deleted here: teardown
    # requires strict dependency ordering handled by `destroy()` directly.
    # "iam.inline_policy" (CodeBuild deploy policy, backend send/receive
    # policies) and "lambda.event_source_mapping" are also deliberately
    # unhandled: those roles/users/functions belong to externally
    # provisioned principals, so this tool never deletes them or anything
    # attached to them.


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


def destroy(ctx: OrchestratorContext) -> None:
    """Delete every tool-managed resource, in reverse dependency order.

    Inventory is not persisted, so this is a no-op unless a future caller
    supplies a manifest collected from ``apply``.
    """

    _LOGGER.warning(
        "destroy: no persisted inventory for %s; refusing to guess resources to delete",
        ctx.config.resource_prefix,
    )
