"""Resource specification DTOs shared by the provisioner utils.

Each ``ensure_*`` function in ``infra.aws.utils`` takes one spec dataclass
instead of a long, error-prone keyword-argument list.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class VpcSpec:
    name: str
    cidr_block: str


@dataclass(frozen=True, slots=True)
class SubnetSpec:
    name: str
    vpc_id: str
    cidr_block: str
    availability_zone: str
    public: bool


@dataclass(frozen=True, slots=True)
class InternetGatewaySpec:
    name: str
    vpc_id: str


@dataclass(frozen=True, slots=True)
class NatGatewaySpec:
    name: str
    public_subnet_id: str


@dataclass(frozen=True, slots=True)
class RouteTableSpec:
    name: str
    vpc_id: str
    subnet_ids: tuple[str, ...]
    # Exactly one of internet_gateway_id / nat_gateway_id is set.
    internet_gateway_id: str | None = None
    nat_gateway_id: str | None = None


@dataclass(frozen=True, slots=True)
class SecurityGroupSpec:
    name: str
    vpc_id: str
    description: str


@dataclass(frozen=True, slots=True)
class S3GatewayEndpointSpec:
    vpc_id: str
    region: str
    route_table_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class NetworkSpec:
    vpc: VpcSpec
    public_subnets: tuple[SubnetSpec, ...]
    private_subnets: tuple[SubnetSpec, ...]
    worker_security_group: SecurityGroupSpec


@dataclass(frozen=True, slots=True)
class KmsKeySpec:
    alias: str
    description: str
    key_policy: dict[str, Any]


@dataclass(frozen=True, slots=True)
class BucketSpec:
    name: str
    region: str
    kms_key_arn: str
    expiration_days: int
    allowed_role_arns: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class QueueSpec:
    name: str
    kms_key_id: str
    visibility_timeout_seconds: int
    message_retention_seconds: int
    dlq_arn: str | None
    max_receive_count: int
    allowed_role_arns: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RepositorySpec:
    name: str
    kms_key_arn: str
    max_tagged_images: int


@dataclass(frozen=True, slots=True)
class SecretSpec:
    name: str
    kms_key_arn: str
    description: str
    allowed_role_arns: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RoleSpec:
    name: str
    trust_policy: dict[str, Any]
    description: str
    inline_policies: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LogGroupSpec:
    name: str
    kms_key_arn: str
    retention_days: int


@dataclass(frozen=True, slots=True)
class AlarmSpec:
    name: str
    namespace: str
    metric_name: str
    dimensions: dict[str, str]
    comparison_operator: str
    threshold: float
    evaluation_periods: int
    period_seconds: int
    statistic: str
    alarm_actions: tuple[str, ...]
    treat_missing_data: str = "notBreaching"


@dataclass(frozen=True, slots=True)
class ClusterSpec:
    name: str


@dataclass(frozen=True, slots=True)
class ContainerSpec:
    name: str
    image: str
    log_group: str
    region: str
    environment: dict[str, str]
    secrets: dict[str, str]


@dataclass(frozen=True, slots=True)
class TaskDefinitionSpec:
    family: str
    cpu: str
    memory: str
    execution_role_arn: str
    task_role_arn: str
    ephemeral_storage_gib: int
    stop_timeout_seconds: int
    container: ContainerSpec


@dataclass(frozen=True, slots=True)
class ServiceSpec:
    cluster: str
    name: str
    task_definition_arn: str
    desired_count: int
    subnet_ids: tuple[str, ...]
    security_group_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ScalableTargetSpec:
    service_namespace: str
    resource_id: str
    scalable_dimension: str
    min_capacity: int
    max_capacity: int


@dataclass(frozen=True, slots=True)
class StepAdjustment:
    scaling_adjustment: int
    metric_interval_lower_bound: float | None = None
    metric_interval_upper_bound: float | None = None


@dataclass(frozen=True, slots=True)
class StepScalingPolicySpec:
    name: str
    service_namespace: str
    resource_id: str
    scalable_dimension: str
    adjustment_type: str
    step_adjustments: tuple[StepAdjustment, ...]
    metric_aggregation_type: str = "Maximum"
    cooldown_seconds: int = 60


@dataclass(frozen=True, slots=True)
class CodeCommitRepositorySpec:
    name: str
    description: str
    default_branch: str


@dataclass(frozen=True, slots=True)
class CodeBuildProjectSpec:
    name: str
    description: str
    service_role_arn: str
    buildspec_path: str
    image: str
    compute_type: str
    privileged_mode: bool
    log_group_name: str
    environment_variables: dict[str, str]


@dataclass(frozen=True, slots=True)
class CodePipelineArtifactStoreSpec:
    bucket_name: str
    kms_key_arn: str


@dataclass(frozen=True, slots=True)
class CodePipelineSourceStageSpec:
    repository_name: str
    branch_name: str
    output_artifact_name: str


@dataclass(frozen=True, slots=True)
class CodePipelineBuildStageSpec:
    name: str
    project_name: str
    input_artifact_name: str
    output_artifact_name: str | None = None


@dataclass(frozen=True, slots=True)
class CodePipelineSpec:
    name: str
    service_role_arn: str
    artifact_store: CodePipelineArtifactStoreSpec
    source: CodePipelineSourceStageSpec
    build_stages: tuple[CodePipelineBuildStageSpec, ...]


@dataclass(frozen=True, slots=True)
class EventBridgeRuleSpec:
    name: str
    description: str
    event_pattern: dict[str, Any]
    target_arn: str
    target_role_arn: str
    target_id: str
