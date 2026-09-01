"""IAM trust and permission policy document builders.

Every function returns a plain policy-document dict; nothing here calls
boto3. Keeping policy *shape* separate from IAM API calls (``utils/iam_utils``)
makes each statement independently unit-testable and reviewable.
"""
from __future__ import annotations

from typing import Any

ECS_TASKS_PRINCIPAL = "ecs-tasks.amazonaws.com"
REQUEST_PREFIX_TEMPLATE = "{stage}/media/ai_integrity_review_requests/*"
RESULT_PREFIX_TEMPLATE = "{stage}/media/ai_integrity_review_results/*"
RECORDING_PREFIX_TEMPLATE = "{stage}/media/*"


def _statement(
    *, sid: str, actions: list[str], resources: list[str], condition: dict[str, Any] | None = None
) -> dict[str, Any]:
    statement: dict[str, Any] = {
        "Sid": sid,
        "Effect": "Allow",
        "Action": actions,
        "Resource": resources,
    }
    if condition:
        statement["Condition"] = condition
    return statement


def _document(statements: list[dict[str, Any]]) -> dict[str, Any]:
    return {"Version": "2012-10-17", "Statement": statements}


def ecs_task_trust_policy() -> dict[str, Any]:
    return _document(
        [
            {
                "Sid": "AllowEcsTasksAssume",
                "Effect": "Allow",
                "Principal": {"Service": ECS_TASKS_PRINCIPAL},
                "Action": "sts:AssumeRole",
            }
        ]
    )


def oidc_trust_policy(
    *, provider_arn: str, repository: str, allowed_branch: str, allowed_environment: str
) -> dict[str, Any]:
    """CI/provisioner trust policy: only the named repo, branch, and environment may assume."""

    return _document(
        [
            {
                "Sid": "AllowGithubOidc",
                "Effect": "Allow",
                "Principal": {"Federated": provider_arn},
                "Action": "sts:AssumeRoleWithWebIdentity",
                "Condition": {
                    "StringEquals": {
                        "token.actions.githubusercontent.com:aud": "sts.amazonaws.com",
                        "token.actions.githubusercontent.com:sub": (
                            f"repo:{repository}:ref:refs/heads/{allowed_branch}"
                            f":environment:{allowed_environment}"
                        ),
                    }
                },
            }
        ]
    )


def ecs_task_execution_role_policy(
    *, repository_arn: str, log_group_arn: str, gemini_secret_arn: str, kms_key_arns: list[str]
) -> dict[str, Any]:
    return _document(
        [
            _statement(
                sid="AllowEcrAuth", actions=["ecr:GetAuthorizationToken"], resources=["*"]
            ),
            _statement(
                sid="AllowEcrPull",
                actions=[
                    "ecr:BatchCheckLayerAvailability",
                    "ecr:GetDownloadUrlForLayer",
                    "ecr:BatchGetImage",
                ],
                resources=[repository_arn],
            ),
            _statement(
                sid="AllowLogDelivery",
                actions=["logs:CreateLogStream", "logs:PutLogEvents"],
                resources=[f"{log_group_arn}:*"],
            ),
            _statement(
                sid="AllowGeminiSecretRead",
                actions=["secretsmanager:GetSecretValue"],
                resources=[gemini_secret_arn],
            ),
            _statement(sid="AllowKmsDecrypt", actions=["kms:Decrypt"], resources=kms_key_arns),
        ]
    )


def ecs_task_s3_policy(*, storage_bucket_arn: str, stage: str) -> dict[str, Any]:
    """Least privilege on the single shared media bucket, scoped to beta's own prefixes."""

    request_prefix = REQUEST_PREFIX_TEMPLATE.format(stage=stage)
    result_prefix = RESULT_PREFIX_TEMPLATE.format(stage=stage)
    recording_prefix = RECORDING_PREFIX_TEMPLATE.format(stage=stage)
    return _document(
        [
            _statement(
                sid="AllowRecordingRead",
                actions=["s3:GetObject"],
                resources=[f"{storage_bucket_arn}/{recording_prefix}"],
            ),
            _statement(
                sid="AllowRequestRead",
                actions=["s3:GetObject"],
                resources=[f"{storage_bucket_arn}/{request_prefix}"],
            ),
            _statement(
                sid="AllowResultReadWrite",
                actions=["s3:GetObject", "s3:PutObject"],
                resources=[f"{storage_bucket_arn}/{result_prefix}"],
            ),
            _statement(
                sid="AllowResultPrefixList",
                actions=["s3:ListBucket"],
                resources=[storage_bucket_arn],
                condition={"StringLike": {"s3:prefix": [result_prefix]}},
            ),
        ]
    )


def ecs_task_sqs_policy(*, request_queue_arn: str, response_queue_arn: str) -> dict[str, Any]:
    return _document(
        [
            _statement(
                sid="AllowRequestQueueConsume",
                actions=[
                    "sqs:ReceiveMessage",
                    "sqs:DeleteMessage",
                    "sqs:ChangeMessageVisibility",
                ],
                resources=[request_queue_arn],
            ),
            _statement(
                sid="AllowResponseQueuePublish",
                actions=["sqs:SendMessage"],
                resources=[response_queue_arn],
            ),
        ]
    )


def ecs_task_ai_usage_logs_policy(*, log_group_arn: str) -> dict[str, Any]:
    """Least privilege for the worker's boto3 PutLogEvents path to the shared custom-ai-logs group."""

    return _document(
        [
            _statement(
                sid="AllowAiUsageLogPublish",
                actions=[
                    "logs:CreateLogStream",
                    "logs:DescribeLogStreams",
                    "logs:PutLogEvents",
                ],
                resources=[log_group_arn],
            )
        ]
    )


def ecs_task_kms_policy(*, read_key_arns: list[str], write_key_arns: list[str]) -> dict[str, Any]:
    statements = []
    if read_key_arns:
        statements.append(
            _statement(sid="AllowDecryptReads", actions=["kms:Decrypt"], resources=read_key_arns)
        )
    if write_key_arns:
        statements.append(
            _statement(
                sid="AllowEncryptWrites",
                actions=["kms:Encrypt", "kms:GenerateDataKey"],
                resources=write_key_arns,
            )
        )
    return _document(statements)


def ecs_task_protection_policy(*, cluster_arn: str) -> dict[str, Any]:
    task_resource_arn = cluster_arn.replace(":cluster/", ":task/", 1) + "/*"
    return _document(
        [
            _statement(
                sid="AllowTaskProtectionOnTestCluster",
                actions=["ecs:UpdateTaskProtection"],
                resources=[task_resource_arn],
            )
        ]
    )


def codebuild_manual_deploy_policy(
    *,
    ecr_repository_arn: str,
    ecs_service_arn: str,
    ecs_execution_role_arn: str,
    ecs_task_role_arn: str,
) -> dict[str, Any]:
    """Execution-role policy for the environment's CodeBuild project, scoped to
    exactly what `cicd/buildspec.manual.yml` runs on every push-triggered build:
    push a new worker image, then re-register and roll the ECS service onto it.

    This is unrelated to the `ecs_task_*` policies above, which govern the
    worker's own *runtime* role -- this one governs the CI *build* role.
    It deliberately excludes every VPC/ECS/ECR/IAM create-or-delete action that
    the old (now removed) `infrastructure_provisioner_policy` granted: initial
    infra is provisioned once, by hand, via `infra.aws.cli`, not by this
    CodeBuild project. Each environment has its own CodeBuild project and
    execution role (e.g. beta's `nw_assessments_ai_analysis_backend_beta`), so
    call this once per environment with that environment's own ARNs rather
    than sharing one policy document across stages.
    """
    return _document(
        [
            _statement(
                sid="AllowEcrAuth", actions=["ecr:GetAuthorizationToken"], resources=["*"]
            ),
            _statement(
                sid="AllowEcrPushWorkerImage",
                actions=[
                    "ecr:BatchCheckLayerAvailability",
                    "ecr:InitiateLayerUpload",
                    "ecr:UploadLayerPart",
                    "ecr:CompleteLayerUpload",
                    "ecr:PutImage",
                ],
                resources=[ecr_repository_arn],
            ),
            _statement(
                # Task-definition revision numbers are assigned by ECS on
                # register, so this pair can't be scoped to a specific ARN.
                sid="AllowTaskDefinitionDescribeAndRegister",
                actions=["ecs:DescribeTaskDefinition", "ecs:RegisterTaskDefinition"],
                resources=["*"],
            ),
            _statement(
                sid="AllowWorkerServiceUpdate",
                actions=["ecs:UpdateService", "ecs:DescribeServices"],
                resources=[ecs_service_arn],
            ),
            _statement(
                sid="AllowPassEcsRolesOnRegister",
                actions=["iam:PassRole"],
                resources=[ecs_execution_role_arn, ecs_task_role_arn],
                condition={"StringEquals": {"iam:PassedToService": ECS_TASKS_PRINCIPAL}},
            ),
        ]
    )


def backend_request_queue_send_policy(*, request_queue_arn: str, kms_key_arn: str) -> dict[str, Any]:
    """Grants the backend's existing SQS-send IAM user just enough to enqueue AI-review requests.

    That user is the one the Django backend authenticates as via
    ``CUSTOM_AWS_ACCESS_KEY_ID``/``SECRET`` -- it is not provisioned by this
    tool, so this policy only covers the send side of the request queue plus
    the KMS actions SQS needs to encrypt on send.
    """

    return _document(
        [
            _statement(
                sid="AllowRequestQueueSend",
                actions=["sqs:SendMessage"],
                resources=[request_queue_arn],
            ),
            _statement(
                sid="AllowKmsForSend",
                actions=["kms:GenerateDataKey", "kms:Decrypt"],
                resources=[kms_key_arn],
            ),
        ]
    )


def backend_response_queue_receive_policy(*, response_queue_arn: str, kms_key_arn: str) -> dict[str, Any]:
    """Grants the backend's existing Lambda execution role just enough to consume AI-review responses.

    The Lambda function and its role are not provisioned by this tool, so
    this policy only covers the receive side of the response queue plus the
    KMS decrypt action the event-source mapping needs to poll it.
    """

    return _document(
        [
            _statement(
                sid="AllowResponseQueueConsume",
                actions=["sqs:ReceiveMessage", "sqs:DeleteMessage", "sqs:GetQueueAttributes"],
                resources=[response_queue_arn],
            ),
            _statement(
                sid="AllowKmsForReceive", actions=["kms:Decrypt"], resources=[kms_key_arn]
            ),
        ]
    )


def s3_tls_only_and_public_deny_policy(
    *, bucket_arn: str, allowed_role_arns: list[str]
) -> dict[str, Any]:
    statements = [
        {
            "Sid": "DenyInsecureTransport",
            "Effect": "Deny",
            "Principal": "*",
            "Action": "s3:*",
            "Resource": [bucket_arn, f"{bucket_arn}/*"],
            "Condition": {"Bool": {"aws:SecureTransport": "false"}},
        }
    ]
    if allowed_role_arns:
        statements.append(
            {
                "Sid": "AllowApprovedRolesOnly",
                "Effect": "Allow",
                "Principal": {"AWS": allowed_role_arns},
                "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket"],
                "Resource": [bucket_arn, f"{bucket_arn}/*"],
            }
        )
    return _document(statements)


def sqs_queue_policy(*, queue_arn: str, allowed_role_arns: list[str]) -> dict[str, Any]:
    return _document(
        [
            {
                "Sid": "AllowApprovedRolesOnly",
                "Effect": "Allow",
                "Principal": {"AWS": allowed_role_arns},
                "Action": [
                    "sqs:SendMessage",
                    "sqs:ReceiveMessage",
                    "sqs:DeleteMessage",
                    "sqs:ChangeMessageVisibility",
                    "sqs:GetQueueAttributes",
                ],
                "Resource": queue_arn,
            }
        ]
    )


def kms_key_policy(
    *, account_id: str, admin_role_arn: str, user_role_arns: list[str]
) -> dict[str, Any]:
    statements = [
        {
            "Sid": "AllowRootAccountAdmin",
            "Effect": "Allow",
            "Principal": {"AWS": f"arn:aws:iam::{account_id}:root"},
            "Action": "kms:*",
            "Resource": "*",
        },
        {
            "Sid": "AllowKeyAdministration",
            "Effect": "Allow",
            "Principal": {"AWS": admin_role_arn},
            "Action": [
                "kms:Create*",
                "kms:Describe*",
                "kms:Enable*",
                "kms:List*",
                "kms:Put*",
                "kms:Update*",
                "kms:Revoke*",
                "kms:Disable*",
                "kms:Get*",
                "kms:Delete*",
                "kms:TagResource",
                "kms:UntagResource",
                "kms:ScheduleKeyDeletion",
                "kms:CancelKeyDeletion",
            ],
            "Resource": "*",
        },
    ]
    if user_role_arns:
        statements.append(
            {
                "Sid": "AllowKeyUseByRuntimeRoles",
                "Effect": "Allow",
                "Principal": {"AWS": user_role_arns},
                "Action": ["kms:Decrypt", "kms:Encrypt", "kms:GenerateDataKey", "kms:DescribeKey"],
                "Resource": "*",
            }
        )
    return _document(statements)


def secrets_manager_resource_policy(*, allowed_role_arns: list[str]) -> dict[str, Any]:
    return _document(
        [
            {
                "Sid": "AllowApprovedRolesOnly",
                "Effect": "Allow",
                "Principal": {"AWS": allowed_role_arns},
                "Action": "secretsmanager:GetSecretValue",
                "Resource": "*",
            }
        ]
    )
