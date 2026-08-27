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


def ecs_task_sqs_policy(*, request_queue_arn: str, result_queue_arn: str) -> dict[str, Any]:
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
                sid="AllowResultQueuePublish",
                actions=["sqs:SendMessage"],
                resources=[result_queue_arn],
            ),
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
    return _document(
        [
            _statement(
                sid="AllowTaskProtectionOnTestCluster",
                actions=["ecs:UpdateTaskProtection"],
                resources=[f"{cluster_arn}/*"],
            )
        ]
    )


def infrastructure_provisioner_policy(
    *,
    resource_prefix: str,
    region: str,
    account_id: str,
    execution_role_arn: str,
    task_role_arn: str,
) -> dict[str, Any]:
    statements = [
        _statement(
            sid="AllowUnscopableActions",
            actions=[
                # EC2 networking: create + describe only. Network deletes
                # (DeleteVpc/DeleteSubnet/DetachInternetGateway/DeleteNatGateway/
                # ReleaseAddress/DeleteRouteTable/DisassociateRouteTable/
                # DeleteSecurityGroup/DeleteVpcEndpoints/DeleteTags) are
                # deliberately excluded; network teardown is the admin's job.
                # All EC2 actions stay on "*": create calls return AWS-assigned
                # IDs that don't exist yet, and Describe* is not resource-level.
                "ec2:Describe*",
                "ec2:CreateTags",
                "ec2:CreateVpc",
                "ec2:ModifyVpcAttribute",
                "ec2:CreateSubnet",
                "ec2:ModifySubnetAttribute",
                "ec2:CreateInternetGateway",
                "ec2:AttachInternetGateway",
                "ec2:AllocateAddress",
                "ec2:CreateNatGateway",
                "ec2:CreateRouteTable",
                "ec2:CreateRoute",
                "ec2:AssociateRouteTable",
                "ec2:CreateSecurityGroup",
                "ec2:AuthorizeSecurityGroupEgress",
                "ec2:RevokeSecurityGroupEgress",
                "ec2:CreateVpcEndpoint",
                # ECR authorization tokens are account-level, not repo-scoped;
                # this same role also runs `docker login`/push for the worker image.
                "ecr:GetAuthorizationToken",
                # KMS keys are UUID ARNs, never name-scopable to resource_prefix.
                # Decrypt/GenerateDataKey are also needed here for the state
                # manifest (SSE-KMS S3 object) on every apply/destroy/status call.
                "kms:ListAliases",
                "kms:CreateKey",
                "kms:CreateAlias",
                "kms:PutKeyPolicy",
                "kms:DescribeKey",
                "kms:DisableKey",
                "kms:ScheduleKeyDeletion",
                "kms:GenerateDataKey",
                "kms:Decrypt",
                # CreateKey with Tags also checks kms:TagResource.
                "kms:TagResource",
                # ECS list/describe and task-definition register/deregister are
                # not usefully resource-scoped (task-def revision is assigned
                # on register; describes/lists are account-wide).
                "ecs:RegisterTaskDefinition",
                "ecs:DeregisterTaskDefinition",
                "ecs:ListTaskDefinitions",
                "ecs:DescribeClusters",
                "ecs:DescribeServices",
                "ecs:DescribeTasks",
                "ecs:ListTasks",
                "logs:DescribeLogGroups",
                "cloudwatch:DescribeAlarms",
                # Application Auto Scaling has no resource-level permissions for
                # these actions (ResourceId is an API parameter, not an ARN).
                "application-autoscaling:DescribeScalableTargets",
                "application-autoscaling:DescribeScalingPolicies",
                "application-autoscaling:DescribeScalingActivities",
                "application-autoscaling:RegisterScalableTarget",
                "application-autoscaling:DeregisterScalableTarget",
                "application-autoscaling:PutScalingPolicy",
                "application-autoscaling:DeleteScalingPolicy",
                # RegisterScalableTarget with Tags also checks this action.
                "application-autoscaling:TagResource",
            ],
            resources=["*"],
        ),
        _statement(
            sid="AllowManageTaggedTestResources",
            actions=[
                "ecr:*",
                "ecs:CreateCluster",
                "ecs:DeleteCluster",
                "ecs:CreateService",
                "ecs:UpdateService",
                "ecs:DeleteService",
                "ecs:GetTaskProtection",
                "s3:*",
                "sqs:*",
                "secretsmanager:*",
                "logs:CreateLogGroup",
                "logs:PutRetentionPolicy",
                "logs:DeleteLogGroup",
                "logs:StartQuery",
                "logs:GetQueryResults",
                # CreateLogGroup with tags also checks logs:TagResource.
                "logs:TagResource",
                "cloudwatch:PutMetricAlarm",
                "cloudwatch:DeleteAlarms",
                # PutMetricAlarm with Tags also checks cloudwatch:TagResource.
                "cloudwatch:TagResource",
            ],
            resources=[f"arn:aws:*:*:*:*{resource_prefix}*"],
        ),
        _statement(
            sid="AllowIamRoleManagementForTestRoles",
            actions=[
                "iam:CreateRole",
                "iam:UpdateRole",
                "iam:DeleteRole",
                "iam:GetRole",
                "iam:PutRolePolicy",
                "iam:DeleteRolePolicy",
                "iam:GetRolePolicy",
                "iam:TagRole",
            ],
            resources=[f"arn:aws:iam::{account_id}:role/{resource_prefix}-*"],
        ),
        _statement(
            sid="AllowPassEcsRoles",
            actions=["iam:PassRole"],
            resources=[execution_role_arn, task_role_arn],
            condition={"StringEquals": {"iam:PassedToService": ECS_TASKS_PRINCIPAL}},
        ),
        _statement(
            sid="AllowCreateEcsAutoscalingServiceLinkedRole",
            actions=["iam:CreateServiceLinkedRole"],
            resources=[
                f"arn:aws:iam::{account_id}:role/aws-service-role/"
                "ecs.application-autoscaling.amazonaws.com/"
                "AWSServiceRoleForApplicationAutoScaling_ECSService"
            ],
            condition={
                "StringEquals": {"iam:AWSServiceName": "ecs.application-autoscaling.amazonaws.com"}
            },
        ),
        _statement(
            sid="AllowCallerIdentityGuard", actions=["sts:GetCallerIdentity"], resources=["*"]
        ),
    ]
    return _document(statements)


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
