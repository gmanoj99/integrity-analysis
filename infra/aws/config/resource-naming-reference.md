# Resource naming reference (for admin/enterprise-standards review)

Every name this provisioner creates is derived from `resource_prefix = f"{project}-{environment}"`
(see `EnvironmentConfig.resource_prefix` in [`infra/aws/config.py`](config.py)). For
the two environments currently defined:

- `ecs-test` -> prefix `integrity-review-ecs-test`
- `beta` -> prefix `integrity-review-beta`

This file is a flat inventory for a naming-convention review; it is not
consumed by any code. Update it whenever a name pattern changes in
`orchestrator.resource_names()` or `policies.py`.

## 1. AWS resource names (`orchestrator.resource_names()`)

| Key | Pattern | AWS resource |
| --- | --- | --- |
| `vpc` | `{prefix}-vpc` | VPC |
| `public_subnet_0` / `public_subnet_1` | `{prefix}-public-0` / `-1` | Public subnets |
| `private_subnet_0` / `private_subnet_1` | `{prefix}-private-0` / `-1` | Private subnets |
| `internet_gateway` | `{prefix}-igw` | Internet gateway |
| `nat_gateway` | `{prefix}-nat` | NAT gateway |
| `public_route_table` / `private_route_table` | `{prefix}-public-rt` / `-private-rt` | Route tables |
| `security_group` | `{prefix}-worker-sg` | ECS worker security group |
| `kms_data` | `{prefix}-data` | KMS alias (S3/SQS/artifact encryption) |
| `kms_secrets` | `{prefix}-secrets` | KMS alias (Secrets Manager) |
| `kms_logs` | `{prefix}-logs` | KMS alias (CloudWatch Logs) |
| `media_bucket` | `{prefix}-media-{account_id}` | S3 bucket |
| `request_bucket` | `{prefix}-requests-{account_id}` | S3 bucket |
| `result_bucket` | `{prefix}-results-{account_id}` | S3 bucket |
| `request_queue` / `request_dlq` | `{prefix}-request` / `{prefix}-request-dlq` | SQS queues |
| `result_queue` / `result_dlq` | `{prefix}-result` / `{prefix}-result-dlq` | SQS queues |
| `log_group` | `/ecs/{prefix}-worker` | CloudWatch Logs group (worker) |
| `ecr_repository` | `{prefix}-worker` | ECR repository |
| `gemini_secret` | `{prefix}/gemini-api-key` | Secrets Manager secret |
| `execution_role` | `{prefix}-ecs-execution` | IAM role |
| `task_role` | `{prefix}-ecs-task` | IAM role |
| `image_publisher_role` | `{prefix}-image-publisher` | IAM role |
| `test_operator_role` | `{prefix}-test-operator` | IAM role |
| `cluster` | `{prefix}-cluster` | ECS cluster |
| `task_family` | `{prefix}-worker` | ECS task definition family |
| `service` | `{prefix}-worker-service` | ECS service |
| `source_repository` | `{prefix}-source` | CodeCommit repository |
| `pipeline_artifact_bucket` | `{prefix}-pipeline-artifacts-{account_id}` | S3 bucket (pipeline artifacts) |
| `build_log_group` | `/aws/codebuild/{prefix}-build` | CloudWatch Logs group |
| `deploy_log_group` | `/aws/codebuild/{prefix}-deploy` | CloudWatch Logs group |
| `codebuild_role` | `{prefix}-codebuild` | IAM role (image build+push) |
| `deploy_role` | `{prefix}-deploy-runner` | IAM role (runs provisioner apply) |
| `build_project` | `{prefix}-build` | CodeBuild project |
| `deploy_project` | `{prefix}-deploy` | CodeBuild project |
| `codepipeline_role` | `{prefix}-codepipeline` | IAM role |
| `pipeline` | `{prefix}-pipeline` | CodePipeline pipeline |
| `events_role` | `{prefix}-pipeline-trigger` | IAM role (EventBridge target invocation) |
| `pipeline_trigger_rule` | `{prefix}-pipeline-trigger` | EventBridge rule |

Alarm and autoscaling-policy names (`orchestrator._alarm_specs` / `_ensure_autoscaling`),
all `{prefix}-...`:

- `{prefix}-ecs-running-below-desired`
- `{prefix}-request-dlq-not-empty`
- `{prefix}-result-dlq-not-empty`
- `{prefix}-request-queue-age`
- `{prefix}-scale-out-on-backlog` (alarm + step-scaling policy, same name)
- `{prefix}-scale-in-on-idle` (alarm + step-scaling policy, same name)

## 2. IAM roles and their inline policy names

| Role (`resource_names()` key) | Inline policy names (`RoleSpec.inline_policies`) |
| --- | --- |
| `execution_role` | `execution` |
| `task_role` | `s3`, `sqs`, `kms`, `task_protection` |
| `image_publisher_role` | `ecr_push` |
| `test_operator_role` | `operator` |
| `codebuild_role` | `build` |
| `deploy_role` | `deploy`, `provisioner` |
| `codepipeline_role` | `pipeline` |
| `events_role` | `trigger` |

The human/CI *provisioner principal* (whoever runs `infra.aws.cli`) is not a
role created by this tool; it is granted permissions out-of-band via the
static [`operator-iam-policy.json`](operator-iam-policy.json) (`ecs-test`) /
[`operator-iam-policy.beta.json`](operator-iam-policy.beta.json) (`beta`), which
must be kept in sync with `policies.infrastructure_provisioner_policy()`.

## 3. IAM policy `Sid`s (`infra/aws/policies.py`)

| Function | Statement `Sid`s |
| --- | --- |
| `ecs_task_trust_policy` | `AllowEcsTasksAssume` |
| `assume_role_trust_policy` | `AllowNamedPrincipalsAssume` |
| `oidc_trust_policy` | `AllowGithubOidc` |
| `codebuild_trust_policy` | `AllowCodeBuildAssume` |
| `codepipeline_trust_policy` | `AllowCodePipelineAssume` |
| `events_trust_policy` | `AllowEventsAssume` |
| `ecs_task_execution_role_policy` | `AllowEcrAuth`, `AllowEcrPull`, `AllowLogDelivery`, `AllowGeminiSecretRead`, `AllowKmsDecrypt` |
| `ecs_task_s3_policy` | `AllowMediaRead`, `AllowRequestRead`, `AllowResultReadWrite`, `AllowResultPrefixList` |
| `ecs_task_sqs_policy` | `AllowRequestQueueConsume`, `AllowResultQueuePublish` |
| `ecs_task_kms_policy` | `AllowDecryptReads`, `AllowEncryptWrites` |
| `ecs_task_protection_policy` | `AllowTaskProtectionOnTestCluster` |
| `infrastructure_provisioner_policy` | `AllowUnscopableActions`, `AllowManageTaggedTestResources`, `AllowIamRoleManagementForTestRoles`, `AllowPassEcsRoles`, `AllowCreateEcsAutoscalingServiceLinkedRole`, `AllowAssumeImagePublisher`, `AllowCallerIdentityGuard`, `AllowPassCiCdRoles` (only when `cicd_role_arns` is given) |
| `image_publisher_policy` | `AllowEcrAuth`, `AllowEcrPush` |
| `test_operator_policy` | `AllowUploadApprovedFixtures`, `AllowReadResults`, `AllowSendRequest`, `AllowReadResultQueue`, `AllowInspectEcs`, `AllowReadLogs` |
| `codebuild_build_role_policy` | `AllowLogDelivery`, `AllowEcrAuth`, `AllowEcrPush`, `AllowArtifactBucketAccess`, `AllowArtifactKmsUse`, `AllowSourceGitPull` |
| `codebuild_deploy_role_policy` | `AllowLogDelivery`, `AllowArtifactBucketRead`, `AllowArtifactKmsDecrypt` |
| `codepipeline_service_role_policy` | `AllowSourceRead`, `AllowArtifactBucketAccess`, `AllowArtifactKmsUse`, `AllowInvokeCodeBuild` |
| `pipeline_trigger_events_policy` | `AllowStartPipelineExecution` |
| `s3_tls_only_and_public_deny_policy` | `DenyInsecureTransport`, `AllowApprovedRolesOnly` |
| `sqs_queue_policy` | `AllowApprovedRolesOnly` |
| `kms_key_policy` | `AllowRootAccountAdmin`, `AllowKeyAdministration`, `AllowKeyUseByRuntimeRoles` |
| `secrets_manager_resource_policy` | `AllowApprovedRolesOnly` |

Note the same `Sid` (`AllowLogDelivery`, `AllowEcrAuth`, `AllowEcrPush`,
`AllowApprovedRolesOnly`, `AllowArtifactBucketAccess`, `AllowArtifactKmsUse`)
is deliberately reused across unrelated policy *documents* (each `Sid` only
needs to be unique within its own document); flag if the enterprise standard
requires globally unique `Sid`s instead.

## 4. CI/CD pipeline artifact and stage names

- CodeCommit branch that triggers the pipeline: `main` (`orchestrator._PIPELINE_SOURCE_BRANCH`)
- CodePipeline stages: `Source` -> `Build` -> `Deploy`
- CodePipeline output artifacts: `SourceOutput`, `BuildOutput`
- EventBridge target id: `{prefix}-pipeline-trigger-target`
- Buildspec paths: `infra/aws/cicd/buildspec.build.yml`, `infra/aws/cicd/buildspec.deploy.yml`

## 5. Tag keys (`EnvironmentConfig.resolved_tags()`)

Always present: `Project`, `Environment`, `Owner`, `CostCenter`, `ManagedBy` (fixed
value `infra.aws`). Per-environment extras from each config's `tags` map, currently:
`DataClassification`, `Expiry`.

## 6. Config file names

- `infra/aws/config/ecs-test.json`, `ecs-test.example.json` -> environment `ecs-test`
- `infra/aws/config/beta.json` -> environment `beta`
- `infra/aws/config/operator-iam-policy.json` -> static operator policy for `ecs-test`
- `infra/aws/config/operator-iam-policy.beta.json` -> static operator policy for `beta`

## Review checklist for admins

- [ ] Confirm `integrity-review` as the enterprise-approved project slug (all resource names inherit it).
- [ ] Confirm environment slugs `ecs-test` / `beta` match the org's environment-naming standard (vs. e.g. `dev`/`stg`/`prod`).
- [ ] Confirm CI/CD role names (`{prefix}-codebuild`, `{prefix}-deploy-runner`, `{prefix}-codepipeline`, `{prefix}-pipeline-trigger`) meet any required role-naming prefix/suffix convention (e.g. `role-`, `-svc`, department code).
- [ ] Confirm KMS alias names (`{prefix}-data`, `-secrets`, `-logs`) don't collide with an existing account-wide KMS naming scheme.
- [ ] Confirm tag keys (`Project`, `Environment`, `Owner`, `CostCenter`, `ManagedBy`, `DataClassification`, `Expiry`) match required cost-allocation/compliance tag keys.
- [ ] Confirm the CodeCommit branch name (`main`) matches the org's default-branch standard.
