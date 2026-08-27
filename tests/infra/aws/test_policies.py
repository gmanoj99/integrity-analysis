"""Shape assertions for the generated IAM/S3/SQS/KMS policy documents."""

from __future__ import annotations

from infra.aws import policies


def _actions(document: dict, sid: str) -> list[str]:
    statement = next(s for s in document["Statement"] if s["Sid"] == sid)
    actions = statement["Action"]
    return actions if isinstance(actions, list) else [actions]


def test_ecs_task_trust_policy_only_allows_ecs_tasks() -> None:
    document = policies.ecs_task_trust_policy()

    statement = document["Statement"][0]
    assert statement["Principal"] == {"Service": "ecs-tasks.amazonaws.com"}
    assert statement["Action"] == "sts:AssumeRole"


def test_task_execution_policy_scopes_secret_and_ecr() -> None:
    document = policies.ecs_task_execution_role_policy(
        repository_arn="arn:aws:ecr:ap-south-1:111111111111:repository/test-worker",
        log_group_arn="arn:aws:logs:ap-south-1:111111111111:log-group:/ecs/test-worker",
        gemini_secret_arn="arn:aws:secretsmanager:ap-south-1:111111111111:secret:test/gemini-api-key",
        kms_key_arns=["arn:aws:kms:ap-south-1:111111111111:key/abc"],
    )

    secret_statement = next(
        s for s in document["Statement"] if s["Sid"] == "AllowGeminiSecretRead"
    )
    assert secret_statement["Resource"] == [
        "arn:aws:secretsmanager:ap-south-1:111111111111:secret:test/gemini-api-key"
    ]
    assert "secretsmanager:GetSecretValue" in secret_statement["Action"]


def test_task_s3_policy_restricts_to_stage_prefixes() -> None:
    document = policies.ecs_task_s3_policy(
        storage_bucket_arn="arn:aws:s3:::storage-bucket",
        stage="beta",
    )

    request_statement = next(s for s in document["Statement"] if s["Sid"] == "AllowRequestRead")
    assert request_statement["Resource"] == [
        "arn:aws:s3:::storage-bucket/beta/media/ai_integrity_review_requests/*"
    ]
    recording_statement = next(s for s in document["Statement"] if s["Sid"] == "AllowRecordingRead")
    assert recording_statement["Resource"] == ["arn:aws:s3:::storage-bucket/beta/media/*"]
    result_statement = next(s for s in document["Statement"] if s["Sid"] == "AllowResultReadWrite")
    assert result_statement["Action"] == ["s3:GetObject", "s3:PutObject"]
    list_statement = next(s for s in document["Statement"] if s["Sid"] == "AllowResultPrefixList")
    assert list_statement["Action"] == ["s3:ListBucket"]
    assert "s3:HeadObject" not in str(document)


def test_task_sqs_policy_separates_consume_and_publish() -> None:
    document = policies.ecs_task_sqs_policy(
        request_queue_arn="arn:aws:sqs:ap-south-1:111111111111:test-request",
        result_queue_arn="arn:aws:sqs:ap-south-1:111111111111:test-result",
    )

    consume = _actions(document, "AllowRequestQueueConsume")
    publish = _actions(document, "AllowResultQueuePublish")
    assert set(consume) == {"sqs:ReceiveMessage", "sqs:DeleteMessage", "sqs:ChangeMessageVisibility"}
    assert publish == ["sqs:SendMessage"]


def test_task_protection_policy_scopes_to_cluster() -> None:
    document = policies.ecs_task_protection_policy(
        cluster_arn="arn:aws:ecs:ap-south-1:111111111111:cluster/test-cluster"
    )

    statement = document["Statement"][0]
    assert statement["Action"] == ["ecs:UpdateTaskProtection"]
    assert statement["Resource"] == ["arn:aws:ecs:ap-south-1:111111111111:cluster/test-cluster/*"]


def _provisioner_policy() -> dict:
    return policies.infrastructure_provisioner_policy(
        resource_prefix="integrity-review-beta",
        region="ap-south-1",
        account_id="111111111111",
        execution_role_arn="arn:aws:iam::111111111111:role/integrity-review-beta-ecs-execution",
        task_role_arn="arn:aws:iam::111111111111:role/integrity-review-beta-ecs-task",
        image_publisher_role_arn="arn:aws:iam::111111111111:role/integrity-review-beta-image-publisher",
    )


def test_provisioner_policy_scopes_pass_role_to_ecs_tasks() -> None:
    document = _provisioner_policy()

    pass_role_statement = next(s for s in document["Statement"] if s["Sid"] == "AllowPassEcsRoles")
    assert pass_role_statement["Condition"] == {
        "StringEquals": {"iam:PassedToService": "ecs-tasks.amazonaws.com"}
    }
    assert set(pass_role_statement["Resource"]) == {
        "arn:aws:iam::111111111111:role/integrity-review-beta-ecs-execution",
        "arn:aws:iam::111111111111:role/integrity-review-beta-ecs-task",
    }


def test_provisioner_policy_has_no_dynamodb() -> None:
    document = _provisioner_policy()

    all_actions = [action for statement in document["Statement"] for action in statement["Action"]]
    assert not any(action.startswith("dynamodb:") for action in all_actions)


def test_provisioner_policy_scopes_cicd_actions_to_tagged_resources() -> None:
    document = _provisioner_policy()

    tagged_statement = next(
        s for s in document["Statement"] if s["Sid"] == "AllowManageTaggedTestResources"
    )
    for action in (
        "codecommit:CreateRepository",
        "codecommit:GitPull",
        "codebuild:CreateProject",
        "codebuild:StartBuild",
        "codepipeline:CreatePipeline",
        "codepipeline:StartPipelineExecution",
        "events:PutRule",
        "events:PutTargets",
    ):
        assert action in tagged_statement["Action"]
    assert tagged_statement["Resource"] == ["arn:aws:*:*:*:*integrity-review-beta*"]


def test_provisioner_policy_omits_pass_cicd_roles_statement_when_no_roles_given() -> None:
    document = _provisioner_policy()

    assert not any(s["Sid"] == "AllowPassCiCdRoles" for s in document["Statement"])


def test_provisioner_policy_scopes_pass_role_to_cicd_roles_when_provided() -> None:
    document = policies.infrastructure_provisioner_policy(
        resource_prefix="integrity-review-beta",
        region="ap-south-1",
        account_id="111111111111",
        execution_role_arn="arn:aws:iam::111111111111:role/integrity-review-beta-ecs-execution",
        task_role_arn="arn:aws:iam::111111111111:role/integrity-review-beta-ecs-task",
        image_publisher_role_arn="arn:aws:iam::111111111111:role/integrity-review-beta-image-publisher",
        cicd_role_arns=(
            "arn:aws:iam::111111111111:role/integrity-review-beta-codebuild",
            "arn:aws:iam::111111111111:role/integrity-review-beta-deploy-runner",
        ),
    )

    statement = next(s for s in document["Statement"] if s["Sid"] == "AllowPassCiCdRoles")
    assert set(statement["Resource"]) == {
        "arn:aws:iam::111111111111:role/integrity-review-beta-codebuild",
        "arn:aws:iam::111111111111:role/integrity-review-beta-deploy-runner",
    }
    assert set(statement["Condition"]["StringEquals"]["iam:PassedToService"]) == {
        "codebuild.amazonaws.com",
        "codepipeline.amazonaws.com",
        "events.amazonaws.com",
    }


def test_provisioner_policy_grants_application_autoscaling_for_sqs_driven_scaling() -> None:
    document = _provisioner_policy()

    all_actions = [action for statement in document["Statement"] for action in statement["Action"]]
    expected_actions = {
        "application-autoscaling:DescribeScalableTargets",
        "application-autoscaling:DescribeScalingPolicies",
        "application-autoscaling:DescribeScalingActivities",
        "application-autoscaling:RegisterScalableTarget",
        "application-autoscaling:DeregisterScalableTarget",
        "application-autoscaling:PutScalingPolicy",
        "application-autoscaling:DeleteScalingPolicy",
    }
    assert expected_actions.issubset(all_actions)

    unscopable_statement = next(
        s for s in document["Statement"] if s["Sid"] == "AllowUnscopableActions"
    )
    assert unscopable_statement["Resource"] == ["*"]


def test_provisioner_policy_scopes_service_linked_role_creation_to_ecs_autoscaling() -> None:
    document = _provisioner_policy()

    statement = next(
        s
        for s in document["Statement"]
        if s["Sid"] == "AllowCreateEcsAutoscalingServiceLinkedRole"
    )
    assert statement["Action"] == ["iam:CreateServiceLinkedRole"]
    assert statement["Resource"] == [
        "arn:aws:iam::111111111111:role/aws-service-role/"
        "ecs.application-autoscaling.amazonaws.com/"
        "AWSServiceRoleForApplicationAutoScaling_ECSService"
    ]
    assert statement["Condition"] == {
        "StringEquals": {"iam:AWSServiceName": "ecs.application-autoscaling.amazonaws.com"}
    }


def test_provisioner_policy_has_no_wildcard_resource_on_data_services() -> None:
    document = _provisioner_policy()

    for statement in document["Statement"]:
        if any(action.endswith(":*") for action in statement["Action"]):
            assert statement["Resource"] != ["*"], (
                f"{statement['Sid']} grants a wildcard service action against Resource '*'"
            )

    unscoped_sids = {
        statement["Sid"] for statement in document["Statement"] if statement["Resource"] == ["*"]
    }
    assert unscoped_sids == {"AllowUnscopableActions", "AllowCallerIdentityGuard"}

    tagged_statement = next(
        s for s in document["Statement"] if s["Sid"] == "AllowManageTaggedTestResources"
    )
    assert tagged_statement["Resource"] == ["arn:aws:*:*:*:*integrity-review-beta*"]


def test_provisioner_policy_excludes_ec2_network_deletes() -> None:
    document = _provisioner_policy()

    all_actions = [action for statement in document["Statement"] for action in statement["Action"]]
    deleted_ec2_actions = (
        "ec2:DeleteVpc", "ec2:DeleteSubnet", "ec2:DeleteInternetGateway", "ec2:DetachInternetGateway",
        "ec2:DeleteNatGateway", "ec2:ReleaseAddress", "ec2:DeleteRouteTable", "ec2:DisassociateRouteTable",
        "ec2:DeleteSecurityGroup", "ec2:DeleteVpcEndpoints", "ec2:DeleteTags",
    )
    assert not any(action in deleted_ec2_actions for action in all_actions)
    assert "ec2:AllocateAddress" in all_actions


def test_provisioner_policy_scopes_ecr_to_tagged_resources_and_grants_assume_role() -> None:
    document = _provisioner_policy()

    tagged_statement = next(
        s for s in document["Statement"] if s["Sid"] == "AllowManageTaggedTestResources"
    )
    assert "ecr:*" in tagged_statement["Action"]
    assert tagged_statement["Resource"] == ["arn:aws:*:*:*:*integrity-review-beta*"]

    assume_statement = next(s for s in document["Statement"] if s["Sid"] == "AllowAssumeImagePublisher")
    assert assume_statement["Action"] == ["sts:AssumeRole"]
    assert assume_statement["Resource"] == [
        "arn:aws:iam::111111111111:role/integrity-review-beta-image-publisher"
    ]


def test_assume_role_trust_policy_trusts_named_principals() -> None:
    document = policies.assume_role_trust_policy(
        trusted_principal_arns=["arn:aws:iam::111111111111:user/operator"]
    )

    statement = document["Statement"][0]
    assert statement["Principal"] == {"AWS": ["arn:aws:iam::111111111111:user/operator"]}
    assert statement["Action"] == "sts:AssumeRole"


def test_image_publisher_policy_has_no_application_data_access() -> None:
    document = policies.image_publisher_policy(repository_arn="arn:aws:ecr:ap-south-1:111111111111:repository/test-worker")

    all_actions = [action for statement in document["Statement"] for action in statement["Action"]]
    assert not any(action.startswith("s3:") for action in all_actions)
    assert not any(action.startswith("secretsmanager:") for action in all_actions)


def test_s3_tls_only_policy_denies_insecure_transport() -> None:
    document = policies.s3_tls_only_and_public_deny_policy(
        bucket_arn="arn:aws:s3:::test-bucket", allowed_role_arns=["arn:aws:iam::111111111111:role/task"]
    )

    deny_statement = next(s for s in document["Statement"] if s["Sid"] == "DenyInsecureTransport")
    assert deny_statement["Effect"] == "Deny"
    assert deny_statement["Condition"] == {"Bool": {"aws:SecureTransport": "false"}}


def test_codebuild_trust_policy_only_allows_codebuild() -> None:
    document = policies.codebuild_trust_policy()

    statement = document["Statement"][0]
    assert statement["Principal"] == {"Service": "codebuild.amazonaws.com"}
    assert statement["Action"] == "sts:AssumeRole"


def test_codepipeline_trust_policy_only_allows_codepipeline() -> None:
    document = policies.codepipeline_trust_policy()

    statement = document["Statement"][0]
    assert statement["Principal"] == {"Service": "codepipeline.amazonaws.com"}


def test_events_trust_policy_only_allows_events() -> None:
    document = policies.events_trust_policy()

    statement = document["Statement"][0]
    assert statement["Principal"] == {"Service": "events.amazonaws.com"}


def test_codebuild_build_role_policy_scopes_ecr_push_and_source_pull() -> None:
    document = policies.codebuild_build_role_policy(
        repository_arn="arn:aws:ecr:ap-south-1:111111111111:repository/integrity-review-beta-worker",
        log_group_arn="arn:aws:logs:ap-south-1:111111111111:log-group:/aws/codebuild/integrity-review-beta-build",
        artifact_bucket_arn="arn:aws:s3:::integrity-review-beta-pipeline-artifacts-111111111111",
        kms_key_arn="arn:aws:kms:ap-south-1:111111111111:key/abc",
        source_repository_arn="arn:aws:codecommit:ap-south-1:111111111111:integrity-review-beta-source",
    )

    push_statement = next(s for s in document["Statement"] if s["Sid"] == "AllowEcrPush")
    assert push_statement["Resource"] == [
        "arn:aws:ecr:ap-south-1:111111111111:repository/integrity-review-beta-worker"
    ]
    pull_statement = next(s for s in document["Statement"] if s["Sid"] == "AllowSourceGitPull")
    assert pull_statement["Action"] == ["codecommit:GitPull"]


def test_codebuild_deploy_role_policy_has_no_ecr_or_codecommit_access() -> None:
    document = policies.codebuild_deploy_role_policy(
        log_group_arn="arn:aws:logs:ap-south-1:111111111111:log-group:/aws/codebuild/integrity-review-beta-deploy",
        artifact_bucket_arn="arn:aws:s3:::integrity-review-beta-pipeline-artifacts-111111111111",
        kms_key_arn="arn:aws:kms:ap-south-1:111111111111:key/abc",
    )

    all_actions = [action for statement in document["Statement"] for action in statement["Action"]]
    assert not any(action.startswith("ecr:") for action in all_actions)
    assert not any(action.startswith("codecommit:") for action in all_actions)
    assert "s3:GetObject" in all_actions
    assert "s3:PutObject" not in all_actions


def test_codepipeline_service_role_policy_scopes_invoke_to_given_projects() -> None:
    document = policies.codepipeline_service_role_policy(
        source_repository_arn="arn:aws:codecommit:ap-south-1:111111111111:integrity-review-beta-source",
        artifact_bucket_arn="arn:aws:s3:::integrity-review-beta-pipeline-artifacts-111111111111",
        kms_key_arn="arn:aws:kms:ap-south-1:111111111111:key/abc",
        build_project_arns=[
            "arn:aws:codebuild:ap-south-1:111111111111:project/integrity-review-beta-build",
            "arn:aws:codebuild:ap-south-1:111111111111:project/integrity-review-beta-deploy",
        ],
    )

    invoke_statement = next(s for s in document["Statement"] if s["Sid"] == "AllowInvokeCodeBuild")
    assert len(invoke_statement["Resource"]) == 2


def test_pipeline_trigger_events_policy_scopes_to_single_pipeline() -> None:
    document = policies.pipeline_trigger_events_policy(
        pipeline_arn="arn:aws:codepipeline:ap-south-1:111111111111:integrity-review-beta-pipeline"
    )

    statement = document["Statement"][0]
    assert statement["Action"] == ["codepipeline:StartPipelineExecution"]
    assert statement["Resource"] == [
        "arn:aws:codepipeline:ap-south-1:111111111111:integrity-review-beta-pipeline"
    ]


def test_kms_key_policy_separates_admin_from_runtime_users() -> None:
    document = policies.kms_key_policy(
        account_id="111111111111",
        admin_role_arn="arn:aws:iam::111111111111:role/security-admin",
        user_role_arns=["arn:aws:iam::111111111111:role/ecs-task"],
    )

    admin_statement = next(s for s in document["Statement"] if s["Sid"] == "AllowKeyAdministration")
    user_statement = next(s for s in document["Statement"] if s["Sid"] == "AllowKeyUseByRuntimeRoles")
    assert "kms:ScheduleKeyDeletion" in admin_statement["Action"]
    assert "kms:ScheduleKeyDeletion" not in user_statement["Action"]
    assert user_statement["Action"] == [
        "kms:Decrypt", "kms:Encrypt", "kms:GenerateDataKey", "kms:DescribeKey"
    ]
