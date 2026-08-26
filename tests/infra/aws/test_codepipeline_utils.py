"""botocore Stubber coverage for CodePipeline provisioning and status reads."""

from __future__ import annotations

from infra.aws.specs import (
    CodePipelineArtifactStoreSpec,
    CodePipelineBuildStageSpec,
    CodePipelineSourceStageSpec,
    CodePipelineSpec,
)
from infra.aws.utils import codepipeline_utils

from .aws_stub import stubbed_client

PIPELINE_NAME = "integrity-review-beta-pipeline"
CODEPIPELINE_ROLE_ARN = "arn:aws:iam::111111111111:role/integrity-review-beta-codepipeline"


def _spec() -> CodePipelineSpec:
    return CodePipelineSpec(
        name=PIPELINE_NAME,
        service_role_arn=CODEPIPELINE_ROLE_ARN,
        artifact_store=CodePipelineArtifactStoreSpec(
            bucket_name="integrity-review-beta-pipeline-artifacts-111111111111",
            kms_key_arn="arn:aws:kms:ap-south-1:111111111111:key/abc",
        ),
        source=CodePipelineSourceStageSpec(
            repository_name="integrity-review-beta-source",
            branch_name="main",
            output_artifact_name="SourceOutput",
        ),
        build_stages=(
            CodePipelineBuildStageSpec(
                name="Build",
                project_name="integrity-review-beta-build",
                input_artifact_name="SourceOutput",
                output_artifact_name="BuildOutput",
            ),
            CodePipelineBuildStageSpec(
                name="Deploy",
                project_name="integrity-review-beta-deploy",
                input_artifact_name="BuildOutput",
                output_artifact_name=None,
            ),
        ),
    )


def _pipeline_response() -> dict:
    return {
        "pipeline": {
            "name": PIPELINE_NAME,
            "roleArn": CODEPIPELINE_ROLE_ARN,
            "artifactStore": {"type": "S3", "location": "integrity-review-beta-pipeline-artifacts"},
            "stages": [
                {
                    "name": "Source",
                    "actions": [
                        {
                            "name": "Source",
                            "actionTypeId": {
                                "category": "Source",
                                "owner": "AWS",
                                "provider": "CodeCommit",
                                "version": "1",
                            },
                        }
                    ],
                }
            ],
        }
    }


def test_pipeline_exists_false_when_not_found() -> None:
    client, stubber = stubbed_client("codepipeline")
    stubber.add_client_error(
        "get_pipeline", service_error_code="PipelineNotFoundException", http_status_code=400
    )
    with stubber:
        assert codepipeline_utils.pipeline_exists(client, PIPELINE_NAME) is False


def test_ensure_pipeline_creates_when_missing_and_scopes_source_polling_off() -> None:
    client, stubber = stubbed_client("codepipeline")
    stubber.add_client_error(
        "get_pipeline", service_error_code="PipelineNotFoundException", http_status_code=400
    )
    stubber.add_response("create_pipeline", _pipeline_response())

    with stubber:
        arn = codepipeline_utils.ensure_pipeline(client, _spec(), {"Project": "integrity-review"})

    assert arn == f"arn:aws:codepipeline:ap-south-1:111111111111:{PIPELINE_NAME}"
    stubber.assert_no_pending_responses()


def test_ensure_pipeline_updates_when_existing() -> None:
    client, stubber = stubbed_client("codepipeline")
    stubber.add_response("get_pipeline", _pipeline_response())
    stubber.add_response("update_pipeline", _pipeline_response())

    with stubber:
        arn = codepipeline_utils.ensure_pipeline(client, _spec(), {"Project": "integrity-review"})

    assert arn == f"arn:aws:codepipeline:ap-south-1:111111111111:{PIPELINE_NAME}"
    stubber.assert_no_pending_responses()


def test_latest_execution_status_none_when_no_executions() -> None:
    client, stubber = stubbed_client("codepipeline")
    stubber.add_response(
        "list_pipeline_executions",
        {"pipelineExecutionSummaries": []},
        {"pipelineName": PIPELINE_NAME, "maxResults": 1},
    )
    with stubber:
        assert codepipeline_utils.latest_execution_status(client, PIPELINE_NAME) is None
