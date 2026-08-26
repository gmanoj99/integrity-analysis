"""CodePipeline provisioning: CodeCommit source -> CodeBuild build -> CodeBuild deploy.

Source changes are delivered by the EventBridge trigger rule (see
``events_utils``), not CodePipeline's own polling, so every source action sets
``PollForSourceChanges: "false"``.
"""

from __future__ import annotations

from typing import Any

from botocore.exceptions import ClientError

from ..specs import CodePipelineSpec


def _source_stage(spec: CodePipelineSpec) -> dict[str, Any]:
    return {
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
                "configuration": {
                    "RepositoryName": spec.source.repository_name,
                    "BranchName": spec.source.branch_name,
                    "PollForSourceChanges": "false",
                },
                "outputArtifacts": [{"name": spec.source.output_artifact_name}],
            }
        ],
    }


def _build_stage(stage: Any) -> dict[str, Any]:
    action: dict[str, Any] = {
        "name": stage.name,
        "actionTypeId": {"category": "Build", "owner": "AWS", "provider": "CodeBuild", "version": "1"},
        "configuration": {"ProjectName": stage.project_name},
        "inputArtifacts": [{"name": stage.input_artifact_name}],
    }
    if stage.output_artifact_name:
        action["outputArtifacts"] = [{"name": stage.output_artifact_name}]
    return {"name": stage.name, "actions": [action]}


def _pipeline_definition(spec: CodePipelineSpec) -> dict[str, Any]:
    return {
        "name": spec.name,
        "roleArn": spec.service_role_arn,
        "artifactStore": {
            "type": "S3",
            "location": spec.artifact_store.bucket_name,
            "encryptionKey": {"id": spec.artifact_store.kms_key_arn, "type": "KMS"},
        },
        "stages": [_source_stage(spec)] + [_build_stage(stage) for stage in spec.build_stages],
    }


def _get_pipeline(client: Any, name: str) -> dict[str, Any] | None:
    try:
        return client.get_pipeline(name=name)["pipeline"]
    except ClientError as error:
        if error.response.get("Error", {}).get("Code") == "PipelineNotFoundException":
            return None
        raise


def pipeline_exists(client: Any, name: str) -> bool:
    return _get_pipeline(client, name) is not None


def ensure_pipeline(client: Any, spec: CodePipelineSpec, tags: dict[str, str]) -> str:
    """Create or update the pipeline. Returns the pipeline ARN."""

    definition = _pipeline_definition(spec)
    existing = _get_pipeline(client, spec.name)
    if existing is None:
        created = client.create_pipeline(
            pipeline=definition,
            tags=[{"key": key, "value": value} for key, value in tags.items()],
        )["pipeline"]
    else:
        created = client.update_pipeline(pipeline=definition)["pipeline"]

    account_id = spec.service_role_arn.split(":")[4]
    region = client.meta.region_name
    return f"arn:aws:codepipeline:{region}:{account_id}:{created['name']}"


def delete_pipeline(client: Any, name: str) -> None:
    client.delete_pipeline(name=name)


def start_pipeline_execution(client: Any, name: str) -> str:
    return client.start_pipeline_execution(name=name)["pipelineExecutionId"]


def latest_execution_status(client: Any, name: str) -> str | None:
    summaries = client.list_pipeline_executions(pipelineName=name, maxResults=1)[
        "pipelineExecutionSummaries"
    ]
    return summaries[0]["status"] if summaries else None
