"""Read/attach helpers against an existing Lambda function this tool does not provision."""

from __future__ import annotations

from typing import Any


def get_function_execution_role_arn(client: Any, function_name: str) -> str:
    """Return the execution role ARN of an already-deployed Lambda function."""

    return client.get_function_configuration(FunctionName=function_name)["Role"]


def ensure_event_source_mapping(
    client: Any, *, function_name: str, event_source_arn: str, batch_size: int
) -> str:
    """Create the queue trigger on the function if missing. Returns the mapping UUID."""

    existing = client.list_event_source_mappings(
        FunctionName=function_name, EventSourceArn=event_source_arn
    )["EventSourceMappings"]
    if existing:
        return existing[0]["UUID"]
    created = client.create_event_source_mapping(
        FunctionName=function_name, EventSourceArn=event_source_arn, BatchSize=batch_size
    )
    return created["UUID"]
