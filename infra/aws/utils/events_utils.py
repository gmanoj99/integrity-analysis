"""EventBridge rule provisioning: triggers the pipeline on a CodeCommit push."""

from __future__ import annotations

import json
from typing import Any

from ..specs import EventBridgeRuleSpec


def ensure_rule(client: Any, spec: EventBridgeRuleSpec, tags: dict[str, str]) -> str:
    """Create/update the rule and point it at the pipeline-start target. Returns the rule ARN."""

    rule_arn = client.put_rule(
        Name=spec.name,
        Description=spec.description,
        EventPattern=json.dumps(spec.event_pattern),
        State="ENABLED",
        Tags=[{"Key": key, "Value": value} for key, value in tags.items()],
    )["RuleArn"]

    client.put_targets(
        Rule=spec.name,
        Targets=[
            {
                "Id": spec.target_id,
                "Arn": spec.target_arn,
                "RoleArn": spec.target_role_arn,
            }
        ],
    )
    return rule_arn


def delete_rule(client: Any, name: str, *, target_id: str) -> None:
    client.remove_targets(Rule=name, Ids=[target_id])
    client.delete_rule(Name=name)
