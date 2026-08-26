"""botocore Stubber coverage for the EventBridge pipeline-trigger rule."""

from __future__ import annotations

from infra.aws.specs import EventBridgeRuleSpec
from infra.aws.utils import events_utils

from .aws_stub import stubbed_client

RULE_NAME = "integrity-review-beta-pipeline-trigger"
RULE_ARN = "arn:aws:events:ap-south-1:111111111111:rule/integrity-review-beta-pipeline-trigger"
TARGET_ID = f"{RULE_NAME}-target"


def _spec() -> EventBridgeRuleSpec:
    return EventBridgeRuleSpec(
        name=RULE_NAME,
        description="Starts integrity-review-beta-pipeline on push to main",
        event_pattern={"source": ["aws.codecommit"]},
        target_arn="arn:aws:codepipeline:ap-south-1:111111111111:integrity-review-beta-pipeline",
        target_role_arn="arn:aws:iam::111111111111:role/integrity-review-beta-pipeline-trigger",
        target_id=TARGET_ID,
    )


def test_ensure_rule_puts_rule_then_target() -> None:
    client, stubber = stubbed_client("events")
    spec = _spec()
    stubber.add_response("put_rule", {"RuleArn": RULE_ARN})
    stubber.add_response(
        "put_targets",
        {"FailedEntryCount": 0, "FailedEntries": []},
        {
            "Rule": RULE_NAME,
            "Targets": [
                {"Id": TARGET_ID, "Arn": spec.target_arn, "RoleArn": spec.target_role_arn}
            ],
        },
    )

    with stubber:
        arn = events_utils.ensure_rule(client, spec, {"Project": "integrity-review"})

    assert arn == RULE_ARN
    stubber.assert_no_pending_responses()


def test_delete_rule_removes_targets_before_deleting() -> None:
    client, stubber = stubbed_client("events")
    stubber.add_response("remove_targets", {"FailedEntryCount": 0, "FailedEntries": []})
    stubber.add_response("delete_rule", {})

    with stubber:
        events_utils.delete_rule(client, RULE_NAME, target_id=TARGET_ID)

    stubber.assert_no_pending_responses()
