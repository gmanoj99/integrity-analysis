"""botocore Stubber coverage for the Application Auto Scaling step-scaling helpers."""

from __future__ import annotations

from infra.aws.specs import ScalableTargetSpec, StepAdjustment, StepScalingPolicySpec
from infra.aws.utils import autoscaling_utils

from .aws_stub import stubbed_client

RESOURCE_ID = "service/integrity-review-beta-cluster/integrity-review-beta-worker-service"
SCALABLE_DIMENSION = "ecs:service:DesiredCount"
SERVICE_NAMESPACE = "ecs"
POLICY_ARN = (
    "arn:aws:autoscaling:ap-south-1:111111111111:scalingPolicy:abc123:resource/ecs/"
    f"{RESOURCE_ID}:policyName/integrity-review-beta-scale-out-on-backlog"
)


def test_register_scalable_target_sends_min_and_max_capacity() -> None:
    client, stubber = stubbed_client("application-autoscaling")
    spec = ScalableTargetSpec(
        service_namespace=SERVICE_NAMESPACE,
        resource_id=RESOURCE_ID,
        scalable_dimension=SCALABLE_DIMENSION,
        min_capacity=0,
        max_capacity=2,
    )

    stubber.add_response(
        "register_scalable_target",
        {},
        {
            "ServiceNamespace": SERVICE_NAMESPACE,
            "ResourceId": RESOURCE_ID,
            "ScalableDimension": SCALABLE_DIMENSION,
            "MinCapacity": 0,
            "MaxCapacity": 2,
        },
    )

    with stubber:
        autoscaling_utils.register_scalable_target(client, spec)

    stubber.assert_no_pending_responses()


def test_deregister_scalable_target_scopes_to_resource_and_dimension() -> None:
    client, stubber = stubbed_client("application-autoscaling")

    stubber.add_response(
        "deregister_scalable_target",
        {},
        {
            "ServiceNamespace": SERVICE_NAMESPACE,
            "ResourceId": RESOURCE_ID,
            "ScalableDimension": SCALABLE_DIMENSION,
        },
    )

    with stubber:
        autoscaling_utils.deregister_scalable_target(
            client,
            service_namespace=SERVICE_NAMESPACE,
            resource_id=RESOURCE_ID,
            scalable_dimension=SCALABLE_DIMENSION,
        )

    stubber.assert_no_pending_responses()


def test_put_step_scaling_policy_builds_step_adjustments_and_returns_arn() -> None:
    client, stubber = stubbed_client("application-autoscaling")
    spec = StepScalingPolicySpec(
        name="integrity-review-beta-scale-out-on-backlog",
        service_namespace=SERVICE_NAMESPACE,
        resource_id=RESOURCE_ID,
        scalable_dimension=SCALABLE_DIMENSION,
        adjustment_type="ExactCapacity",
        step_adjustments=(
            StepAdjustment(scaling_adjustment=1, metric_interval_lower_bound=0.0, metric_interval_upper_bound=2.0),
            StepAdjustment(scaling_adjustment=2, metric_interval_lower_bound=2.0),
        ),
    )

    stubber.add_response(
        "put_scaling_policy",
        {"PolicyARN": POLICY_ARN, "Alarms": []},
        {
            "PolicyName": spec.name,
            "ServiceNamespace": SERVICE_NAMESPACE,
            "ResourceId": RESOURCE_ID,
            "ScalableDimension": SCALABLE_DIMENSION,
            "PolicyType": "StepScaling",
            "StepScalingPolicyConfiguration": {
                "AdjustmentType": "ExactCapacity",
                "MetricAggregationType": "Maximum",
                "Cooldown": 60,
                "StepAdjustments": [
                    {
                        "ScalingAdjustment": 1,
                        "MetricIntervalLowerBound": 0.0,
                        "MetricIntervalUpperBound": 2.0,
                    },
                    {"ScalingAdjustment": 2, "MetricIntervalLowerBound": 2.0},
                ],
            },
        },
    )

    with stubber:
        policy_arn = autoscaling_utils.put_step_scaling_policy(client, spec)

    assert policy_arn == POLICY_ARN
    stubber.assert_no_pending_responses()


def test_put_step_scaling_policy_omits_bounds_that_are_none() -> None:
    client, stubber = stubbed_client("application-autoscaling")
    spec = StepScalingPolicySpec(
        name="integrity-review-beta-scale-in-on-idle",
        service_namespace=SERVICE_NAMESPACE,
        resource_id=RESOURCE_ID,
        scalable_dimension=SCALABLE_DIMENSION,
        adjustment_type="ExactCapacity",
        step_adjustments=(StepAdjustment(scaling_adjustment=0, metric_interval_upper_bound=0.0),),
    )

    stubber.add_response(
        "put_scaling_policy",
        {"PolicyARN": POLICY_ARN, "Alarms": []},
        {
            "PolicyName": spec.name,
            "ServiceNamespace": SERVICE_NAMESPACE,
            "ResourceId": RESOURCE_ID,
            "ScalableDimension": SCALABLE_DIMENSION,
            "PolicyType": "StepScaling",
            "StepScalingPolicyConfiguration": {
                "AdjustmentType": "ExactCapacity",
                "MetricAggregationType": "Maximum",
                "Cooldown": 60,
                "StepAdjustments": [{"ScalingAdjustment": 0, "MetricIntervalUpperBound": 0.0}],
            },
        },
    )

    with stubber:
        autoscaling_utils.put_step_scaling_policy(client, spec)

    stubber.assert_no_pending_responses()


def test_delete_scaling_policy_scopes_to_name_and_resource() -> None:
    client, stubber = stubbed_client("application-autoscaling")

    stubber.add_response(
        "delete_scaling_policy",
        {},
        {
            "PolicyName": "integrity-review-beta-scale-out-on-backlog",
            "ServiceNamespace": SERVICE_NAMESPACE,
            "ResourceId": RESOURCE_ID,
            "ScalableDimension": SCALABLE_DIMENSION,
        },
    )

    with stubber:
        autoscaling_utils.delete_scaling_policy(
            client,
            name="integrity-review-beta-scale-out-on-backlog",
            service_namespace=SERVICE_NAMESPACE,
            resource_id=RESOURCE_ID,
            scalable_dimension=SCALABLE_DIMENSION,
        )

    stubber.assert_no_pending_responses()
