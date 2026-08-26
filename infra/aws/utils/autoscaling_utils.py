"""Application Auto Scaling: scalable target and step scaling policies for the ECS worker service.

Step scaling (not target tracking) is used because target tracking's backlog-
per-task math divides by the running task count, so it cannot scale from or to
zero tasks. Step scaling driven by a raw SQS-depth alarm can.
"""

from __future__ import annotations

from typing import Any

from ..specs import ScalableTargetSpec, StepScalingPolicySpec


def register_scalable_target(client: Any, spec: ScalableTargetSpec) -> None:
    """Register (or update) the scalable target.

    Deliberately untagged: ``RegisterScalableTarget`` only requires
    ``application-autoscaling:TagResource`` in the caller's IAM policy when
    a non-empty ``Tags`` map is supplied, and the state manifest (not
    AWS-side tags) tracks resource ownership for ``status``/``destroy``.
    """

    client.register_scalable_target(
        ServiceNamespace=spec.service_namespace,
        ResourceId=spec.resource_id,
        ScalableDimension=spec.scalable_dimension,
        MinCapacity=spec.min_capacity,
        MaxCapacity=spec.max_capacity,
    )


def deregister_scalable_target(
    client: Any, *, service_namespace: str, resource_id: str, scalable_dimension: str
) -> None:
    client.deregister_scalable_target(
        ServiceNamespace=service_namespace,
        ResourceId=resource_id,
        ScalableDimension=scalable_dimension,
    )


def _step_adjustment_kwargs(step: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"ScalingAdjustment": step.scaling_adjustment}
    if step.metric_interval_lower_bound is not None:
        kwargs["MetricIntervalLowerBound"] = step.metric_interval_lower_bound
    if step.metric_interval_upper_bound is not None:
        kwargs["MetricIntervalUpperBound"] = step.metric_interval_upper_bound
    return kwargs


def put_step_scaling_policy(client: Any, spec: StepScalingPolicySpec) -> str:
    """Create or update a step scaling policy. Returns the policy ARN."""

    response = client.put_scaling_policy(
        PolicyName=spec.name,
        ServiceNamespace=spec.service_namespace,
        ResourceId=spec.resource_id,
        ScalableDimension=spec.scalable_dimension,
        PolicyType="StepScaling",
        StepScalingPolicyConfiguration={
            "AdjustmentType": spec.adjustment_type,
            "MetricAggregationType": spec.metric_aggregation_type,
            "Cooldown": spec.cooldown_seconds,
            "StepAdjustments": [_step_adjustment_kwargs(step) for step in spec.step_adjustments],
        },
    )
    return response["PolicyARN"]


def delete_scaling_policy(
    client: Any, *, name: str, service_namespace: str, resource_id: str, scalable_dimension: str
) -> None:
    client.delete_scaling_policy(
        PolicyName=name,
        ServiceNamespace=service_namespace,
        ResourceId=resource_id,
        ScalableDimension=scalable_dimension,
    )
