"""CloudWatch alarms covering ECS, queue, and DLQ health."""

from __future__ import annotations

from typing import Any

from ..specs import AlarmSpec


def ensure_alarm(client: Any, spec: AlarmSpec) -> None:
    """Create or update the alarm.

    Deliberately untagged: ``PutMetricAlarm`` only requires
    ``cloudwatch:TagResource`` in the caller's IAM policy when a non-empty
    ``Tags`` list is supplied, and the state manifest (not AWS-side tags)
    tracks resource ownership for ``status``/``destroy``.
    """

    client.put_metric_alarm(
        AlarmName=spec.name,
        Namespace=spec.namespace,
        MetricName=spec.metric_name,
        Dimensions=[{"Name": key, "Value": value} for key, value in spec.dimensions.items()],
        ComparisonOperator=spec.comparison_operator,
        Threshold=spec.threshold,
        EvaluationPeriods=spec.evaluation_periods,
        Period=spec.period_seconds,
        Statistic=spec.statistic,
        AlarmActions=list(spec.alarm_actions),
        TreatMissingData=spec.treat_missing_data,
    )


def alarm_state(client: Any, alarm_name: str) -> str | None:
    alarms = client.describe_alarms(AlarmNames=[alarm_name])["MetricAlarms"]
    return alarms[0]["StateValue"] if alarms else None
