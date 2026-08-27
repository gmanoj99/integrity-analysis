"""Rollback ordering and best-effort continuation on partial delete failures.

These exercise ``orchestrator._rollback`` directly with fake boto3 clients so
we can assert *deletion order* (reverse of creation order) without touching
AWS: alarms/compute/roles must be torn down before the data plane.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from infra.aws import orchestrator
from infra.aws.config import EnvironmentConfig, NetworkConfig, RetentionConfig, SizingConfig
from infra.aws.state_store import DeploymentManifest, ResourceRecord


class FakePaginator:
    def __init__(self, pages: list[dict[str, Any]]) -> None:
        self._pages = pages

    def paginate(self, **_kwargs: Any) -> list[dict[str, Any]]:
        return self._pages


class FakeClient:
    def __init__(
        self,
        service_name: str,
        events: list[tuple[str, str]],
        *,
        responses: dict[str, Any] | None = None,
        paginators: dict[str, list[dict[str, Any]]] | None = None,
        raises: dict[str, Exception] | None = None,
    ) -> None:
        self._service_name = service_name
        self._events = events
        self._responses = responses or {}
        self._paginators = paginators or {}
        self._raises = raises or {}

    def get_paginator(self, operation_name: str) -> FakePaginator:
        return FakePaginator(self._paginators.get(operation_name, []))

    def __getattr__(self, name: str) -> Any:
        def _call(**_kwargs: Any) -> Any:
            self._events.append((self._service_name, name))
            if name in self._raises:
                raise self._raises[name]
            return self._responses.get(name, {})

        return _call


@dataclass
class FakeOrchestratorContext:
    config: EnvironmentConfig
    clients: dict[str, FakeClient]

    def client(self, service_name: str) -> FakeClient:
        return self.clients[service_name]


def _config() -> EnvironmentConfig:
    return EnvironmentConfig(
        account_id="111111111111",
        region="ap-south-1",
        environment="beta",
        project="integrity-review",
        owner="platform-team",
        cost_center="eng-test",
        stage="beta",
        organization_id="local",
        storage_bucket_name="nxtwave-assessments-backend-nxtwave-media-static",
        storage_kms_key_arn=None,
        network=NetworkConfig(
            vpc_cidr_block="10.90.0.0/16",
            public_subnet_cidrs=(),
            private_subnet_cidrs=(),
            availability_zones=(),
        ),
        sizing=SizingConfig(
            task_cpu="2048", task_memory="4096", ephemeral_storage_gib=20,
            desired_count=0, max_concurrent_reviews=4, gemini_task_limit=24,
            gemini_per_review_limit=12,
            min_task_count=0, max_task_count=2, scale_in_idle_periods=5,
        ),
        retention=RetentionConfig(
            log_retention_days=14, bucket_expiration_days=7,
            queue_message_retention_seconds=345600, ecr_max_tagged_images=10,
        ),
        tags={},
    )


def _record(resource_type: str, identifier: str) -> ResourceRecord:
    return ResourceRecord(
        resource_type=resource_type,
        identifier=identifier,
        arn=None,
        config_hash="deadbeef",
        status="created",
        managed_by_tool=True,
    )


def _manifest() -> DeploymentManifest:
    return DeploymentManifest(
        resource_prefix="integrity-review-beta",
        config_hash="deadbeef",
        resources={
            "media_bucket": _record("s3.bucket", "integrity-review-beta-media"),
            "request_queue": _record("sqs.queue", "https://sqs.example/request"),
            "log_group": _record("logs.log_group", "/ecs/integrity-review-beta-worker"),
            "ecr_repository": _record("ecr.repository", "integrity-review-beta-worker"),
            "gemini_secret": _record("secretsmanager.secret", "integrity-review-beta/gemini-api-key"),
            "task_role": _record("iam.role", "integrity-review-beta-ecs-task"),
            "alarm.dlq-depth": _record("cloudwatch.alarm", "integrity-review-beta-request-dlq-not-empty"),
        },
    )


def _index_of(events: list[tuple[str, str]], event: tuple[str, str]) -> int:
    return events.index(event)


def test_rollback_deletes_in_reverse_creation_order() -> None:
    events: list[tuple[str, str]] = []
    ctx = FakeOrchestratorContext(
        config=_config(),
        clients={
            "s3": FakeClient("s3", events, paginators={"list_object_versions": []}),
            "sqs": FakeClient("sqs", events),
            "logs": FakeClient("logs", events),
            "ecr": FakeClient("ecr", events),
            "secretsmanager": FakeClient("secretsmanager", events),
            "iam": FakeClient("iam", events, responses={"list_role_policies": {"PolicyNames": []}}),
            "cloudwatch": FakeClient("cloudwatch", events),
        },
    )
    manifest = _manifest()

    orchestrator._rollback(ctx, manifest, created_this_run=set(manifest.resources))

    cloudwatch_at = _index_of(events, ("cloudwatch", "delete_alarms"))
    iam_at = _index_of(events, ("iam", "delete_role"))
    secret_at = _index_of(events, ("secretsmanager", "delete_secret"))
    ecr_at = _index_of(events, ("ecr", "delete_repository"))
    logs_at = _index_of(events, ("logs", "delete_log_group"))
    sqs_at = _index_of(events, ("sqs", "delete_queue"))
    s3_at = _index_of(events, ("s3", "delete_bucket"))

    assert cloudwatch_at < iam_at < secret_at < ecr_at < logs_at
    assert sqs_at > logs_at
    assert s3_at > logs_at
    assert all(record.status == "destroyed" for record in manifest.resources.values())


def test_rollback_continues_past_a_single_delete_failure() -> None:
    events: list[tuple[str, str]] = []
    ctx = FakeOrchestratorContext(
        config=_config(),
        clients={
            "s3": FakeClient("s3", events, paginators={"list_object_versions": []}),
            "sqs": FakeClient("sqs", events),
            "logs": FakeClient("logs", events),
            "ecr": FakeClient(
                "ecr", events, raises={"delete_repository": RuntimeError("ecr delete denied")}
            ),
            "secretsmanager": FakeClient("secretsmanager", events),
            "iam": FakeClient("iam", events, responses={"list_role_policies": {"PolicyNames": []}}),
            "cloudwatch": FakeClient("cloudwatch", events),
        },
    )
    manifest = _manifest()

    orchestrator._rollback(ctx, manifest, created_this_run=set(manifest.resources))

    assert manifest.resources["ecr_repository"].status == "created"  # left in place after the failure
    assert manifest.resources["log_group"].status == "destroyed"  # later-created resource still cleaned up
    assert manifest.resources["task_role"].status == "destroyed"
