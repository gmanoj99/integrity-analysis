"""ECS cluster, task definition, and Fargate service provisioning/inspection."""

from __future__ import annotations

from typing import Any

from ..specs import ClusterSpec, ServiceSpec, TaskDefinitionSpec


def ensure_cluster(client: Any, spec: ClusterSpec, tags: dict[str, str]) -> str:
    described = client.describe_clusters(clusters=[spec.name])["clusters"]
    active = [cluster for cluster in described if cluster["status"] == "ACTIVE"]
    if active:
        return active[0]["clusterArn"]
    return client.create_cluster(
        clusterName=spec.name, tags=[{"key": key, "value": value} for key, value in tags.items()]
    )["cluster"]["clusterArn"]


def _container_definition(spec: TaskDefinitionSpec) -> dict[str, Any]:
    container = spec.container
    return {
        "name": container.name,
        "image": container.image,
        "essential": True,
        "environment": [
            {"name": key, "value": value} for key, value in sorted(container.environment.items())
        ],
        "secrets": [
            {"name": key, "valueFrom": value} for key, value in sorted(container.secrets.items())
        ],
        "stopTimeout": spec.stop_timeout_seconds,
        "readonlyRootFilesystem": True,
        "logConfiguration": {
            "logDriver": "awslogs",
            "options": {
                "awslogs-group": container.log_group,
                "awslogs-region": container.region,
                "awslogs-stream-prefix": "worker",
            },
        },
    }


def register_task_definition(client: Any, spec: TaskDefinitionSpec, tags: dict[str, str]) -> str:
    response = client.register_task_definition(
        family=spec.family,
        requiresCompatibilities=["FARGATE"],
        networkMode="awsvpc",
        cpu=spec.cpu,
        memory=spec.memory,
        ephemeralStorage={"sizeInGiB": spec.ephemeral_storage_gib},
        executionRoleArn=spec.execution_role_arn,
        taskRoleArn=spec.task_role_arn,
        runtimePlatform={"cpuArchitecture": "X86_64", "operatingSystemFamily": "LINUX"},
        containerDefinitions=[_container_definition(spec)],
        tags=[{"key": key, "value": value} for key, value in tags.items()],
    )
    return response["taskDefinition"]["taskDefinitionArn"]


def _describe_service(client: Any, cluster: str, name: str) -> dict[str, Any] | None:
    described = client.describe_services(cluster=cluster, services=[name])["services"]
    active = [service for service in described if service["status"] != "INACTIVE"]
    return active[0] if active else None


def ensure_service(client: Any, spec: ServiceSpec, tags: dict[str, str]) -> str:
    """Create the service (desired count as given) or update it to the new task definition.

    ``desiredCount`` is intentionally omitted from the update call: once
    Application Auto Scaling owns this service's ``DesiredCount`` (see
    ``orchestrator._ensure_autoscaling``), re-running ``apply`` must not reset
    it back to the static config value and fight the scaler.
    """

    existing = _describe_service(client, spec.cluster, spec.name)
    network_configuration = {
        "awsvpcConfiguration": {
            "subnets": list(spec.subnet_ids),
            "securityGroups": list(spec.security_group_ids),
            "assignPublicIp": "DISABLED",
        }
    }
    if existing is None:
        return client.create_service(
            cluster=spec.cluster,
            serviceName=spec.name,
            taskDefinition=spec.task_definition_arn,
            desiredCount=spec.desired_count,
            launchType="FARGATE",
            networkConfiguration=network_configuration,
            deploymentConfiguration={
                "deploymentCircuitBreaker": {"enable": True, "rollback": True},
                "maximumPercent": 200,
                "minimumHealthyPercent": 50,
            },
            enableExecuteCommand=False,
            tags=[{"key": key, "value": value} for key, value in tags.items()],
        )["service"]["serviceArn"]

    return client.update_service(
        cluster=spec.cluster,
        service=spec.name,
        taskDefinition=spec.task_definition_arn,
        networkConfiguration=network_configuration,
    )["service"]["serviceArn"]


def update_desired_count(client: Any, *, cluster: str, service: str, desired_count: int) -> None:
    client.update_service(cluster=cluster, service=service, desiredCount=desired_count)


def wait_for_service_stable(client: Any, *, cluster: str, service: str) -> None:
    client.get_waiter("services_stable").wait(cluster=cluster, services=[service])


def list_task_arns(client: Any, *, cluster: str, service: str) -> list[str]:
    return client.list_tasks(cluster=cluster, serviceName=service)["taskArns"]


def describe_tasks(client: Any, *, cluster: str, task_arns: list[str]) -> list[dict[str, Any]]:
    if not task_arns:
        return []
    return client.describe_tasks(cluster=cluster, tasks=task_arns)["tasks"]


def get_task_protection(client: Any, *, cluster: str, task_arns: list[str]) -> list[dict[str, Any]]:
    if not task_arns:
        return []
    return client.get_task_protection(cluster=cluster, tasks=task_arns)["protectedTasks"]
