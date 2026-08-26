"""VPC networking: VPC, subnets, gateways, route tables, and security groups.

Fargate tasks attach to ENIs inside a VPC, so this covers the networking a
Terraform ``network.tf`` would otherwise own. Every ``ensure_*`` function
looks the resource up by its ``Name`` tag before creating anything.
"""

from __future__ import annotations

from typing import Any

from ..specs import (
    InternetGatewaySpec,
    NatGatewaySpec,
    RouteTableSpec,
    S3GatewayEndpointSpec,
    SecurityGroupSpec,
    SubnetSpec,
    VpcSpec,
)


def _tag_specifications(resource_type: str, name: str, tags: dict[str, str]) -> list[dict[str, Any]]:
    return [
        {
            "ResourceType": resource_type,
            "Tags": [{"Key": "Name", "Value": name}] + [
                {"Key": key, "Value": value} for key, value in tags.items()
            ],
        }
    ]


def _find_by_name_tag(client: Any, describe_method: str, list_key: str, name: str) -> dict[str, Any] | None:
    response = getattr(client, describe_method)(
        Filters=[{"Name": "tag:Name", "Values": [name]}]
    )
    resources = response.get(list_key, [])
    return resources[0] if resources else None


def ensure_vpc(client: Any, spec: VpcSpec, tags: dict[str, str]) -> str:
    existing = _find_by_name_tag(client, "describe_vpcs", "Vpcs", spec.name)
    if existing:
        return existing["VpcId"]
    created = client.create_vpc(
        CidrBlock=spec.cidr_block,
        TagSpecifications=_tag_specifications("vpc", spec.name, tags),
    )["Vpc"]
    vpc_id = created["VpcId"]
    client.modify_vpc_attribute(VpcId=vpc_id, EnableDnsSupport={"Value": True})
    client.modify_vpc_attribute(VpcId=vpc_id, EnableDnsHostnames={"Value": True})
    client.get_waiter("vpc_available").wait(VpcIds=[vpc_id])
    return vpc_id


def ensure_subnet(client: Any, spec: SubnetSpec, tags: dict[str, str]) -> str:
    existing = _find_by_name_tag(client, "describe_subnets", "Subnets", spec.name)
    if existing:
        return existing["SubnetId"]
    created = client.create_subnet(
        VpcId=spec.vpc_id,
        CidrBlock=spec.cidr_block,
        AvailabilityZone=spec.availability_zone,
        TagSpecifications=_tag_specifications("subnet", spec.name, tags),
    )["Subnet"]
    subnet_id = created["SubnetId"]
    if spec.public:
        client.modify_subnet_attribute(SubnetId=subnet_id, MapPublicIpOnLaunch={"Value": True})
    return subnet_id


def ensure_internet_gateway(client: Any, spec: InternetGatewaySpec, tags: dict[str, str]) -> str:
    existing = _find_by_name_tag(client, "describe_internet_gateways", "InternetGateways", spec.name)
    if existing:
        return existing["InternetGatewayId"]
    igw_id = client.create_internet_gateway(
        TagSpecifications=_tag_specifications("internet-gateway", spec.name, tags),
    )["InternetGateway"]["InternetGatewayId"]
    client.attach_internet_gateway(InternetGatewayId=igw_id, VpcId=spec.vpc_id)
    return igw_id


def ensure_nat_gateway(client: Any, spec: NatGatewaySpec, tags: dict[str, str]) -> str:
    existing = _find_by_name_tag(client, "describe_nat_gateways", "NatGateways", spec.name)
    if existing and existing["State"] not in {"deleted", "deleting", "failed"}:
        return existing["NatGatewayId"]
    allocation_id = client.allocate_address(
        Domain="vpc", TagSpecifications=_tag_specifications("elastic-ip", spec.name, tags)
    )["AllocationId"]
    nat_gateway_id = client.create_nat_gateway(
        SubnetId=spec.public_subnet_id,
        AllocationId=allocation_id,
        TagSpecifications=_tag_specifications("natgateway", spec.name, tags),
    )["NatGateway"]["NatGatewayId"]
    client.get_waiter("nat_gateway_available").wait(NatGatewayIds=[nat_gateway_id])
    return nat_gateway_id


def ensure_route_table(client: Any, spec: RouteTableSpec, tags: dict[str, str]) -> str:
    existing = _find_by_name_tag(client, "describe_route_tables", "RouteTables", spec.name)
    if existing:
        route_table_id = existing["RouteTableId"]
    else:
        route_table_id = client.create_route_table(
            VpcId=spec.vpc_id,
            TagSpecifications=_tag_specifications("route-table", spec.name, tags),
        )["RouteTable"]["RouteTableId"]

    route_kwargs: dict[str, Any] = {"DestinationCidrBlock": "0.0.0.0/0"}
    if spec.internet_gateway_id:
        route_kwargs["GatewayId"] = spec.internet_gateway_id
    elif spec.nat_gateway_id:
        route_kwargs["NatGatewayId"] = spec.nat_gateway_id
    if spec.internet_gateway_id or spec.nat_gateway_id:
        try:
            client.create_route(RouteTableId=route_table_id, **route_kwargs)
        except client.exceptions.ClientError as error:
            if "RouteAlreadyExists" not in str(error):
                raise

    associated_subnet_ids = {
        association["SubnetId"]
        for association in client.describe_route_tables(RouteTableIds=[route_table_id])[
            "RouteTables"
        ][0].get("Associations", [])
        if association.get("SubnetId")
    }
    for subnet_id in spec.subnet_ids:
        if subnet_id not in associated_subnet_ids:
            client.associate_route_table(RouteTableId=route_table_id, SubnetId=subnet_id)
    return route_table_id


def ensure_worker_security_group(client: Any, spec: SecurityGroupSpec, tags: dict[str, str]) -> str:
    """Security group with no inbound rules and HTTPS-only egress."""

    existing = _find_by_name_tag(client, "describe_security_groups", "SecurityGroups", spec.name)
    if existing:
        security_group_id = existing["GroupId"]
    else:
        security_group_id = client.create_security_group(
            GroupName=spec.name,
            Description=spec.description,
            VpcId=spec.vpc_id,
            TagSpecifications=_tag_specifications("security-group", spec.name, tags),
        )["GroupId"]
        # A newly created SG has an implicit allow-all egress rule; revoke it
        # so only the explicit HTTPS rule below applies.
        client.revoke_security_group_egress(
            GroupId=security_group_id,
            IpPermissions=[
                {"IpProtocol": "-1", "IpRanges": [{"CidrIp": "0.0.0.0/0"}]}
            ],
        )

    try:
        client.authorize_security_group_egress(
            GroupId=security_group_id,
            IpPermissions=[
                {
                    "IpProtocol": "tcp",
                    "FromPort": 443,
                    "ToPort": 443,
                    "IpRanges": [{"CidrIp": "0.0.0.0/0", "Description": "HTTPS egress only"}],
                }
            ],
        )
    except client.exceptions.ClientError as error:
        if "InvalidPermission.Duplicate" not in str(error):
            raise
    return security_group_id


def ensure_s3_gateway_endpoint(client: Any, spec: S3GatewayEndpointSpec, tags: dict[str, str]) -> str:
    service_name = f"com.amazonaws.{spec.region}.s3"
    existing = client.describe_vpc_endpoints(
        Filters=[
            {"Name": "vpc-id", "Values": [spec.vpc_id]},
            {"Name": "service-name", "Values": [service_name]},
        ]
    )["VpcEndpoints"]
    if existing:
        return existing[0]["VpcEndpointId"]
    return client.create_vpc_endpoint(
        VpcId=spec.vpc_id,
        ServiceName=service_name,
        VpcEndpointType="Gateway",
        RouteTableIds=list(spec.route_table_ids),
        TagSpecifications=_tag_specifications("vpc-endpoint", f"{spec.vpc_id}-s3", tags),
    )["VpcEndpoint"]["VpcEndpointId"]
