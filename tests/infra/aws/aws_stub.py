"""Shared botocore Stubber client factory for infra.aws util tests."""

from __future__ import annotations

import boto3
from botocore.stub import Stubber

REGION = "ap-south-1"


def stubbed_client(service_name: str) -> tuple[boto3.client, Stubber]:
    client = boto3.client(
        service_name,
        region_name=REGION,
        aws_access_key_id="test",
        aws_secret_access_key="test",
    )
    stubber = Stubber(client)
    return client, stubber
