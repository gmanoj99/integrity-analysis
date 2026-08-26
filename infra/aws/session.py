"""Explicit boto3 session construction with an account-identity guard.

No implicit credential resolution: callers must pass a region and either a
profile name or an IAM role ARN to assume. Every session is validated against
the expected account ID before it is handed to the orchestrator.
"""

from __future__ import annotations

from dataclasses import dataclass

import boto3
from botocore.config import Config

_RETRY_CONFIG = Config(retries={"mode": "adaptive", "max_attempts": 10})


class AccountMismatchError(RuntimeError):
    """Raised when the resolved caller identity does not match the expected account."""


@dataclass(frozen=True, slots=True)
class SessionRequest:
    region: str
    expected_account_id: str
    profile_name: str | None = None
    assume_role_arn: str | None = None
    session_name: str = "integrity-review-provisioner"


def _assume_role_session(request: SessionRequest) -> boto3.Session:
    base_session = boto3.Session(region_name=request.region, profile_name=request.profile_name)
    sts_client = base_session.client("sts", config=_RETRY_CONFIG)
    credentials = sts_client.assume_role(
        RoleArn=request.assume_role_arn,
        RoleSessionName=request.session_name,
    )["Credentials"]
    return boto3.Session(
        region_name=request.region,
        aws_access_key_id=credentials["AccessKeyId"],
        aws_secret_access_key=credentials["SecretAccessKey"],
        aws_session_token=credentials["SessionToken"],
    )


def build_session(request: SessionRequest) -> boto3.Session:
    """Build a boto3 session and verify it resolves to ``expected_account_id``."""

    session = (
        _assume_role_session(request)
        if request.assume_role_arn
        else boto3.Session(region_name=request.region, profile_name=request.profile_name)
    )
    caller_identity = session.client("sts", config=_RETRY_CONFIG).get_caller_identity()
    resolved_account_id = caller_identity["Account"]
    if resolved_account_id != request.expected_account_id:
        raise AccountMismatchError(
            f"session resolved to account {resolved_account_id!r}, "
            f"expected {request.expected_account_id!r}"
        )
    return session


def client(session: boto3.Session, service_name: str) -> "boto3.client":
    """Create a service client with the shared adaptive-retry configuration."""

    return session.client(service_name, config=_RETRY_CONFIG)
