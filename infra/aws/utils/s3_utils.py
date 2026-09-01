"""Read-only helpers for the externally owned media bucket."""

from __future__ import annotations

from typing import Any

from botocore.exceptions import ClientError


def bucket_exists(client: Any, bucket_name: str) -> bool:
    try:
        client.head_bucket(Bucket=bucket_name)
        return True
    except ClientError as error:
        if error.response.get("Error", {}).get("Code") in {"404", "NoSuchBucket"}:
            return False
        raise
