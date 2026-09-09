"""S3-backed implementation of SebLogObjectStore for streaming access to log files."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

try:
    import boto3
    from botocore.config import Config
except ImportError:
    boto3 = None
    Config = None


class S3ObjectStore:
    """S3 object store with streaming line access for large log files.

    Implements the SebLogObjectStore protocol for accessing SEB log files
    stored in S3 without loading entire files into memory.
    """

    def __init__(
        self,
        bucket: str,
        region_name: str = "ap-south-1",
        chunk_size: int = 8192,
    ) -> None:
        """Initialize S3 object store.

        Args:
            bucket: S3 bucket name
            region_name: AWS region for the bucket
            chunk_size: Size of chunks to read when streaming (default 8KB)
        """
        if boto3 is None:
            raise RuntimeError(
                "boto3 is required for S3 access. Install with: pip install boto3"
            )

        self._bucket = bucket
        self._chunk_size = chunk_size
        self._client = boto3.client(
            "s3",
            region_name=region_name,
            config=Config(
                max_pool_connections=20,
                retries={"max_attempts": 3, "mode": "adaptive"},
            ),
        )

    def list_keys(self, prefix: str) -> Iterator[str]:
        """List all object keys under the given prefix.

        Uses pagination to handle buckets with many objects.
        """
        paginator = self._client.get_paginator("list_objects_v2")

        for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                yield obj["Key"]

    def iter_lines(self, key: str) -> Iterator[str]:
        """Stream lines from an S3 object without loading it entirely into memory.

        Reads the object in chunks and yields complete lines. Handles UTF-8
        decoding and proper line splitting.
        """
        response = self._client.get_object(Bucket=self._bucket, Key=key)
        body = response["Body"]

        buffer = ""
        try:
            while True:
                chunk = body.read(self._chunk_size)
                if not chunk:
                    break

                try:
                    text = chunk.decode("utf-8")
                except UnicodeDecodeError:
                    text = chunk.decode("utf-8", errors="replace")

                buffer += text

                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    yield line

            if buffer:
                yield buffer
        finally:
            body.close()

    def get_json(self, key: str) -> dict[str, Any]:
        """Load and parse a JSON object from S3."""
        response = self._client.get_object(Bucket=self._bucket, Key=key)
        body = response["Body"]
        try:
            content = body.read()
            return json.loads(content.decode("utf-8"))
        finally:
            body.close()

    def get_object_size(self, key: str) -> int:
        """Get the size of an object in bytes."""
        response = self._client.head_object(Bucket=self._bucket, Key=key)
        return response["ContentLength"]


class LocalFileStore:
    """Local filesystem implementation of SebLogObjectStore for testing.

    Useful for running the pipeline against local log files during development.
    """

    def __init__(self, base_path: str) -> None:
        """Initialize with a base path for local files."""
        self._base_path = base_path.rstrip("/")

    def list_keys(self, prefix: str) -> Iterator[str]:
        """List all files under the prefix directory."""
        import os

        full_path = os.path.join(self._base_path, prefix.lstrip("/"))

        if not os.path.exists(full_path):
            return

        for root, _, files in os.walk(full_path):
            for filename in files:
                file_path = os.path.join(root, filename)
                relative_path = os.path.relpath(file_path, self._base_path)
                yield relative_path

    def iter_lines(self, key: str) -> Iterator[str]:
        """Read lines from a local file."""
        import os

        full_path = os.path.join(self._base_path, key.lstrip("/"))

        with open(full_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                yield line.rstrip("\n\r")

    def get_json(self, key: str) -> dict[str, Any]:
        """Load and parse a local JSON file."""
        import os

        full_path = os.path.join(self._base_path, key.lstrip("/"))

        with open(full_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_object_size(self, key: str) -> int:
        """Get the size of a local file in bytes."""
        import os

        full_path = os.path.join(self._base_path, key.lstrip("/"))
        return os.path.getsize(full_path)
