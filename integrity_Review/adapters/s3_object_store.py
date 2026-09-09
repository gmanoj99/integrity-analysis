"""boto3-backed S3 adapter: implements both ``ObjectStore`` and ``MediaUriProvider``.

One instance is bound to a single bucket. The worker creates one per bucket role
(media, staged requests, results) sharing a single boto3 client.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

import boto3
from botocore.exceptions import ClientError

if TYPE_CHECKING:
    from ..contracts.evidence import EvidenceChunkRef

DEFAULT_PRESIGN_EXPIRES_IN_SECONDS = 3_600

_MISSING_OBJECT_CODES = {"404", "NoSuchKey", "NoSuchBucket"}


class ObjectNotFoundError(FileNotFoundError):
    """Raised when a key does not exist, so callers can treat it as terminal.

    A missing object never becomes present by retrying the same message, which
    is what separates it from a throttled or transient S3 error.
    """


class S3ObjectStore:
    def __init__(
        self,
        bucket: str,
        *,
        region_name: str | None = None,
        client: Any | None = None,
        presign_expires_in_seconds: int = DEFAULT_PRESIGN_EXPIRES_IN_SECONDS,
    ) -> None:
        self._bucket = bucket
        self._client = client or boto3.client("s3", region_name=region_name)
        self._presign_expires_in_seconds = presign_expires_in_seconds

    async def get_bytes(self, ref: str) -> bytes:
        return await asyncio.to_thread(self._get_bytes_sync, ref)

    def _get_bytes_sync(self, ref: str) -> bytes:
        try:
            response = self._client.get_object(Bucket=self._bucket, Key=ref)
        except ClientError as error:
            if error.response.get("Error", {}).get("Code") in _MISSING_OBJECT_CODES:
                raise ObjectNotFoundError(f"s3://{self._bucket}/{ref} not found") from error
            raise
        return response["Body"].read()

    async def get_json(self, ref: str) -> Any:
        return json.loads((await self.get_bytes(ref)).decode("utf-8"))

    async def head_object(self, ref: str) -> bool:
        """Return ``True`` if ``ref`` already exists in this bucket."""

        return await asyncio.to_thread(self._head_object_sync, ref)

    def _head_object_sync(self, ref: str) -> bool:
        try:
            self._client.head_object(Bucket=self._bucket, Key=ref)
            return True
        except ClientError as error:
            if error.response.get("Error", {}).get("Code") in _MISSING_OBJECT_CODES:
                return False
            raise

    async def put_object(
        self, ref: str, body: bytes, *, content_type: str = "application/json"
    ) -> None:
        await asyncio.to_thread(self._put_object_sync, ref, body, content_type)

    def _put_object_sync(self, ref: str, body: bytes, content_type: str) -> None:
        self._client.put_object(
            Bucket=self._bucket, Key=ref, Body=body, ContentType=content_type
        )

    def generate_presigned_url(
        self, ref: str, *, expires_in: int | None = None
    ) -> str:
        return self._client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self._bucket, "Key": ref},
            ExpiresIn=expires_in or self._presign_expires_in_seconds,
        )

    async def media_uri_for(self, chunk: EvidenceChunkRef) -> str:
        return self.generate_presigned_url(chunk.source_ref)
