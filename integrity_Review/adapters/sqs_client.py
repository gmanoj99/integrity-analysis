"""boto3-backed SQS adapter, bound to a single queue URL."""

from __future__ import annotations

import asyncio
from typing import Any

import boto3


class SqsClient:
    def __init__(
        self, queue_url: str, *, region_name: str | None = None, client: Any | None = None
    ) -> None:
        self._queue_url = queue_url
        self._client = client or boto3.client("sqs", region_name=region_name)

    async def receive(
        self, *, max_messages: int, wait_time_seconds: int, visibility_timeout: int
    ) -> list[dict[str, Any]]:
        return await asyncio.to_thread(
            self._receive_sync, max_messages, wait_time_seconds, visibility_timeout
        )

    def _receive_sync(
        self, max_messages: int, wait_time_seconds: int, visibility_timeout: int
    ) -> list[dict[str, Any]]:
        # ApproximateReceiveCount / SentTimestamp are the only way the processor
        # can tell a first delivery from a redelivery and give up on a message
        # that keeps failing before it silently reaches the DLQ.
        response = self._client.receive_message(
            QueueUrl=self._queue_url,
            MaxNumberOfMessages=max_messages,
            WaitTimeSeconds=wait_time_seconds,
            VisibilityTimeout=visibility_timeout,
            AttributeNames=["ApproximateReceiveCount", "SentTimestamp"],
        )
        return response.get("Messages", [])

    async def delete(self, receipt_handle: str) -> None:
        await asyncio.to_thread(
            self._client.delete_message,
            QueueUrl=self._queue_url,
            ReceiptHandle=receipt_handle,
        )

    async def change_visibility(self, receipt_handle: str, visibility_timeout: int) -> None:
        await asyncio.to_thread(
            self._client.change_message_visibility,
            QueueUrl=self._queue_url,
            ReceiptHandle=receipt_handle,
            VisibilityTimeout=visibility_timeout,
        )

    async def send(self, body: str) -> str | None:
        response = await asyncio.to_thread(
            self._client.send_message,
            QueueUrl=self._queue_url,
            MessageBody=body,
        )
        message_id = response.get("MessageId") if isinstance(response, dict) else None
        return message_id if isinstance(message_id, str) else None
