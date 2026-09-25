from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any


class InMemoryPerceptionCache:
    def __init__(self) -> None:
        self._values: dict[str, Mapping[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Mapping[str, Any] | None:
        async with self._lock:
            return self._values.get(key)

    async def set(self, key: str, value: Mapping[str, Any]) -> None:
        async with self._lock:
            self._values[key] = value
