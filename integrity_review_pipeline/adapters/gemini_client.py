"""Google Gemini client adapter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


class GoogleGeminiClient:
    def __init__(self, api_key: str) -> None:
        try:
            from google import genai
        except ImportError as error:  # pragma: no cover - dependency error
            raise RuntimeError("google-genai is required for live analysis") from error
        self._client = genai.Client(api_key=api_key)

    async def generate(
        self,
        *,
        model: str,
        contents: Sequence[Mapping[str, Any] | Any],
        config: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        response = await self._client.aio.models.generate_content(
            model=model,
            contents=list(contents),
            config=dict(config),
        )
        text = response.text or "{}"
        return {"text": text}
