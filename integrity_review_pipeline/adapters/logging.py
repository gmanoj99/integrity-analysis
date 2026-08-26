"""Structured, per-review JSON logger."""

from __future__ import annotations

import json
import logging
from typing import Any


class StructuredLogger:
    def __init__(self, review_id: str) -> None:
        self._review_id = review_id
        self._logger = logging.getLogger("integrity_review_pipeline")

    def _log(self, level: int, message: str, fields: dict[str, Any]) -> None:
        self._logger.log(
            level,
            "%s %s",
            message,
            json.dumps({"reviewId": self._review_id, **fields}, default=str),
        )

    def debug(self, message: str, **fields: Any) -> None:
        self._log(logging.DEBUG, message, fields)

    def info(self, message: str, **fields: Any) -> None:
        self._log(logging.INFO, message, fields)

    def warning(self, message: str, **fields: Any) -> None:
        self._log(logging.WARNING, message, fields)

    def error(self, message: str, **fields: Any) -> None:
        self._log(logging.ERROR, message, fields)
