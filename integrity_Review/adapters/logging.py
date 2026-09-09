"""Structured, per-review JSON logger.

Every event is one line: a stable human-readable message followed by a JSON
object of fields. CloudWatch Logs Insights can filter on the message and parse
the JSON, so this is the only logging shape the worker and pipeline use.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

LOGGER_NAME = "integrity_review_pipeline"


class StructuredLogger:
    """One instance per review (or per worker scope when ``review_id`` is None).

    ``error=`` may be passed either a string or the exception itself; passing the
    exception adds ``errorType`` and, with ``exc_info=True``, the traceback.
    """

    __slots__ = ("_context", "_logger", "_review_id")

    def __init__(self, review_id: str | None = None, **context: Any) -> None:
        self._review_id = review_id
        self._context = context
        self._logger = logging.getLogger(LOGGER_NAME)

    def bind(self, **context: Any) -> StructuredLogger:
        """Return a logger carrying these extra fields on every event."""

        return StructuredLogger(self._review_id, **{**self._context, **context})

    def _log(self, level: int, message: str, fields: dict[str, Any]) -> None:
        if not self._logger.isEnabledFor(level):
            return
        error = fields.pop("error", None)
        exc_info = fields.pop("exc_info", False)
        payload: dict[str, Any] = {}
        if self._review_id is not None:
            payload["reviewId"] = self._review_id
        payload.update(self._context)
        payload.update(fields)
        if error is not None:
            payload["error"] = str(error)
            if isinstance(error, BaseException):
                payload["errorType"] = type(error).__name__
        self._logger.log(
            level,
            "%s %s",
            message,
            json.dumps(payload, default=str),
            exc_info=error if exc_info and isinstance(error, BaseException) else exc_info,
        )

    def debug(self, message: str, **fields: Any) -> None:
        self._log(logging.DEBUG, message, fields)

    def info(self, message: str, **fields: Any) -> None:
        self._log(logging.INFO, message, fields)

    def warning(self, message: str, **fields: Any) -> None:
        self._log(logging.WARNING, message, fields)

    def error(self, message: str, **fields: Any) -> None:
        self._log(logging.ERROR, message, fields)


def configure_logging(level: str = "INFO") -> None:
    """Send single-line, timestamped logs to stdout for the ECS awslogs driver.

    ``force=True`` replaces any handler a dependency installed at import time,
    which is what otherwise makes container logs disappear or duplicate.
    """

    logging.basicConfig(
        level=level.upper(),
        stream=sys.stdout,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
        force=True,
    )
    # These are chatty at INFO and drown out the pipeline's own events.
    for noisy in ("boto3", "botocore", "urllib3", "httpx", "httpcore", "google_genai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
