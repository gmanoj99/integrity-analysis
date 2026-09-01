#!/usr/bin/env python3
"""Run SEB log reduction pipeline from S3 to JSON output files.

This is the CLI entrypoint for Process 1 (Trigger) and Process 2 (Deterministic Analysis).
It discovers all SEB log sessions under the given S3 prefix and produces reduced JSON
output files for each session.

Usage:
    python run_seb_log_reduction.py --bucket my-bucket --prefix topin_prod/media/tsb_logs/
    python run_seb_log_reduction.py --bucket my-bucket --prefix topin_prod/media/tsb_logs/ --org-assessment-id 18b5742d-...
    python run_seb_log_reduction.py --local /path/to/logs --prefix topin_prod/media/tsb_logs/

Output files are written to output/seb_logs/{session_folder}.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from integrity_review_pipeline.adapters.s3_object_store import (
    LocalFileStore,
    S3ObjectStore,
)
from integrity_review_pipeline.seb_logs.deps import SebLogPipelineDeps
from integrity_review_pipeline.seb_logs.io import write_reduced_evidence
from integrity_review_pipeline.seb_logs.pipeline import run_seb_log_reduction


class StructuredLogger:
    """Simple structured logger for CLI usage."""

    def __init__(self, name: str = "seb_log_reduction") -> None:
        self._logger = logging.getLogger(name)

    def _log(self, level: int, message: str, fields: dict[str, Any]) -> None:
        self._logger.log(
            level,
            "%s %s",
            message,
            json.dumps(fields, default=str) if fields else "",
        )

    def debug(self, message: str, **fields: Any) -> None:
        self._log(logging.DEBUG, message, fields)

    def info(self, message: str, **fields: Any) -> None:
        self._log(logging.INFO, message, fields)

    def warning(self, message: str, **fields: Any) -> None:
        self._log(logging.WARNING, message, fields)

    def error(self, message: str, **fields: Any) -> None:
        self._log(logging.ERROR, message, fields)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    storage_group = parser.add_mutually_exclusive_group(required=True)
    storage_group.add_argument(
        "--bucket",
        type=str,
        help="S3 bucket name containing the log files",
    )
    storage_group.add_argument(
        "--local",
        type=Path,
        help="Path to local directory containing log files (for testing)",
    )

    parser.add_argument(
        "--prefix",
        type=str,
        required=True,
        help="S3 prefix or local path prefix to search for sessions",
    )
    parser.add_argument(
        "--region",
        type=str,
        default="ap-south-1",
        help="AWS region for S3 bucket (default: ap-south-1)",
    )
    parser.add_argument(
        "--org-assessment-id",
        type=str,
        help="Filter sessions by org assessment ID",
    )
    parser.add_argument(
        "--user-id",
        type=str,
        help="Filter sessions by user ID",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "output" / "seb_logs",
        help="Directory for output JSON files (default: output/seb_logs/)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    return parser.parse_args()


def build_deps(args: argparse.Namespace) -> SebLogPipelineDeps:
    """Build pipeline dependencies from CLI arguments."""
    logger = StructuredLogger()

    if args.bucket:
        object_store = S3ObjectStore(
            bucket=args.bucket,
            region_name=args.region,
        )
    else:
        object_store = LocalFileStore(
            base_path=str(args.local),
        )

    return SebLogPipelineDeps(
        object_store=object_store,
        logger=logger,
    )


def main() -> int:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    deps = build_deps(args)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    session_count = 0
    error_count = 0

    try:
        for evidence in run_seb_log_reduction(
            deps=deps,
            bucket_prefix=args.prefix,
            org_assessment_id=args.org_assessment_id,
            user_id=args.user_id,
        ):
            session_count += 1

            try:
                output_path = write_reduced_evidence(evidence, output_dir)
                print(f"Written: {output_path}")
            except Exception as e:
                error_count += 1
                print(f"Error writing output: {e}", file=sys.stderr)

    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"Pipeline error: {e}", file=sys.stderr)
        return 1

    print(f"\nProcessed {session_count} session(s)")
    if error_count > 0:
        print(f"Errors: {error_count}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
