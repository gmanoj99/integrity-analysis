#!/usr/bin/env python3
"""Run one local signed-URL review from input.json to evidence_bundle.json."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from integrity_review_pipeline.adapters import build_default_deps
from integrity_review_pipeline.io import load_review_request, write_evidence_bundle
from integrity_review_pipeline.pipeline import run_integrity_review


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "input" / "input.json",
        help="Path to signed-URL input JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "output" / "evidence_bundle.json",
        help="Path for the final EvidenceBundle JSON",
    )
    return parser.parse_args()


async def run(input_path: Path, output_path: Path) -> None:
    request = load_review_request(input_path)
    deps = build_default_deps(
        review_id=f"{request.assessment_id}:{request.candidate_id}"
    )
    bundle = await run_integrity_review(request, deps)
    write_evidence_bundle(bundle, output_path)


def main() -> None:
    args = parse_args()
    asyncio.run(run(args.input, args.output))
    print(f"EvidenceBundle written to {args.output}")


if __name__ == "__main__":
    main()
