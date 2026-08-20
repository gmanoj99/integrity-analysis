"""Read the local test input and write the final evidence bundle."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from ..contracts.review import ReviewInput, ReviewRequest
from .manifest_builder import build_manifest_set


def load_review_request(path: str | Path) -> ReviewRequest:
    input_path = Path(path)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    review_input = ReviewInput.model_validate(payload)
    return ReviewRequest(
        candidate_id=review_input.candidate_id,
        assessment_id=review_input.assessment_id,
        activity_timeline=review_input.activity_timeline,
        sections=review_input.sections,
        evidence=build_manifest_set(review_input),
    )


def write_evidence_bundle(bundle: BaseModel | dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        bundle.model_dump(mode="json", by_alias=True, exclude_none=True)
        if isinstance(bundle, BaseModel)
        else bundle
    )
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(output_path)
