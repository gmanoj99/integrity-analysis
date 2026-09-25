from typing import Any

from pydantic import Field

from .base import ContractModel
from .evidence import ManifestSet


class SectionInput(ContractModel):
    section_id: str = Field(min_length=1)
    exam_attempt_id: str | None = None
    exam_id: str | None = None
    order: int | None = None
    section_type: str | None = None
    title: str | None = None
    start_datetime: str | None = None
    end_datetime: str | None = None


class ReviewRequest(ContractModel):
    candidate_id: str
    assessment_id: str
    activity_timeline: list[dict[str, Any]]
    sections: list[SectionInput]
    evidence: ManifestSet
