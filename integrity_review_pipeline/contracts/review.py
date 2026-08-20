"""Local test input and normalized pipeline request contracts."""

from typing import Any

from pydantic import Field, field_validator

from .base import ContractModel
from .evidence import ManifestSet


class SectionInput(ContractModel):
    exam_attempt_id: str = Field(min_length=1)
    exam_id: str = Field(min_length=1)
    section_type: str | None = None
    title: str | None = None

    @property
    def section_id(self) -> str:
        if self.section_type:
            return self.section_type.lower()
        if self.title:
            return "-".join(self.title.lower().split())
        return self.exam_attempt_id[:8]


class ReviewInput(ContractModel):
    """Exact shape accepted from ``input/input.json``."""

    candidate_id: str = Field(min_length=1)
    assessment_id: str = Field(min_length=1)
    camera_recordings: list[str] = Field(min_length=1)
    session_recordings: list[str] = Field(default_factory=list)
    activity_timeline: list[dict[str, Any]] = Field(min_length=1)
    sections: list[SectionInput] = Field(min_length=1)

    @field_validator("camera_recordings", "session_recordings")
    @classmethod
    def require_https_urls(cls, urls: list[str]) -> list[str]:
        invalid = [url for url in urls if not url.startswith("https://")]
        if invalid:
            raise ValueError("recording URLs must use https")
        return urls


class ReviewRequest(ContractModel):
    candidate_id: str
    assessment_id: str
    activity_timeline: list[dict[str, Any]]
    sections: list[SectionInput]
    evidence: ManifestSet
