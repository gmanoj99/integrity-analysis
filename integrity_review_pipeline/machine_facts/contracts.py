"""Machine facts compute-once contracts."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from ..contracts.base import ContractModel
from .kinds import MachineFactKind

MACHINE_FACTS_LOGIC_VERSION = "mf-py-v2-ts-parity"

FactSource = Literal[
    "rrweb_deterministic", "application_event", "video_cv", "client_reported"
]
EvidenceSource = Literal["video", "keystroke", "screen", "client_reported"]
Attribution = Literal["candidate", "environment", "other_person", "unclear"]


class MachineFact(ContractModel):
    id: str
    start_offset_ms: int = Field(ge=0)
    end_offset_ms: int = Field(ge=0)
    raw_timestamp_ms: int | None = None
    kind: str
    source: FactSource
    evidence_source: EvidenceSource
    attribution: Attribution = "unclear"
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    detail: dict[str, Any] = Field(default_factory=dict)


class MachineFactCategoryCounts(ContractModel):
    assessment_lifecycle: int = 0
    section_lifecycle: int = 0
    focus_and_visibility: int = 0
    face_and_camera: int = 0
    input_and_clipboard: int = 0
    environment: int = 0
    activity_gaps: int = 0
    scoring_and_submissions: int = 0
    other: int = 0


class MachineFactsSummary(ContractModel):
    total_facts: int
    by_kind: dict[str, int] = Field(default_factory=dict)
    by_section: dict[str, int] = Field(default_factory=dict)
    categories: MachineFactCategoryCounts


class ClientReportedEvent(ContractModel):
    """Optional proctoring / platform events supplied with the review request."""

    timestamp_ms: int
    kind: str
    evidence_type: EvidenceSource = "client_reported"
    signal_source: Literal["client_reported", "rrweb_deterministic", "platform_cv"] = (
        "client_reported"
    )
    detail: dict[str, Any] = Field(default_factory=dict)
    source_ref: dict[str, Any] | None = None


class MachineFactsBundle(ContractModel):
    """Immutable Scope 1 artifact — downstream scopes read only this bundle."""

    model_config = ContractModel.model_config | {"frozen": True}

    candidate_id: str
    assessment_id: str
    produced_at: str
    logic_version: str = MACHINE_FACTS_LOGIC_VERSION
    session_start_ms: int
    duration_ms: int
    facts: list[MachineFact]
    summary: MachineFactsSummary
    limitations: list[str] = Field(default_factory=list)
    exam_mode: Literal["screen", "rrweb", "none"] = "none"


class RrwebChunkEvents(ContractModel):
    chunk_id: str
    sequence: int
    events: list[dict[str, Any]]
