"""Evidence manifests produced from test signed URLs."""

from enum import StrEnum

from pydantic import AnyHttpUrl, Field

from .base import ContractModel


class EvidenceType(StrEnum):
    VIDEO = "video"
    AUDIO = "audio"
    SCREEN_RECORDING = "screenRecording"
    KEYSTROKE_DATA = "keystrokeData"


class ExamMode(StrEnum):
    SCREEN = "screen"
    RRWEB = "rrweb"
    NONE = "none"


class EvidenceChunkRef(ContractModel):
    evidence_type: EvidenceType
    chunk_id: str = Field(min_length=1)
    sequence: int = Field(ge=0)
    signed_url: AnyHttpUrl
    duration_ms: int | None = Field(default=None, ge=0)
    section_id: str | None = None
    sidecar_role: str | None = None
    parent_chunk_stem: str | None = None


class EvidenceManifest(ContractModel):
    evidence_type: EvidenceType
    total_chunks: int = Field(ge=0)
    chunks: list[EvidenceChunkRef] = Field(default_factory=list)
    total_duration_ms: int | None = Field(default=None, ge=0)


class ManifestSet(ContractModel):
    exam_mode: ExamMode
    manifests: list[EvidenceManifest]

    def by_type(self, evidence_type: EvidenceType) -> EvidenceManifest:
        return next(
            (
                manifest
                for manifest in self.manifests
                if manifest.evidence_type == evidence_type
            ),
            EvidenceManifest(evidence_type=evidence_type, total_chunks=0),
        )
