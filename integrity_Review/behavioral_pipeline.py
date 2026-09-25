from __future__ import annotations

from dataclasses import dataclass

from .baseline import build_statistical_baseline
from .baseline.contracts import StatisticalBaseline
from .contracts.evidence import ExamMode
from .contracts.perception import PerceptionBundle
from .contracts.timeline import MasterTimeline
from .correlation import derive_correlated_signals
from .correlation.contracts import CorrelatedSignalsBundle
from .findings import (
    derive_keystroke_findings,
    derive_screen_findings,
    derive_video_findings,
    derive_video_findings_from_perception,
)
from .findings.contracts import (
    EvidenceFindingsResult,
    ScreenPerceptionBundle,
    VideoDerivationResult,
    VideoObservationWindow,
)
from .machine_facts import build_machine_facts
from .machine_facts.contracts import ClientReportedEvent, MachineFactsBundle, RrwebChunkEvents
from .media.keystroke_reader import decode_rrweb_chunk


@dataclass(frozen=True, slots=True)
class BehavioralArtifacts:
    machine_facts: MachineFactsBundle
    keystroke_findings: EvidenceFindingsResult | None
    screen_findings: EvidenceFindingsResult | None
    video_findings: EvidenceFindingsResult | VideoDerivationResult | None
    baseline: StatisticalBaseline
    correlated_signals: CorrelatedSignalsBundle


def decode_rrweb_chunks(
    chunk_payloads: list[tuple[str, int, bytes]],
) -> list[RrwebChunkEvents]:
    return [
        RrwebChunkEvents(
            chunk_id=chunk_id,
            sequence=sequence,
            events=decode_rrweb_chunk(raw),
        )
        for chunk_id, sequence, raw in chunk_payloads
    ]


def run_behavioral_analysis(
    timeline: MasterTimeline,
    *,
    exam_mode: ExamMode,
    rrweb_chunks: list[RrwebChunkEvents] | None = None,
    screen_bundle: ScreenPerceptionBundle | None = None,
    perception_bundle: PerceptionBundle | None = None,
    video_windows: list[VideoObservationWindow] | None = None,
    client_reported_events: list[ClientReportedEvent] | None = None,
) -> BehavioralArtifacts:
    screen_synthetic = None
    screen_findings_result: EvidenceFindingsResult | None = None
    if exam_mode == ExamMode.SCREEN and screen_bundle is not None:
        screen_findings_result, screen_synthetic = derive_screen_findings(screen_bundle)

    machine_facts = build_machine_facts(
        timeline,
        exam_mode=exam_mode,
        rrweb_chunks=rrweb_chunks if exam_mode == ExamMode.RRWEB else None,
        client_reported_events=client_reported_events,
        screen_synthetic_facts=screen_synthetic,
    )

    keystroke_findings = (
        derive_keystroke_findings(machine_facts)
        if exam_mode == ExamMode.RRWEB
        else None
    )
    video_findings = (
        derive_video_findings_from_perception(perception_bundle)
        if perception_bundle is not None
        else derive_video_findings(video_windows)
        if video_windows
        else None
    )
    baseline = build_statistical_baseline(
        machine_facts,
        video_windows=video_windows,
    )
    correlated = derive_correlated_signals(
        machine_facts=machine_facts,
        baseline=baseline,
        video_findings=video_findings.findings if video_findings else None,
        screen_findings=screen_findings_result.findings if screen_findings_result else None,
    )
    return BehavioralArtifacts(
        machine_facts=machine_facts,
        keystroke_findings=keystroke_findings,
        screen_findings=screen_findings_result,
        video_findings=video_findings,
        baseline=baseline,
        correlated_signals=correlated,
    )
