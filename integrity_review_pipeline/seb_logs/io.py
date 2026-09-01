"""I/O utilities for writing reduced SEB log evidence."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from .contracts import ReducedSessionEvidence


def write_reduced_evidence(
    evidence: ReducedSessionEvidence,
    output_dir: str | Path,
) -> Path:
    """Write reduced evidence to a JSON file.

    Uses atomic temp-file write pattern for safety, following the same
    approach as integrity_review_pipeline/io/review_io.py.

    Args:
        evidence: The reduced session evidence to write
        output_dir: Directory to write the output file

    Returns:
        Path to the written file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    filename = f"{evidence.session_ref.session_folder}.json"
    final_path = output_path / filename

    json_data = evidence.model_dump(
        mode="json",
        by_alias=True,
        exclude_none=True,
    )

    fd, temp_path = tempfile.mkstemp(
        dir=output_path,
        prefix=".tmp_",
        suffix=".json",
    )

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, default=str)

        os.replace(temp_path, final_path)
    except Exception:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise

    return final_path


def write_all_evidence(
    evidences: list[ReducedSessionEvidence],
    output_dir: str | Path,
) -> list[Path]:
    """Write multiple evidence bundles to JSON files.

    Args:
        evidences: List of reduced session evidence to write
        output_dir: Directory to write output files

    Returns:
        List of paths to written files
    """
    paths = []
    for evidence in evidences:
        path = write_reduced_evidence(evidence, output_dir)
        paths.append(path)
    return paths
