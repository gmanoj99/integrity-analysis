#!/usr/bin/env python3
"""Generate findings/video_derivation.py from embedded source."""

from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "integrity_review_pipeline/findings/video_derivation.py"

# Full port lives in generated file; run: python scripts/gen_video_derivation.py
SOURCE = Path(__file__).with_name("_video_derivation_body.py")
OUT.write_text(SOURCE.read_text(encoding="utf-8"), encoding="utf-8")
print(f"wrote {OUT} ({len(OUT.read_text().splitlines())} lines)")
