#!/usr/bin/env python3
"""One-shot generator for large behavioral port modules from embedded Python sources."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "integrity_review_pipeline"


def write(rel: str, content: str) -> None:
    path = PKG / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(f"wrote {path} ({len(content.splitlines())} lines)")


if __name__ == "__main__":
    print("Run individual module writers from this package.")
