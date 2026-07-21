"""Golden dataset loading (Step 24)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class GoldenCase:
    id: str
    query: str
    ideal_answer: str = ""
    relevant_source_contains: str = ""  # substring expected in a retrieved source's doc_id/title
    expect: str = "answer"  # answer | refusal
    tags: list[str] = field(default_factory=list)


def load_golden(suite: str) -> list[GoldenCase]:
    path = Path(suite)
    if path.is_dir():
        path = path / "cases.yaml"
    raw = yaml.safe_load(path.read_text()) or []
    return [GoldenCase(**c) for c in raw]
