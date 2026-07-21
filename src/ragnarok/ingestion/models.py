"""Canonical ingestion data model (Step 4).

These objects flow through the whole offline plane (connectors -> normalize -> enrich -> chunk ->
embed). Every object is content-hashed so ingestion is idempotent and incremental (Step 5). Tables
are first-class (``Block.kind == "table"``) rather than flattened to text, which is what enables
table-aware chunking (Step 8) and exact-value answers (Step 20).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


def sha256_of(*parts: Any) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(json.dumps(p, sort_keys=True, ensure_ascii=False, default=str).encode())
    return h.hexdigest()


class Block(BaseModel):
    """A structural unit from a source document."""

    kind: Literal["text", "table"] = "text"
    text: str = ""
    heading_level: int = 0  # 0 = body, 1 = H1, 2 = H2 ...
    # table-only fields
    headers: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)

    def to_markdown(self) -> str:
        if self.kind != "table" or not self.rows:
            return self.text
        cols = self.headers or [f"c{i}" for i in range(len(self.rows[0]))]
        out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
        for r in self.rows:
            out.append("| " + " | ".join(str(c) for c in r) + " |")
        return "\n".join(out)


class SourceDocument(BaseModel):
    doc_id: str
    uri: str
    source_type: Literal["gdoc", "file", "confluence", "notion", "web", "db"] = "file"
    title: str = ""
    blocks: list[Block] = Field(default_factory=list)
    acl_tags: list[str] = Field(default_factory=list)
    fetched_at: Optional[str] = None
    content_hash: str = ""

    def compute_hash(self) -> str:
        # timestamp deliberately excluded so identical content re-fetches to the same hash (Step 5)
        return sha256_of(self.uri, self.title, [b.model_dump() for b in self.blocks])

    def with_hash(self) -> "SourceDocument":
        self.content_hash = self.compute_hash()
        return self


class Section(BaseModel):
    """A heading-bounded region: coherent metadata, one chunking target (Step 8)."""

    heading: str = ""
    level: int = 0
    blocks: list[Block] = Field(default_factory=list)

    @property
    def is_table(self) -> bool:
        return any(b.kind == "table" for b in self.blocks)

    @property
    def text(self) -> str:
        return "\n\n".join(b.to_markdown() for b in self.blocks).strip()

    def block_hash(self) -> str:
        return sha256_of(self.heading, [b.model_dump() for b in self.blocks])


class NormalizedDoc(BaseModel):
    doc_id: str
    uri: str
    source_type: str
    title: str
    sections: list[Section] = Field(default_factory=list)
    acl_tags: list[str] = Field(default_factory=list)
    content_hash: str = ""

    def render_for_llm(self, max_chars: int = 12000) -> str:
        """Compact rendering used as enrichment input (Step 6)."""
        parts = [f"# {self.title}"]
        for s in self.sections:
            if s.heading:
                parts.append(f"{'#' * max(s.level, 1)} {s.heading}")
            parts.append(s.text)
        return "\n\n".join(parts)[:max_chars]
