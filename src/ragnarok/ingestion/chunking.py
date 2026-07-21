"""Table-aware, structure-bounded chunking (Step 8).

Chunk quality caps retrieval quality. Rules:
- Respect section boundaries (a chunk never straddles sections -> coherent metadata).
- ~target tokens with ~15% overlap (recover boundary-straddling facts cheaply).
- Contextual prefix on every chunk ("Document > Section. <summary>") so chunks that are meaningless
  in isolation become retrievable ("contextual retrieval").
- Tables become their own chunks carrying THREE representations: NL description (embedded),
  exact markdown (given verbatim to the generator), header keywords (for BM25).
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from ragnarok.config import get_settings
from ragnarok.ingestion.enrich import EnrichedDocument
from ragnarok.ingestion.models import Section, sha256_of
from ragnarok.tokenization import count_tokens

_PARA_RE = re.compile(r"\n\s*\n")
_SENT_RE = re.compile(r"(?<=[.!?])\s+")


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    contextual_prefix: str = ""
    chunk_type: str = "text"  # text | table
    table_markdown: str | None = None
    position: int = 0
    token_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)
    content_hash: str = ""

    def embedding_text(self) -> str:
        """What actually gets embedded: prefix + text (Step 9)."""
        return f"{self.contextual_prefix}\n{self.text}".strip()


def _context_header(doc: EnrichedDocument, section: Section) -> str:
    summary = doc.summary_for(section.heading) if section.heading else doc.enrichment.doc_summary
    loc = f"Document: {doc.title}"
    if section.heading:
        loc += f" > Section: {section.heading}"
    return f"{loc}. {summary}".strip()


def _inherit_metadata(doc: EnrichedDocument, section: Section) -> dict[str, Any]:
    m = doc.enrichment.custom_metadata
    return {
        "title": doc.title,
        "section": section.heading,
        "uri": doc.uri,
        "doc_type": m.doc_type,
        "audience": m.audience,
        "topics": m.topics,
        "entities": m.entities,
        "authority": m.authority,
        "freshness_date": m.freshness_date,
        "access_tags": m.access_tags,
        "keywords": doc.enrichment.keywords,
    }


def _recursive_split(text: str, target: int, overlap: float) -> list[str]:
    """Split into ~target-token pieces on paragraph/sentence boundaries with overlap."""
    units = [u.strip() for u in _PARA_RE.split(text) if u.strip()]
    # further split any paragraph that alone exceeds the target
    fine: list[str] = []
    for u in units:
        if count_tokens(u) > target:
            fine.extend(s.strip() for s in _SENT_RE.split(u) if s.strip())
        else:
            fine.append(u)

    chunks: list[str] = []
    cur: list[str] = []
    cur_tok = 0
    for unit in fine:
        ut = count_tokens(unit)
        if cur and cur_tok + ut > target:
            chunks.append(" ".join(cur))
            # carry overlap: keep trailing units summing to ~overlap*target tokens
            carry: list[str] = []
            carry_tok = 0
            for prev in reversed(cur):
                pt = count_tokens(prev)
                if carry_tok + pt > overlap * target:
                    break
                carry.insert(0, prev)
                carry_tok += pt
            cur = carry
            cur_tok = carry_tok
        cur.append(unit)
        cur_tok += ut
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def _chunk_table(doc: EnrichedDocument, section: Section, position: int) -> list[Chunk]:
    chunks = []
    prefix = _context_header(doc, section)
    meta = _inherit_metadata(doc, section)
    for block in [b for b in section.blocks if b.kind == "table"]:
        desc = doc.table_desc_for(section.heading) or section.text
        md = block.to_markdown()
        meta_t = {**meta, "table_headers": block.headers}
        text = desc  # NL description is what gets embedded (representation 1)
        chunks.append(
            Chunk(
                chunk_id=sha256_of(doc.doc_id, position, md),
                doc_id=doc.doc_id,
                text=text,
                contextual_prefix=prefix,
                chunk_type="table",
                table_markdown=md,  # representation 2: exact structure for the generator
                position=position,
                token_count=count_tokens(text),
                metadata=meta_t,  # representation 3: header keywords for BM25
                content_hash=sha256_of(text, md),
            )
        )
        position += 1
    return chunks


def chunk_document(doc: EnrichedDocument) -> list[Chunk]:
    cfg = get_settings().chunking
    chunks: list[Chunk] = []
    position = 0
    for section in doc.sections:
        if section.is_table:
            table_chunks = _chunk_table(doc, section, position)
            chunks.extend(table_chunks)
            position += len(table_chunks)
            # also chunk any prose in the same section (below/above the table)
            prose = "\n\n".join(b.text for b in section.blocks if b.kind == "text").strip()
            if not prose:
                continue
            section = Section(heading=section.heading, level=section.level,
                              blocks=[b for b in section.blocks if b.kind == "text"])
        prefix = _context_header(doc, section)
        meta = _inherit_metadata(doc, section)
        for piece in _recursive_split(section.text, cfg.target_tokens, cfg.overlap):
            chunks.append(
                Chunk(
                    chunk_id=sha256_of(doc.doc_id, position, piece),
                    doc_id=doc.doc_id,
                    text=piece,
                    contextual_prefix=prefix,
                    chunk_type="text",
                    position=position,
                    token_count=count_tokens(piece),
                    metadata=meta,
                    content_hash=sha256_of(piece),
                )
            )
            position += 1
    return chunks
