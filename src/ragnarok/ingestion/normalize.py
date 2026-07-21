"""Normalization: SourceDocument -> NormalizedDoc grouped into heading-bounded sections (Step 4).

Cleans whitespace/unicode, de-hyphenates soft line breaks, and groups blocks under their nearest
heading so each Section carries coherent metadata (a chunk never straddles two sections -> its
tier/authority/topic stay consistent, Step 8).
"""

from __future__ import annotations

import re
import unicodedata

from ragnarok.ingestion.models import Block, NormalizedDoc, Section, SourceDocument

_WS_RE = re.compile(r"[ \t]+")
_HYPHEN_BREAK_RE = re.compile(r"(\w)-\n(\w)")


def _clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _HYPHEN_BREAK_RE.sub(r"\1\2", text)  # de-hyphenate wrapped words (PDFs)
    text = _WS_RE.sub(" ", text)
    return text.strip()


def normalize(doc: SourceDocument) -> NormalizedDoc:
    sections: list[Section] = []
    current = Section(heading="", level=0, blocks=[])

    for block in doc.blocks:
        if block.kind == "text" and block.heading_level > 0:
            # start a new section at a heading
            if current.blocks or current.heading:
                sections.append(current)
            current = Section(heading=_clean_text(block.text), level=block.heading_level, blocks=[])
        elif block.kind == "table":
            current.blocks.append(block)  # tables kept structured, not cleaned as prose
        else:
            cleaned = _clean_text(block.text)
            if cleaned:
                current.blocks.append(Block(kind="text", text=cleaned))
    if current.blocks or current.heading:
        sections.append(current)

    # drop fully-empty sections
    sections = [s for s in sections if s.blocks or s.heading]

    return NormalizedDoc(
        doc_id=doc.doc_id,
        uri=doc.uri,
        source_type=doc.source_type,
        title=_clean_text(doc.title),
        sections=sections,
        acl_tags=doc.acl_tags,
        content_hash=doc.content_hash or doc.compute_hash(),
    )
