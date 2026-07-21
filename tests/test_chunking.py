"""Tests for table-aware chunking (Step 8)."""

from __future__ import annotations

import json
from pathlib import Path

from ragnarok.ingestion.chunking import chunk_document
from ragnarok.ingestion.connectors.files import load_file
from ragnarok.ingestion.enrich import enrich
from ragnarok.ingestion.normalize import normalize
from ragnarok.providers import FakeLLM, reset_clients, set_role_client

SAMPLE = Path(__file__).resolve().parents[1] / "datasets" / "sample"

_ENR = {
    "doc_summary": "Refund policy with tier-dependent windows.",
    "section_summaries": [{"heading": "Overview", "summary": "Refunds within a tier window."}],
    "table_descriptions": [
        {"section_heading": "Refund Windows by Tier",
         "description": "Refund windows by tier: enterprise annual 30 days, consumer monthly 14 days."}
    ],
    "custom_metadata": {"doc_type": "policy", "audience": ["enterprise", "consumer"],
                         "authority": "official", "freshness_date": "2026-03-01",
                         "topics": ["refunds"], "entities": ["Enterprise"], "access_tags": ["public"]},
    "keywords": ["refund", "enterprise"],
}


async def _enriched():
    reset_clients()
    set_role_client("llm_large", FakeLLM(response=json.dumps(_ENR)))
    doc = await enrich(normalize(load_file(SAMPLE / "refund_policy.md")))
    reset_clients()
    return doc


async def test_table_becomes_its_own_chunk_with_three_representations():
    doc = await _enriched()
    chunks = chunk_document(doc)
    table_chunks = [c for c in chunks if c.chunk_type == "table"]
    assert len(table_chunks) == 1
    tc = table_chunks[0]
    # rep 1: NL description embedded
    assert "30 days" in tc.text
    # rep 2: exact markdown for the generator
    assert tc.table_markdown and "| Tier" in tc.table_markdown
    # rep 3: header keywords for BM25
    assert "Tier" in tc.metadata["table_headers"]


async def test_every_chunk_has_context_prefix_and_metadata():
    doc = await _enriched()
    chunks = chunk_document(doc)
    assert chunks
    for c in chunks:
        assert c.contextual_prefix.startswith("Document: ")
        assert c.metadata["doc_type"] == "policy"
        assert c.metadata["authority"] == "official"
        assert c.token_count > 0
        assert "Refund Policy" in c.embedding_text() or c.contextual_prefix in c.embedding_text()


async def test_chunks_respect_target_size():
    doc = await _enriched()
    chunks = chunk_document(doc)
    # text chunks should not vastly exceed the target (384) — allow some slack for atomic units
    for c in chunks:
        if c.chunk_type == "text":
            assert c.token_count <= 384 * 2
