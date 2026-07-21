"""Tests for Multimodal RAG: image parsing, captioning, image chunks, strategy (Step 36)."""

from __future__ import annotations

import json
from pathlib import Path

from ragnarok.ingestion.chunking import chunk_document
from ragnarok.ingestion.connectors.files import load_file
from ragnarok.ingestion.enrich import enrich
from ragnarok.ingestion.multimodal import caption_image, set_image_captioner
from ragnarok.ingestion.normalize import normalize
from ragnarok.providers import FakeLLM, reset_clients, set_role_client
from ragnarok.retrieval.preprocess import PreprocessResult, QueryPlan, SourcePlan
from ragnarok.strategies import get_strategy
from ragnarok.strategies.base import StrategyContext
from ragnarok.user import User

SAMPLE = Path(__file__).resolve().parents[1] / "datasets" / "sample"


def test_markdown_image_parsed_as_image_block():
    doc = load_file(SAMPLE / "architecture_overview.md")
    imgs = [b for b in doc.blocks if b.kind == "image"]
    assert len(imgs) == 1
    assert imgs[0].src.endswith("rag_architecture.png")
    assert "diagram" in imgs[0].text.lower()


def test_captioner_hook_overrides_default():
    set_image_captioner(lambda src, alt: f"CAPTION::{alt}")
    assert caption_image("x.png", "a chart") == "CAPTION::a chart"
    set_image_captioner(None)
    assert caption_image("x.png", "a chart") == "a chart"  # default = alt text


async def test_image_becomes_image_chunk():
    reset_clients()
    set_role_client("llm_large", FakeLLM(response=json.dumps({
        "doc_summary": "Architecture overview.", "custom_metadata": {"doc_type": "spec",
        "audience": ["internal"], "authority": "official", "access_tags": ["public"]}})))
    doc = await enrich(normalize(load_file(SAMPLE / "architecture_overview.md")))
    chunks = chunk_document(doc)
    image_chunks = [c for c in chunks if c.chunk_type == "image"]
    assert len(image_chunks) == 1
    ic = image_chunks[0]
    assert ic.metadata["modality"] == "image"
    assert ic.metadata["asset_uri"].endswith("rag_architecture.png")
    assert "diagram" in ic.text.lower()  # caption is embedded
    reset_clients()


async def test_multimodal_strategy_retrieves_across_modalities(sample_index):
    store, features = sample_index
    pre = PreprocessResult(plan=QueryPlan(rewritten_query="architecture diagram vector search embedding",
                                          intent="other"), source=SourcePlan(confidence=0.5))
    ctx = StrategyContext(query=pre.plan.rewritten_query, pre=pre, user=User(entitlements=["public"]),
                          store=store, features=features)
    res = await get_strategy("multimodal").run(ctx)
    assert res.strategy == "multimodal"
    assert res.results
    assert "modalities" in res.notes
    # the image chunk (caption mentions diagram/architecture) should be retrievable
    retrieved_types = {r.payload.get("chunk_type") for r in res.results}
    assert "image" in retrieved_types
