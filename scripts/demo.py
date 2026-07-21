#!/usr/bin/env python3
"""Zero-dependency offline demo of the RAGnarok retrieval pipeline.

Runs with just `pip install -e .` — no GPU, no Docker, no LLM, no network. It ingests the sample
corpus using the deterministic local embedder and a model-free heuristic enrichment (the same path
the CI gate uses), then runs hybrid retrieval + cross-encoder rerank for a few queries and prints the
top results. This demonstrates the plumbing end-to-end offline.

For full question-answering (a generated, grounded, cited answer) you need a local LLM — see
docs/SETUP.md, "Full local stack (Ollama)".

    python scripts/demo.py
    python scripts/demo.py "which plans support SSO?"
"""

from __future__ import annotations

import sys
from pathlib import Path

from ragnarok.ingestion.chunking import chunk_document
from ragnarok.ingestion.connectors import load_any
from ragnarok.ingestion.embedding import embed_chunks
from ragnarok.ingestion.enrich import EnrichedDocument, Enrichment
from ragnarok.ingestion.normalize import normalize
from ragnarok.retrieval.hybrid import hybrid_search
from ragnarok.retrieval.preprocess import QueryPlan
from ragnarok.retrieval.rerank import rerank
from ragnarok.stores.vector import InMemoryVectorStore, MetadataFilter
from ragnarok.strategies import available

SAMPLE = str(Path(__file__).resolve().parents[1] / "datasets" / "sample")


def build_index() -> InMemoryVectorStore:
    """Ingest the sample corpus with a model-free heuristic enrichment (no LLM)."""
    store = InMemoryVectorStore()
    n_chunks = 0
    for src in load_any(SAMPLE):
        norm = normalize(src)
        enriched = EnrichedDocument.from_norm(
            norm, Enrichment(), model="heuristic", version="heuristic@0"
        )
        embedded = embed_chunks(chunk_document(enriched))
        store.upsert(embedded)
        n_chunks += len(embedded)
    print(f"Ingested {store.count()} chunks from {SAMPLE} (heuristic enrichment, local embedder)\n")
    return store


def ask(store: InMemoryVectorStore, query: str, top_n: int = 3) -> None:
    plan = QueryPlan(rewritten_query=query, intent="policy_lookup")
    fused = hybrid_search(plan, MetadataFilter(access_tags=["public"]), store)
    results = rerank(query, fused)[:top_n]
    print(f"Q: {query}")
    if not results:
        print("  (no results)\n")
        return
    for i, r in enumerate(results, 1):
        text = (r.payload.get("table_markdown") or r.payload.get("text") or "").replace("\n", " ")
        src = f"{r.payload.get('title', '?')} > {r.payload.get('section', '')}"
        print(f"  [{i}] score={r.rerank_score:.3f}  ({src})")
        print(f"      {text[:160]}")
    print()


def main() -> int:
    print("RAGnarok offline demo — retrieval pipeline (no LLM/GPU/services)\n")
    print(f"Available RAG strategies: {', '.join(available())}\n")
    store = build_index()
    queries = sys.argv[1:] or [
        "What is the refund window for enterprise customers?",
        "Which plans support SAML single sign-on?",
        "how do I enable SSO",
    ]
    for q in queries:
        ask(store, q)
    print("Done. For full grounded answers, run a local LLM (see docs/SETUP.md).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
