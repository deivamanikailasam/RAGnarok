"""Indexing stage: EnrichedDocument -> chunks -> embeddings -> vector store (Steps 8-10).

This is the ``index_fn`` the pipeline calls after enrichment. Kept separate so the pipeline control
flow (Step 5) never changes.
"""

from __future__ import annotations

from collections.abc import Callable

from ragnarok.ingestion.chunking import chunk_document
from ragnarok.ingestion.embedding import EmbeddedChunk, embed_chunks
from ragnarok.ingestion.enrich import EnrichedDocument
from ragnarok.stores.features import FeatureStore, features_from_enriched
from ragnarok.stores.vector import VectorStore, chunk_to_payload


def index_document(
    doc: EnrichedDocument, store: VectorStore, collection: str = "chunks"
) -> list[EmbeddedChunk]:
    chunks = chunk_document(doc)
    embedded = embed_chunks(chunks)
    store.upsert(embedded, collection=collection)
    return embedded


def make_index_fn(
    store: VectorStore,
    collection: str = "chunks",
    feature_store: FeatureStore | None = None,
    graph: object | None = None,
) -> Callable[[object], None]:
    def _index(enriched: object) -> None:
        assert isinstance(enriched, EnrichedDocument)
        embedded = index_document(enriched, store, collection)
        # Step 11: update the document-level feature row (authority/freshness/doc_type).
        if feature_store is not None:
            feature_store.upsert_document(features_from_enriched(enriched))
        # Step 34: populate the knowledge graph (entities + co-occurrence).
        if graph is not None:
            for chunk in embedded:
                graph.add_chunk(chunk.chunk_id, chunk_to_payload(chunk))

    return _index
