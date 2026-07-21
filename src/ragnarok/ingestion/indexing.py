"""Indexing stage: EnrichedDocument -> chunks -> embeddings -> vector store (Steps 8-10).

This is the ``index_fn`` the pipeline calls after enrichment. Kept separate so the pipeline control
flow (Step 5) never changes.
"""

from __future__ import annotations

from typing import Callable

from ragnarok.ingestion.chunking import chunk_document
from ragnarok.ingestion.embedding import embed_chunks
from ragnarok.ingestion.enrich import EnrichedDocument
from ragnarok.stores.vector import VectorStore


def index_document(doc: EnrichedDocument, store: VectorStore, collection: str = "chunks") -> int:
    chunks = chunk_document(doc)
    embedded = embed_chunks(chunks)
    store.upsert(embedded, collection=collection)
    return len(embedded)


def make_index_fn(store: VectorStore, collection: str = "chunks") -> Callable[[object], None]:
    def _index(enriched: object) -> None:
        assert isinstance(enriched, EnrichedDocument)
        index_document(enriched, store, collection)

    return _index
