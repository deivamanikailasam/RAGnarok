"""Composition root for the ingestion pipeline (extended per step).

Keeping wiring here means ``pipeline.py``'s control flow never changes as stages are added — each
step plugs its stage in here: Step 6 enrichment, Steps 8-10 chunk->embed->store.
"""

from __future__ import annotations

from ragnarok.ingestion.enrich import enrich_sync
from ragnarok.ingestion.indexing import make_index_fn
from ragnarok.ingestion.pipeline import IngestionPipeline
from ragnarok.ingestion.registry import Registry, SqliteRegistry
from ragnarok.stores.factory import get_vector_store
from ragnarok.stores.vector import VectorStore


def build_pipeline(
    registry: Registry | None = None,
    store: VectorStore | None = None,
    collection: str = "chunks",
) -> IngestionPipeline:
    registry = registry or SqliteRegistry()
    store = store or get_vector_store()
    return IngestionPipeline(
        registry=registry,
        enrich_fn=enrich_sync,  # Step 6
        index_fn=make_index_fn(store, collection),  # Steps 8-10
    )
