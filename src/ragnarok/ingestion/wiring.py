"""Composition root for the ingestion pipeline (Step 5, extended by later steps).

Keeping wiring here means ``pipeline.py``'s control flow never changes as stages are added — each
step (enrich, chunk+embed+store) plugs its stage in here. In Step 5 no stages are wired, so ingest
performs load -> normalize -> hash-diff -> register (idempotency), which is independently useful.
"""

from __future__ import annotations

from ragnarok.ingestion.pipeline import IngestionPipeline
from ragnarok.ingestion.registry import Registry, SqliteRegistry


def build_pipeline(registry: Registry | None = None) -> IngestionPipeline:
    registry = registry or SqliteRegistry()
    # Steps 6/8/9/10/11 will set enrich_fn / index_fn here.
    return IngestionPipeline(registry=registry)
