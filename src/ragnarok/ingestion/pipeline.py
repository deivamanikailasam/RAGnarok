"""Ingestion orchestration (Step 5, extended by Steps 6/8/9/10/11).

Idempotent + incremental: hash-diff against the registry, skip unchanged docs. Later steps insert
their stage into ``_process_document`` (enrich -> chunk -> embed -> store) so the pipeline grows
without changing this control flow. Decoupled from serving (runs on a queue, Step 5 note) so a big
re-ingest never affects live query latency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from ragnarok.ingestion.connectors import load_any
from ragnarok.ingestion.models import NormalizedDoc, SourceDocument
from ragnarok.ingestion.normalize import normalize
from ragnarok.ingestion.registry import DocRecord, Registry, SqliteRegistry


@dataclass
class IngestSummary:
    processed: int = 0
    skipped: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"ingest: processed={self.processed} skipped={self.skipped} "
            f"failed={self.failed}"
        )


class IngestionPipeline:
    def __init__(
        self,
        registry: Optional[Registry] = None,
        *,
        # later steps inject these; None keeps the stage a no-op so the pipeline stays runnable
        enrich_fn: Callable[[NormalizedDoc], object] | None = None,
        index_fn: Callable[[object], None] | None = None,
    ) -> None:
        self.registry: Registry = registry or SqliteRegistry()
        self._enrich_fn = enrich_fn
        self._index_fn = index_fn

    def _process_document(self, src: SourceDocument) -> None:
        norm = normalize(src)
        # Stage: enrichment (Step 6) — no-op until an enricher is injected.
        enriched: object = self._enrich_fn(norm) if self._enrich_fn else norm
        # Stage: chunk + embed + store (Steps 8-10) — no-op until an indexer is injected.
        if self._index_fn:
            self._index_fn(enriched)

    def run(self, paths: list[str], *, full_rebuild: bool = False) -> IngestSummary:
        summary = IngestSummary()
        for uri in paths:
            for src in load_any(uri):
                if not full_rebuild and self.registry.is_unchanged(src.doc_id, src.content_hash):
                    summary.skipped += 1
                    continue
                try:
                    self._process_document(src)
                    self.registry.upsert(
                        DocRecord(src.doc_id, src.content_hash, status="ingested")
                    )
                    summary.processed += 1
                except Exception as exc:  # noqa: BLE001 - record & continue, don't abort the batch
                    self.registry.upsert(
                        DocRecord(src.doc_id, src.content_hash, status="failed")
                    )
                    summary.failed += 1
                    summary.errors.append(f"{src.doc_id}: {exc}")
        return summary


def ingest_path(path: str, *, full_rebuild: bool = False) -> int:
    """CLI entrypoint (Step 5). Later steps wire real enrich/index functions here."""
    from ragnarok.ingestion.wiring import build_pipeline

    pipeline = build_pipeline()
    summary = pipeline.run([str(Path(path))], full_rebuild=full_rebuild)
    print(summary)
    for err in summary.errors:
        print("  error:", err)
    return 0 if summary.failed == 0 else 1
