"""Tests for idempotent/incremental ingestion (Step 5)."""

from __future__ import annotations

from pathlib import Path

from ragnarok.ingestion.pipeline import IngestionPipeline
from ragnarok.ingestion.registry import DocRecord, InMemoryRegistry, SqliteRegistry

SAMPLE = str(Path(__file__).resolve().parents[1] / "datasets" / "sample")


def test_first_run_processes_second_run_skips():
    reg = InMemoryRegistry()
    seen = []
    pipe = IngestionPipeline(registry=reg, index_fn=lambda doc: seen.append(doc))

    first = pipe.run([SAMPLE])
    assert first.processed >= 2 and first.skipped == 0
    n_indexed = len(seen)

    second = pipe.run([SAMPLE])  # unchanged corpus
    assert second.processed == 0
    assert second.skipped == first.processed
    assert len(seen) == n_indexed  # index not called again


def test_full_rebuild_reprocesses_everything():
    reg = InMemoryRegistry()
    pipe = IngestionPipeline(registry=reg)
    pipe.run([SAMPLE])
    rebuilt = pipe.run([SAMPLE], full_rebuild=True)
    assert rebuilt.processed >= 2 and rebuilt.skipped == 0


def test_enrich_stage_is_invoked_when_wired():
    reg = InMemoryRegistry()
    calls = []
    pipe = IngestionPipeline(
        registry=reg,
        enrich_fn=lambda norm: calls.append(norm.doc_id) or norm,
    )
    pipe.run([SAMPLE])
    assert len(calls) >= 2


def test_sqlite_registry_roundtrip(tmp_path):
    reg = SqliteRegistry(tmp_path / "reg.db")
    reg.upsert(DocRecord("d1", "hash1", status="ingested"))
    assert reg.is_unchanged("d1", "hash1")
    assert not reg.is_unchanged("d1", "hash2")  # changed content
    reg.delete("d1")
    assert reg.get("d1") is None
