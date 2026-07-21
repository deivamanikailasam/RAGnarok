"""Tests for blue/green reindex via alias swap (Step 12)."""

from __future__ import annotations

import pytest

from ragnarok.ingestion.embedding import EmbeddedChunk
from ragnarok.stores.collections import BlueGreenReindexer, _next_version_name
from ragnarok.stores.vector import InMemoryVectorStore


def _chunk(cid, text, doc_id="d1"):
    return EmbeddedChunk(chunk_id=cid, doc_id=doc_id, text=text, token_count=2, content_hash=cid,
                         dense_vector=[1.0, 0.0], sparse_vector={1: 1.0}, embedding_version="v1",
                         metadata={"access_tags": ["public"]})


def test_version_naming():
    assert _next_version_name("chunks", []) == "chunks_v1"
    assert _next_version_name("chunks", ["chunks_v1", "chunks_v2"]) == "chunks_v3"


def test_reindex_swaps_alias_only_after_build():
    store = InMemoryVectorStore()
    # v1 live
    store.upsert([_chunk("a", "old content")], collection="chunks_v1")
    store.switch_alias("chunks", "chunks_v1")
    assert store.count("chunks") == 1

    reindexer = BlueGreenReindexer(store, alias="chunks")
    old = reindexer.current_target()

    def build(new_coll):
        store.upsert([_chunk("a", "new"), _chunk("b", "new2")], collection=new_coll)
        # during build, the alias still points at the old collection (no partial exposure)
        assert reindexer.current_target() == "chunks_v1"

    new = reindexer.reindex(build, validate=lambda c: store.count(c) == 2)
    assert new == "chunks_v2"
    assert store.get_alias_target("chunks") == "chunks_v2"
    assert store.count("chunks") == 2  # alias now resolves to the new collection

    # rollback restores the previous version instantly
    reindexer.rollback(old)
    assert store.count("chunks") == 1


def test_reindex_aborts_on_validation_failure():
    store = InMemoryVectorStore()
    store.upsert([_chunk("a", "old")], collection="chunks_v1")
    store.switch_alias("chunks", "chunks_v1")
    reindexer = BlueGreenReindexer(store)

    with pytest.raises(RuntimeError):
        reindexer.reindex(
            lambda c: store.upsert([_chunk("x", "bad")], collection=c),
            validate=lambda c: False,  # simulate eval-gate failure
        )
    # alias untouched: old index still serving
    assert store.get_alias_target("chunks") == "chunks_v1"
