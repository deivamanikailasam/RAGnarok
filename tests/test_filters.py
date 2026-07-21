"""Tests for access-scoped filter building (Step 15)."""

from __future__ import annotations

from ragnarok.ingestion.embedding import EmbeddedChunk
from ragnarok.retrieval.filters import build_filter, relax_filter
from ragnarok.retrieval.preprocess import SourcePlan
from ragnarok.stores.vector import InMemoryVectorStore, MetadataFilter
from ragnarok.user import User


def _chunk(cid, text, dense, sparse, **meta):
    return EmbeddedChunk(chunk_id=cid, doc_id="d", text=text, token_count=3, content_hash=cid,
                         dense_vector=dense, sparse_vector=sparse, embedding_version="v1", metadata=meta)


def test_model_cannot_widen_access_beyond_entitlements():
    # model tries to reach a 'secret' scope; caller only has 'public'
    source = SourcePlan(equals={"doc_type": ["policy"]}, must_access_tags=["secret"], confidence=0.9)
    flt = build_filter(source, User(entitlements=["public"]))
    assert flt.access_tags == ["public"]  # intersection empty -> falls back to entitlements
    assert "secret" not in flt.access_tags
    assert flt.equals == {"doc_type": ["policy"]}


def test_model_can_narrow_within_entitlements():
    source = SourcePlan(must_access_tags=["team:sales"], confidence=0.9)
    flt = build_filter(source, User(entitlements=["team:sales", "team:eng"]))
    assert flt.access_tags == ["team:sales"]  # narrowed, still within entitlements


def test_relax_filter_keeps_security():
    flt = relax_filter(User(entitlements=["public", "team:cs"]))
    assert flt.equals == {}
    assert set(flt.access_tags) == {"public", "team:cs"}


def test_filter_prevents_scoring_unauthorized_chunk():
    store = InMemoryVectorStore()
    store.upsert([
        _chunk("pub", "public policy", [1.0, 0.0], {1: 1.0}, access_tags=["public"]),
        _chunk("sec", "secret salaries", [1.0, 0.0], {1: 1.0}, access_tags=["secret"]),
    ])
    flt: MetadataFilter = build_filter(None, User(entitlements=["public"]))
    ids = {h.id for h in store.search_dense([1.0, 0.0], filter=flt, limit=10)}
    assert ids == {"pub"}  # secret chunk never scored
