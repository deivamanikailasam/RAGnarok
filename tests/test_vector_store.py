"""Tests for the vector store: upsert, hybrid search, filtering, security (Step 10)."""

from __future__ import annotations

from ragnarok.ingestion.embedding import EmbeddedChunk
from ragnarok.stores.vector import InMemoryVectorStore, MetadataFilter


def _chunk(cid, text, dense, sparse, **meta) -> EmbeddedChunk:
    return EmbeddedChunk(
        chunk_id=cid, doc_id=meta.get("doc_id", "d1"), text=text, token_count=3,
        content_hash=cid, dense_vector=dense, sparse_vector=sparse,
        embedding_version="v1", metadata=meta,
    )


def _store():
    s = InMemoryVectorStore()
    s.upsert([
        _chunk("a", "enterprise refund 30 days", [1.0, 0.0, 0.0], {1: 1.0, 2: 1.0},
               doc_type="policy", audience=["enterprise"], authority="official", access_tags=["public"]),
        _chunk("b", "consumer refund 14 days", [0.9, 0.1, 0.0], {1: 1.0, 3: 1.0},
               doc_type="policy", audience=["consumer"], authority="official", access_tags=["public"]),
        _chunk("c", "sso saml enablement", [0.0, 0.0, 1.0], {9: 1.0},
               doc_type="runbook", audience=["enterprise"], authority="official", access_tags=["secret"]),
    ])
    return s


def test_dense_search_ranks_by_similarity():
    s = _store()
    hits = s.search_dense([1.0, 0.0, 0.0], filter=None, limit=3)
    assert hits[0].id == "a"


def test_sparse_search_matches_terms():
    s = _store()
    hits = s.search_sparse({2: 1.0}, filter=None, limit=3)
    assert hits[0].id == "a"  # only 'a' has sparse index 2


def test_metadata_filter_scopes_by_tier():
    s = _store()
    flt = MetadataFilter(equals={"audience": ["enterprise"]})
    ids = {h.id for h in s.search_dense([1.0, 0.0, 0.0], filter=flt, limit=5)}
    assert "b" not in ids  # consumer-only chunk excluded


def test_access_tags_enforced_inside_search():
    s = _store()
    # a caller with only 'public' entitlement must never see the 'secret' chunk 'c'
    flt = MetadataFilter(access_tags=["public"])
    ids = {h.id for h in s.search_dense([0.0, 0.0, 1.0], filter=flt, limit=5)}
    assert "c" not in ids


def test_delete_by_doc_and_count():
    s = _store()
    assert s.count() == 3
    s.upsert([_chunk("d", "x", [0.0, 1.0, 0.0], {5: 1.0}, doc_id="d2", access_tags=["public"])])
    assert s.count() == 4
    removed = s.delete_by_doc("d2")
    assert removed == 1 and s.count() == 3
