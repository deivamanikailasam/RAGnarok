"""Tests for hybrid retrieval + RRF fusion (Step 16)."""

from __future__ import annotations

from ragnarok.retrieval.filters import relax_filter
from ragnarok.retrieval.hybrid import RetrievalResult, hybrid_search, rrf_fuse
from ragnarok.retrieval.preprocess import QueryPlan
from ragnarok.stores.vector import MetadataFilter, SearchHit
from ragnarok.user import User


def test_rrf_fusion_rewards_agreement():
    dense = [SearchHit("a", 0.9), SearchHit("b", 0.8)]
    sparse = [SearchHit("b", 5.0), SearchHit("c", 4.0)]
    scores = rrf_fuse([dense, sparse], k=60)
    # 'b' appears high in both lists -> should top the fused ranking
    assert max(scores, key=scores.get) == "b"


def test_hybrid_retrieves_relevant_chunk(sample_index):
    store, _ = sample_index
    plan = QueryPlan(rewritten_query="what is the refund window for enterprise customers")
    results = hybrid_search(plan, MetadataFilter(access_tags=["public"]), store)
    assert results
    top = results[0]
    assert isinstance(top, RetrievalResult)
    # the top hit should be from the refund policy and mention the fact
    joined = " ".join(str(top.payload.get(k) or "") for k in ("text", "table_markdown", "title")).lower()
    assert "refund" in joined or "30 days" in joined


def test_tier_filter_scopes_results(sample_index):
    store, _ = sample_index
    plan = QueryPlan(rewritten_query="sso enablement")
    flt = MetadataFilter(equals={"doc_type": ["runbook"]}, access_tags=["public"])
    results = hybrid_search(plan, flt, store)
    assert all(r.payload.get("doc_type") == "runbook" for r in results)


def test_fallback_relaxes_when_filter_starves(sample_index):
    store, _ = sample_index
    plan = QueryPlan(rewritten_query="refund window")
    # an impossible filter yields zero candidates -> fallback to access-scoped search
    impossible = MetadataFilter(equals={"doc_type": ["nonexistent"]}, access_tags=["public"])
    results = hybrid_search(
        plan, impossible, store, relax_fn=lambda: relax_filter(User(entitlements=["public"]))
    )
    assert results  # fallback recovered candidates
