"""Tests for cross-encoder reranking (Step 17)."""

from __future__ import annotations

from ragnarok.retrieval.hybrid import RetrievalResult, hybrid_search
from ragnarok.retrieval.preprocess import QueryPlan
from ragnarok.retrieval.rerank import rerank
from ragnarok.stores.vector import MetadataFilter


def _r(cid, text):
    return RetrievalResult(chunk_id=cid, payload={"text": text}, fused_score=0.5)


def test_rerank_orders_by_query_relevance():
    results = [
        _r("irrelevant", "kubernetes pod autoscaling metrics"),
        _r("relevant", "the enterprise refund window is 30 days from invoice"),
    ]
    ranked = rerank("how long is the enterprise refund window", results)
    assert ranked[0].chunk_id == "relevant"
    assert ranked[0].rerank_score >= ranked[-1].rerank_score


def test_rerank_narrows_to_top_n():
    from ragnarok.config import RetrievalCfg

    results = [_r(str(i), f"refund window text number {i}") for i in range(20)]
    ranked = rerank("refund window", results, cfg=RetrievalCfg(rerank_top_n=5, min_rerank_score=0.0))
    assert len(ranked) == 5


def test_rerank_end_to_end_keeps_relevant_on_top(sample_index):
    store, _ = sample_index
    plan = QueryPlan(rewritten_query="what is the refund window for enterprise customers")
    fused = hybrid_search(plan, MetadataFilter(access_tags=["public"]), store)
    ranked = rerank(plan.rewritten_query, fused)
    assert ranked
    top_text = " ".join(str(ranked[0].payload.get(k) or "") for k in ("text", "table_markdown"))
    assert "refund" in top_text.lower() or "30 days" in top_text.lower()
    assert len(ranked) <= 8  # narrowed
