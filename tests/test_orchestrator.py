"""Tests for the retrieval orchestrator + business signals (Step 18)."""

from __future__ import annotations

from ragnarok.retrieval.hybrid import RetrievalResult
from ragnarok.retrieval.orchestrator import apply_business_signals, retrieve
from ragnarok.retrieval.preprocess import PreprocessResult, QueryPlan, SourcePlan
from ragnarok.stores.features import DocumentFeatures, InMemoryFeatureStore
from ragnarok.user import User


def test_business_signals_demote_deprecated_on_equal_relevance():
    features = InMemoryFeatureStore()
    features.upsert_document(DocumentFeatures("official-doc", authority="official", freshness_days=10))
    features.upsert_document(DocumentFeatures("old-doc", authority="deprecated", freshness_days=1000))
    results = [
        RetrievalResult(chunk_id="c-old", payload={"doc_id": "old-doc"}, rerank_score=0.7),
        RetrievalResult(chunk_id="c-new", payload={"doc_id": "official-doc"}, rerank_score=0.7),
    ]
    ranked = apply_business_signals(results, features)
    assert ranked[0].doc_id == "official-doc"


def test_retrieve_end_to_end(sample_index):
    store, features = sample_index
    pre = PreprocessResult(
        plan=QueryPlan(rewritten_query="what is the refund window for enterprise customers"),
        source=SourcePlan(equals={"doc_type": ["policy"]}, confidence=0.8),
    )
    results = retrieve(pre, User(entitlements=["public"]), store, features)
    assert results
    assert all(r.final_score != 0 for r in results)
    top_text = " ".join(str(results[0].payload.get(k) or "") for k in ("text", "table_markdown"))
    assert "refund" in top_text.lower() or "30 days" in top_text.lower()


def test_retrieve_records_popularity(sample_index):
    store, features = sample_index
    pre = PreprocessResult(plan=QueryPlan(rewritten_query="sso saml enablement steps"))
    results = retrieve(pre, User(entitlements=["public"]), store, features)
    served_docs = {r.doc_id for r in results}
    online = features.get_online(list(served_docs))
    assert any(f.popularity > 0 for f in online.values())
