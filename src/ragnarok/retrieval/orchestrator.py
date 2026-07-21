"""Retrieval orchestrator (Step 18): filter -> hybrid -> rerank -> business signals.

Composes the retrieval stages into one entrypoint used by generation (Step 20). Business signals
(freshness/authority/popularity from the feature store) are applied AFTER semantic reranking as a
bounded adjustment, so current/official sources win ties and deprecated content is demoted without
overriding relevance (Step 13).
"""

from __future__ import annotations

from ragnarok.config import RetrievalCfg, get_settings
from ragnarok.retrieval.filters import build_filter, relax_filter
from ragnarok.retrieval.hybrid import RetrievalResult, hybrid_search
from ragnarok.retrieval.preprocess import PreprocessResult
from ragnarok.retrieval.rerank import rerank
from ragnarok.retrieval.signals import business_score
from ragnarok.stores.features import FeatureStore
from ragnarok.stores.vector import VectorStore
from ragnarok.user import User


def apply_business_signals(
    results: list[RetrievalResult], features: FeatureStore
) -> list[RetrievalResult]:
    if not results:
        return results
    feats = features.get_online(list({r.doc_id for r in results}))
    for r in results:
        f = feats.get(r.doc_id)
        r.final_score = business_score(r.rerank_score, f) if f else r.rerank_score
    results.sort(key=lambda r: r.final_score, reverse=True)
    return results


def retrieve(
    pre: PreprocessResult,
    user: User,
    store: VectorStore,
    features: FeatureStore,
    *,
    cfg: RetrievalCfg | None = None,
    collection: str = "chunks",
) -> list[RetrievalResult]:
    cfg = cfg or get_settings().retrieval
    flt = build_filter(pre.source, user)
    fused = hybrid_search(
        pre.plan, flt, store, cfg=cfg, collection=collection,
        relax_fn=lambda: relax_filter(user),
    )
    ranked = rerank(pre.plan.rewritten_query, fused, cfg=cfg)
    final = apply_business_signals(ranked, features)
    # record usage for popularity (Step 11) — the docs that actually served
    for r in final:
        features.record_hit(r.doc_id)
    return final
