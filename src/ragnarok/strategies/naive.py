"""Naive RAG strategy (Step 29).

The canonical baseline: embed the query, dense top-k, generate. No sparse leg, no rerank, no signals.
Kept as a first-class strategy so the eval harness can quantify exactly how much hybrid + rerank
(Steps 16-18) buy over the baseline — but it is never the production default.
"""

from __future__ import annotations

from ragnarok.config import get_settings
from ragnarok.ingestion.embedding import get_embedding_client
from ragnarok.retrieval.filters import build_filter
from ragnarok.retrieval.hybrid import RetrievalResult
from ragnarok.strategies import register
from ragnarok.strategies.base import StrategyContext, StrategyResult


@register
class NaiveStrategy:
    name = "naive"

    async def run(self, ctx: StrategyContext) -> StrategyResult:
        cfg = get_settings().retrieval
        flt = build_filter(ctx.pre.source, ctx.user)
        dense = get_embedding_client().embed_texts([ctx.pre.plan.rewritten_query])[0].dense
        hits = ctx.store.search_dense(
            dense, filter=flt, limit=cfg.rerank_top_n, collection=ctx.collection
        )
        results = [
            RetrievalResult(chunk_id=h.id, payload=h.payload, dense_score=h.score,
                            fused_score=h.score, rerank_score=h.score, final_score=h.score)
            for h in hits
        ]
        return StrategyResult(results=results, strategy=self.name)
