"""Hybrid RAG strategy (dense + sparse + rerank + signals) — the default (Steps 16-18, 29).

This is RAGnarok's production default: it wraps the retrieval orchestrator (filter -> hybrid search
-> RRF fusion -> cross-encoder rerank -> business signals). "Hybrid" here means dense + sparse (BM25),
the standard IR sense. The vector+graph variant is `hybrid_graph` (Step 35).
"""

from __future__ import annotations

from ragnarok.retrieval.orchestrator import retrieve
from ragnarok.strategies import register
from ragnarok.strategies.base import StrategyContext, StrategyResult


@register
class HybridStrategy:
    name = "hybrid"

    async def run(self, ctx: StrategyContext) -> StrategyResult:
        results = retrieve(ctx.pre, ctx.user, ctx.store, ctx.features, collection=ctx.collection)
        return StrategyResult(results=results, strategy=self.name)
