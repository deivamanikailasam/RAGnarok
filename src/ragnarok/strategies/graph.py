"""Graph RAG strategy (Step 34).

Microsoft GraphRAG (2024) and earlier KG-RAG. Retrieves by traversing a knowledge graph: match the
query's entities, expand to their neighborhood (N hops), and surface the chunks that mention them.
Strong on multi-hop / relationship questions ("which plans include SSO *and* have a 30-day refund?")
that a pure vector store answers poorly. Results are reranked against the real query to restore
precision; if the query names no known entities, it falls back to hybrid retrieval.
"""

from __future__ import annotations

from ragnarok.config import get_settings
from ragnarok.retrieval.hybrid import RetrievalResult
from ragnarok.retrieval.orchestrator import retrieve
from ragnarok.retrieval.rerank import rerank
from ragnarok.stores.factory import get_graph_store
from ragnarok.strategies import register
from ragnarok.strategies.base import StrategyContext, StrategyResult


@register
class GraphStrategy:
    name = "graph"

    async def run(self, ctx: StrategyContext) -> StrategyResult:
        graph = ctx.graph or get_graph_store()
        hops = get_settings().rag.graph_expand_hops
        hits = graph.query(ctx.pre.plan.rewritten_query, hops=hops)

        if not hits:  # no entities matched -> don't return empty; fall back to hybrid
            results = retrieve(ctx.pre, ctx.user, ctx.store, ctx.features, collection=ctx.collection)
            return StrategyResult(results, self.name, notes={"fallback": "hybrid", "reason": "no entities"})

        # access enforcement still applies (graph payloads carry access_tags)
        allowed = set(ctx.user.entitlements)
        results = [
            RetrievalResult(chunk_id=h.chunk_id, payload=h.payload, fused_score=h.score)
            for h in hits
            if set(h.payload.get("access_tags") or []) & allowed
        ]
        results = rerank(ctx.pre.plan.rewritten_query, results)
        return StrategyResult(results, self.name, notes={"graph": graph.stats(), "hops": hops})
