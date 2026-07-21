"""Hybrid RAG — Vector + Graph (Step 35).

Sarmah et al. 2024, *HybridRAG* (arXiv:2408.04948). This is the reference diagram's "Hybrid RAG":
fuse **vector** retrieval (dense+sparse, Step 16) with **graph** retrieval (entity neighborhoods,
Step 34) by Reciprocal Rank Fusion, then rerank. Vector captures semantics; graph captures explicit
relationships — together they beat either alone on questions that mix both ("which enterprise
features touch SSO *and* billing?"). Distinct from `hybrid` (dense+sparse only).
"""

from __future__ import annotations

from collections import defaultdict

from ragnarok.config import get_settings
from ragnarok.retrieval.hybrid import RetrievalResult
from ragnarok.retrieval.orchestrator import retrieve
from ragnarok.retrieval.rerank import rerank
from ragnarok.stores.factory import get_graph_store
from ragnarok.strategies import register
from ragnarok.strategies.base import StrategyContext, StrategyResult


def _rrf(ranked_lists: list[list[RetrievalResult]], k: int = 60) -> list[RetrievalResult]:
    scores: dict[str, float] = defaultdict(float)
    payloads: dict[str, RetrievalResult] = {}
    for ranked in ranked_lists:
        for rank, r in enumerate(ranked):
            scores[r.chunk_id] += 1.0 / (k + rank)
            payloads.setdefault(r.chunk_id, r)
    fused = []
    for cid, score in scores.items():
        r = payloads[cid]
        fused.append(RetrievalResult(chunk_id=cid, payload=r.payload, fused_score=score))
    fused.sort(key=lambda r: r.fused_score, reverse=True)
    return fused


@register
class HybridGraphStrategy:
    name = "hybrid_graph"

    async def run(self, ctx: StrategyContext) -> StrategyResult:
        graph = ctx.graph or get_graph_store()
        hops = get_settings().rag.graph_expand_hops

        vector_results = retrieve(ctx.pre, ctx.user, ctx.store, ctx.features, collection=ctx.collection)
        allowed = set(ctx.user.entitlements)
        graph_hits = [
            RetrievalResult(chunk_id=h.chunk_id, payload=h.payload, fused_score=h.score)
            for h in graph.query(ctx.pre.plan.rewritten_query, hops=hops)
            if set(h.payload.get("access_tags") or []) & allowed
        ]

        fused = _rrf([vector_results, graph_hits])
        reranked = rerank(ctx.pre.plan.rewritten_query, fused)
        return StrategyResult(
            reranked, self.name,
            notes={"vector": len(vector_results), "graph": len(graph_hits)},
        )
