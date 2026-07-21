"""RAG-Fusion — multi-query generation + Reciprocal Rank Fusion (Step 31).

Raudaschl 2023. Under-specified queries retrieve poorly from any single phrasing. RAG-Fusion asks a
small LLM for N diverse reformulations, retrieves each, and fuses the ranked lists with RRF (which
the hybrid retriever already does across sub-queries, Step 16). More robust recall than one query;
the reranker (Step 17) then restores precision.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ragnarok.config import get_settings
from ragnarok.prompts import prompts
from ragnarok.retrieval.orchestrator import retrieve
from ragnarok.retrieval.preprocess import PreprocessResult
from ragnarok.strategies import register
from ragnarok.strategies.base import StrategyContext, StrategyResult
from ragnarok.structured import generate_structured


class QueryVariations(BaseModel):
    queries: list[str] = Field(default_factory=list, max_length=8)


@register
class FusionStrategy:
    name = "fusion"

    async def run(self, ctx: StrategyContext) -> StrategyResult:
        n = get_settings().rag.fusion_num_queries
        original = ctx.pre.plan.rewritten_query
        msgs = prompts().render("query_fusion", "latest", question=original, n=n)
        variations = (await generate_structured("llm_small", msgs, QueryVariations)).queries
        queries = list(dict.fromkeys([original, *variations]))  # dedupe, keep original first

        plan2 = ctx.pre.plan.model_copy(update={"sub_queries": queries})
        pre2 = PreprocessResult(plan=plan2, source=ctx.pre.source)
        results = retrieve(pre2, ctx.user, ctx.store, ctx.features, collection=ctx.collection)
        return StrategyResult(results=results, strategy=self.name, notes={"queries": queries})
