"""HyDE — Hypothetical Document Embeddings (Step 30).

Gao et al. 2022 (arXiv:2212.10496). A short question often embeds far from the passages that answer
it. HyDE asks a small LLM to write a *hypothetical answer passage*, then embeds THAT (it sits closer
to real passages in vector space) and retrieves. We keep the original query as a hedge and reuse the
full hybrid+rerank pipeline. Best on sparse/technical corpora where questions and answers use
different vocabulary.
"""

from __future__ import annotations

from ragnarok import resilience
from ragnarok.prompts import prompts
from ragnarok.retrieval.orchestrator import retrieve
from ragnarok.retrieval.preprocess import PreprocessResult
from ragnarok.strategies import register
from ragnarok.strategies.base import StrategyContext, StrategyResult


@register
class HydeStrategy:
    name = "hyde"

    async def run(self, ctx: StrategyContext) -> StrategyResult:
        msgs = prompts().render("hyde", "latest", question=ctx.pre.plan.rewritten_query)
        resp = await resilience.call("llm_small", msgs, stage="hyde")
        hypothetical = resp.content.strip()

        # retrieve using the hypothetical passage AND the original query (hedge)
        plan2 = ctx.pre.plan.model_copy(
            update={"sub_queries": [hypothetical, ctx.pre.plan.rewritten_query]}
        )
        pre2 = PreprocessResult(plan=plan2, source=ctx.pre.source)
        results = retrieve(pre2, ctx.user, ctx.store, ctx.features, collection=ctx.collection)
        return StrategyResult(results=results, strategy=self.name,
                              notes={"hypothetical": hypothetical})
