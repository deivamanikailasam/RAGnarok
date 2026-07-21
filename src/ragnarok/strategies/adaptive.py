"""Adaptive RAG — route by query complexity (Step 38).

Jeong et al. 2024 (arXiv:2403.14403). One retrieval strategy doesn't fit all queries: a factoid needs
a single hybrid lookup, a comparison needs multi-query fusion, a relationship question needs the
graph. Adaptive RAG classifies each query and delegates to the right sub-strategy — spending compute
in proportion to difficulty. This complements the Step 28 model-routing (which sizes the *generation*
model); here we size the *retrieval* method.

Classification is deterministic (no extra LLM) from signals already computed by the query optimizer
(Step 14): intent and hop count. The mapping is config-driven (`rag.adaptive_multistep_intents`).
"""

from __future__ import annotations

from ragnarok.config import get_settings
from ragnarok.strategies import get_strategy, register
from ragnarok.strategies.base import StrategyContext, StrategyResult


@register
class AdaptiveStrategy:
    name = "adaptive"

    def _route(self, ctx: StrategyContext) -> str:
        cfg = get_settings().rag
        plan = ctx.pre.plan
        multi_step = plan.intent in cfg.adaptive_multistep_intents or len(plan.sub_queries) > 1
        if multi_step:
            return "fusion"  # multi-step: broaden recall across reformulations
        if plan.intent in {"factoid", "policy_lookup", "faq"}:
            return "hybrid"  # single-step lookup
        return "hyde"  # semantic single-step for open/under-specified queries

    async def run(self, ctx: StrategyContext) -> StrategyResult:
        chosen = self._route(ctx)
        res = await get_strategy(chosen).run(ctx)
        res.notes = {**res.notes, "adaptive_route": chosen, "delegated": res.strategy}
        res.strategy = f"adaptive:{chosen}"
        return res
