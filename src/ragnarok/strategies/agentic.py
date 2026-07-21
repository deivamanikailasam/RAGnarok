"""Agentic RAG strategy (Step 39).

The reference diagram's Agentic RAG: an agent with memory (short/long term), planning (ReAct/CoT),
and tools/MCP servers (local data, search, cloud). RAGnarok implements a bounded ReAct loop
(`ragnarok.agent`) whose tools include retrieval, a calculator, and any registered tool servers. The
agent decides which tools to call and gathers evidence; that evidence flows into the SHARED generator,
so an agentic answer is still grounded, cited, guardrailed, and abstains when unsupported.

Evidence is reranked against the original question before generation. If the agent gathered nothing,
it falls back to a single hybrid retrieval so it never returns empty.
"""

from __future__ import annotations

from ragnarok.agent.react import run_react
from ragnarok.retrieval.orchestrator import retrieve
from ragnarok.retrieval.rerank import rerank
from ragnarok.strategies import register
from ragnarok.strategies.base import StrategyContext, StrategyResult


@register
class AgenticStrategy:
    name = "agentic"

    async def run(self, ctx: StrategyContext) -> StrategyResult:
        trace = await run_react(ctx)
        results = trace.results
        if not results:  # agent gathered nothing -> hybrid fallback
            results = retrieve(ctx.pre, ctx.user, ctx.store, ctx.features, collection=ctx.collection)
            return StrategyResult(results, self.name, notes={"steps": trace.steps, "fallback": "hybrid"})
        results = rerank(ctx.pre.plan.rewritten_query, results)
        return StrategyResult(results, self.name,
                              notes={"steps": trace.steps, "tool_calls": len(trace.steps)})
