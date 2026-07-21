"""Self-RAG — self-reflective retrieval (Step 33).

Asai et al. 2023 (arXiv:2310.11511). Self-RAG reflects with special tokens: whether retrieval is
needed (Retrieve), whether each passage is relevant (IsRel), and whether the answer is supported
(IsSup). RAGnarok maps these to concrete stages:

- **Retrieve?** — the ``needs_retrieval`` gate (Step 14) already skips retrieval for chitchat.
- **IsRel** — this strategy filters retrieved chunks by a relevance reflection (rerank-score
  threshold by default; a per-chunk LLM grader plugs in), so the generator only sees relevant
  context.
- **IsSup** — the grounding gate (Step 21) verifies support downstream and abstains if weak.

Net effect: less irrelevant context (fewer tokens, less dilution) and lower hallucination risk.
"""

from __future__ import annotations

from ragnarok.config import get_settings
from ragnarok.retrieval.orchestrator import retrieve
from ragnarok.strategies import register
from ragnarok.strategies.base import StrategyContext, StrategyResult


@register
class SelfRagStrategy:
    name = "self_rag"

    async def run(self, ctx: StrategyContext) -> StrategyResult:
        min_rel = get_settings().rag.self_rag_relevance_min
        results = retrieve(ctx.pre, ctx.user, ctx.store, ctx.features, collection=ctx.collection)

        # IsRel reflection: keep only chunks judged relevant; never return empty (keep best as hedge).
        relevant = [r for r in results if r.rerank_score >= min_rel]
        kept = relevant or results[:1]
        return StrategyResult(
            kept, self.name,
            notes={"kept": len(kept), "dropped": len(results) - len(kept), "reflected": True},
        )
