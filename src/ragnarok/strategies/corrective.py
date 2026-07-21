"""Corrective RAG — CRAG (Step 32).

Yan et al. 2024 (arXiv:2401.15884). A retrieval evaluator grades the retrieved chunks; the action
depends on the grade:
- **Correct** (top score >= min): use the results as-is.
- **Ambiguous** (weak): refine — rewrite the query, re-retrieve, and combine with the originals.
- **Incorrect** (very weak): discard and fall back to a knowledge-refinement search (an access-scoped
  relaxed search here; a web/local search tool plugs in via ``knowledge_search``).

Grading uses the cross-encoder rerank score (a real relevance signal already computed) — an LLM
grader can be swapped in. Prevents the generator from building on irrelevant context.
"""

from __future__ import annotations

from collections.abc import Callable

from ragnarok import resilience
from ragnarok.config import get_settings
from ragnarok.prompts import prompts
from ragnarok.retrieval.filters import relax_filter
from ragnarok.retrieval.hybrid import RetrievalResult, hybrid_search
from ragnarok.retrieval.orchestrator import retrieve
from ragnarok.retrieval.preprocess import PreprocessResult
from ragnarok.retrieval.rerank import rerank
from ragnarok.strategies import register
from ragnarok.strategies.base import StrategyContext, StrategyResult

# Optional external knowledge source (web/local tool). Default: none (fully local).
KnowledgeSearch = Callable[[str], list[RetrievalResult]]
_knowledge_search: KnowledgeSearch | None = None


def set_knowledge_search(fn: KnowledgeSearch | None) -> None:
    global _knowledge_search
    _knowledge_search = fn


def _top_score(results: list[RetrievalResult]) -> float:
    return results[0].rerank_score if results else 0.0


@register
class CorrectiveStrategy:
    name = "corrective"

    async def run(self, ctx: StrategyContext) -> StrategyResult:
        cfg = get_settings().rag
        results = retrieve(ctx.pre, ctx.user, ctx.store, ctx.features, collection=ctx.collection)
        grade = _top_score(results)

        if grade >= cfg.corrective_grade_min:
            return StrategyResult(results, self.name, notes={"action": "correct", "grade": grade})

        # refine: rewrite the query and re-retrieve
        msgs = prompts().render("crag_refine", "latest", question=ctx.pre.plan.rewritten_query)
        rewritten = (await resilience.call("llm_small", msgs, stage="crag")).content.strip()
        plan2 = ctx.pre.plan.model_copy(update={"rewritten_query": rewritten, "sub_queries": []})
        refined = retrieve(
            PreprocessResult(plan=plan2, source=ctx.pre.source),
            ctx.user, ctx.store, ctx.features, collection=ctx.collection,
        )

        combined = _dedupe(results + refined)
        combined = rerank(ctx.pre.plan.rewritten_query, combined)

        if _top_score(combined) >= cfg.corrective_grade_min:
            return StrategyResult(combined, self.name, notes={"action": "refine", "rewritten": rewritten})

        # incorrect: fall back to an access-scoped relaxed search + optional external knowledge
        relaxed = hybrid_search(
            plan2, relax_filter(ctx.user), ctx.store, relax_fn=lambda: relax_filter(ctx.user)
        )
        extra = _knowledge_search(rewritten) if _knowledge_search else []
        fallback = rerank(ctx.pre.plan.rewritten_query, _dedupe(combined + relaxed + extra))
        return StrategyResult(fallback, self.name,
                              notes={"action": "fallback", "used_external": bool(extra)})


def _dedupe(results: list[RetrievalResult]) -> list[RetrievalResult]:
    seen: set[str] = set()
    out = []
    for r in results:
        if r.chunk_id not in seen:
            seen.add(r.chunk_id)
            out.append(r)
    return out
