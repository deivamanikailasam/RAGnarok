"""Per-query cost/latency/token budget with adaptive model routing (Step 28).

The single biggest cost driver in RAG is how many tokens the *large* model reads and writes. Most
questions (factoids, policy lookups, FAQs) are answerable by the small model over a tight context —
so paying for the large model on every query is waste. This module derives a per-query budget from
the query's intent + retrieval confidence and routes:

- simple, high-confidence  -> small generation model, fewer chunks, tighter token budget
- complex / low-confidence -> large generation model, more chunks, larger budget

This is a bounded, deterministic policy (no extra LLM call), tunable by config and gated by eval.
"""

from __future__ import annotations

from dataclasses import dataclass

from ragnarok.config import OptimizationCfg, get_settings
from ragnarok.retrieval.preprocess import QueryPlan


@dataclass
class QueryBudget:
    generation_role: str
    rerank_top_n: int
    context_budget_tokens: int
    max_output_tokens: int
    tier: str  # "simple" | "complex"
    reason: str = ""


def _is_simple(plan: QueryPlan, retrieval_confidence: float, cfg: OptimizationCfg) -> bool:
    single_hop = len(plan.sub_queries) <= 1
    simple_intent = plan.intent in cfg.simple_intents
    confident = retrieval_confidence >= 0.30  # top rerank score high enough to trust a small model
    return simple_intent and single_hop and confident


def budget_for(
    plan: QueryPlan, retrieval_confidence: float, *, cfg: OptimizationCfg | None = None
) -> QueryBudget:
    cfg = cfg or get_settings().optimization
    if not cfg.adaptive_routing:
        return QueryBudget(
            generation_role=cfg.complex_generation_role,
            rerank_top_n=cfg.complex_rerank_top_n,
            context_budget_tokens=cfg.complex_context_tokens,
            max_output_tokens=cfg.complex_max_output_tokens,
            tier="complex",
            reason="adaptive_routing disabled",
        )

    if _is_simple(plan, retrieval_confidence, cfg):
        return QueryBudget(
            generation_role=cfg.simple_generation_role,
            rerank_top_n=cfg.simple_rerank_top_n,
            context_budget_tokens=cfg.simple_context_tokens,
            max_output_tokens=cfg.simple_max_output_tokens,
            tier="simple",
            reason=f"intent={plan.intent}, single-hop, confidence={retrieval_confidence:.2f}",
        )
    return QueryBudget(
        generation_role=cfg.complex_generation_role,
        rerank_top_n=cfg.complex_rerank_top_n,
        context_budget_tokens=cfg.complex_context_tokens,
        max_output_tokens=cfg.complex_max_output_tokens,
        tier="complex",
        reason=f"intent={plan.intent}, sub_queries={len(plan.sub_queries)}, "
        f"confidence={retrieval_confidence:.2f}",
    )
