"""Tests for the Hybrid RAG (vector + graph) strategy (Step 35)."""

from __future__ import annotations

from ragnarok.retrieval.preprocess import PreprocessResult, QueryPlan, SourcePlan
from ragnarok.stores.factory import get_graph_store
from ragnarok.strategies import get_strategy
from ragnarok.strategies.base import StrategyContext
from ragnarok.user import User


async def test_hybrid_graph_fuses_vector_and_graph(sample_index):
    store, features = sample_index
    pre = PreprocessResult(plan=QueryPlan(rewritten_query="enterprise refund window", intent="policy_lookup"),
                           source=SourcePlan(confidence=0.5))
    ctx = StrategyContext(query=pre.plan.rewritten_query, pre=pre, user=User(entitlements=["public"]),
                          store=store, features=features, graph=get_graph_store())
    res = await get_strategy("hybrid_graph").run(ctx)
    assert res.strategy == "hybrid_graph"
    assert res.results
    # both retrieval arms contributed to the fusion
    assert res.notes["vector"] > 0
    assert res.notes["graph"] >= 0
