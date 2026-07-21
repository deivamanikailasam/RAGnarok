"""Tests for the RAG-Fusion strategy (Step 31)."""

from __future__ import annotations

import json

from ragnarok.providers import FakeLLM, reset_clients, set_role_client
from ragnarok.retrieval.preprocess import PreprocessResult, QueryPlan, SourcePlan
from ragnarok.strategies import get_strategy
from ragnarok.strategies.base import StrategyContext
from ragnarok.user import User


async def test_fusion_expands_queries_and_retrieves(sample_index):
    store, features = sample_index
    set_role_client("llm_small", FakeLLM(response=json.dumps({
        "queries": ["enterprise refund period", "how long refund enterprise annual", "refund window enterprise"]
    })))

    pre = PreprocessResult(plan=QueryPlan(rewritten_query="enterprise refund window", intent="policy_lookup"),
                           source=SourcePlan(confidence=0.5))
    ctx = StrategyContext(query=pre.plan.rewritten_query, pre=pre, user=User(entitlements=["public"]),
                          store=store, features=features)

    res = await get_strategy("fusion").run(ctx)
    assert res.strategy == "fusion"
    # original query kept first, variations appended, deduped
    assert res.notes["queries"][0] == "enterprise refund window"
    assert len(res.notes["queries"]) >= 3
    assert res.results
    reset_clients()
