"""Tests for the Adaptive RAG router (Step 38)."""

from __future__ import annotations

import json

from ragnarok.providers import FakeLLM, reset_clients, set_role_client
from ragnarok.retrieval.preprocess import PreprocessResult, QueryPlan, SourcePlan
from ragnarok.strategies import get_strategy
from ragnarok.strategies.base import StrategyContext
from ragnarok.user import User


def _ctx(store, features, *, intent, sub_queries=None):
    plan = QueryPlan(rewritten_query="q", intent=intent, sub_queries=sub_queries or [])
    pre = PreprocessResult(plan=plan, source=SourcePlan(confidence=0.5))
    return StrategyContext(query="q", pre=pre, user=User(entitlements=["public"]),
                           store=store, features=features)


async def test_factoid_routes_to_hybrid(sample_index):
    store, features = sample_index
    res = await get_strategy("adaptive").run(_ctx(store, features, intent="policy_lookup"))
    assert res.notes["adaptive_route"] == "hybrid"
    assert res.strategy == "adaptive:hybrid"


async def test_comparison_routes_to_fusion(sample_index):
    store, features = sample_index
    set_role_client("llm_small", FakeLLM(response=json.dumps({"queries": ["a", "b"]})))
    res = await get_strategy("adaptive").run(_ctx(store, features, intent="comparison"))
    assert res.notes["adaptive_route"] == "fusion"
    reset_clients()


async def test_multihop_by_subqueries_routes_to_fusion(sample_index):
    store, features = sample_index
    set_role_client("llm_small", FakeLLM(response=json.dumps({"queries": ["a", "b"]})))
    res = await get_strategy("adaptive").run(
        _ctx(store, features, intent="other", sub_queries=["x", "y"]))
    assert res.notes["adaptive_route"] == "fusion"
    reset_clients()


async def test_open_query_routes_to_hyde(sample_index):
    store, features = sample_index
    set_role_client("llm_small", FakeLLM(response="a hypothetical passage about the topic"))
    res = await get_strategy("adaptive").run(_ctx(store, features, intent="other"))
    assert res.notes["adaptive_route"] == "hyde"
    reset_clients()
