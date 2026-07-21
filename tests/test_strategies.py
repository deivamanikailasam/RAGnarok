"""Tests for the RAG strategy framework: registry, naive, hybrid, pipeline dispatch (Step 29)."""

from __future__ import annotations

import json

from ragnarok.pipeline import answer
from ragnarok.providers import FakeLLM, reset_clients, set_role_client
from ragnarok.retrieval.preprocess import PreprocessResult, QueryPlan, SourcePlan
from ragnarok.strategies import available, get_strategy
from ragnarok.strategies.base import StrategyContext
from ragnarok.user import User


def _ctx(store, features, query="enterprise refund window"):
    pre = PreprocessResult(plan=QueryPlan(rewritten_query=query, intent="policy_lookup"),
                           source=SourcePlan(confidence=0.5))
    return StrategyContext(query=query, pre=pre, user=User(entitlements=["public"]),
                           store=store, features=features)


def test_registry_has_core_strategies():
    strategies = available()
    assert "hybrid" in strategies and "naive" in strategies


async def test_hybrid_strategy_returns_ranked_results(sample_index):
    store, features = sample_index
    res = await get_strategy("hybrid").run(_ctx(store, features))
    assert res.strategy == "hybrid"
    assert res.results
    assert res.results[0].rerank_score >= 0.0


async def test_naive_strategy_dense_only(sample_index):
    store, features = sample_index
    res = await get_strategy("naive").run(_ctx(store, features))
    assert res.strategy == "naive"
    assert res.results
    # naive uses the dense score as its final score (no rerank stage)
    assert res.results[0].dense_score == res.results[0].final_score


def _small(messages, schema):
    system = messages[0]["content"].lower()
    if "rewrite user questions" in system:
        return json.dumps({"intent": "policy_lookup", "rewritten_query": "enterprise refund window",
                           "needs_retrieval": True})
    if "subset of the knowledge base" in system:
        return json.dumps({"equals": {}, "confidence": 0.5})
    if "post-process" in system:
        return json.dumps({"answer": "Enterprise customers have a 30-day refund window [1].",
                           "claims": [{"text": "enterprise refund window is 30 days", "cite": [1]}]})
    if "numbered context" in system:
        return "Enterprise customers have a 30-day refund window [1]."
    return "ok"


async def test_pipeline_dispatches_to_named_strategy(sample_index):
    store, features = sample_index
    set_role_client("llm_small", FakeLLM(handler=_small))
    set_role_client("llm_large", FakeLLM(handler=_small))
    r = await answer("enterprise refund window?", User(entitlements=["public"]),
                     store=store, features=features, strategy="naive")
    assert r.retrieved > 0
    assert r.grounded
    reset_clients()
