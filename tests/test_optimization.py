"""Tests for the adaptive optimization layer (Step 28)."""

from __future__ import annotations

import json

import ragnarok.cache as cache_mod
from ragnarok.config import OptimizationCfg
from ragnarok.optimization.budget import budget_for
from ragnarok.optimization.semantic_cache import SemanticResponseCache, reset_semantic_cache
from ragnarok.pipeline import answer
from ragnarok.providers import FakeLLM, reset_clients, set_role_client
from ragnarok.retrieval.preprocess import QueryPlan
from ragnarok.user import User

# --- adaptive routing / budgets ---


def test_simple_query_routes_to_small_model():
    plan = QueryPlan(rewritten_query="enterprise refund window", intent="policy_lookup")
    b = budget_for(plan, retrieval_confidence=0.6, cfg=OptimizationCfg())
    assert b.generation_role == "llm_small"
    assert b.tier == "simple"
    assert b.rerank_top_n == 4


def test_complex_query_routes_to_large_model():
    plan = QueryPlan(rewritten_query="compare enterprise vs pro refunds and SSO",
                     intent="comparison", sub_queries=["a", "b"])
    b = budget_for(plan, retrieval_confidence=0.6, cfg=OptimizationCfg())
    assert b.generation_role == "llm_large"
    assert b.tier == "complex"
    assert b.context_budget_tokens == 3500


def test_low_confidence_simple_query_escalates_to_large():
    plan = QueryPlan(rewritten_query="obscure edge case", intent="policy_lookup")
    b = budget_for(plan, retrieval_confidence=0.05, cfg=OptimizationCfg())
    assert b.tier == "complex"  # not confident enough to trust the small model


def test_adaptive_routing_can_be_disabled():
    plan = QueryPlan(rewritten_query="q", intent="factoid")
    b = budget_for(plan, 0.9, cfg=OptimizationCfg(adaptive_routing=False))
    assert b.generation_role == "llm_large"


# --- semantic cache ---


def test_semantic_cache_hits_on_similar_embedding():
    c = SemanticResponseCache(threshold=0.95)
    c.store([1.0, 0.0, 0.0], ["public"], {"answer": "x"})
    assert c.lookup([0.99, 0.01, 0.0], ["public"]) == {"answer": "x"}  # near-identical
    assert c.lookup([0.0, 1.0, 0.0], ["public"]) is None  # orthogonal -> miss


def test_semantic_cache_is_scoped_by_entitlements():
    c = SemanticResponseCache(threshold=0.9)
    c.store([1.0, 0.0], ["public"], {"answer": "x"})
    assert c.lookup([1.0, 0.0], ["secret"]) is None  # different scope never served


# --- end-to-end through the pipeline ---


def _handler(messages, schema):
    system = messages[0]["content"].lower()
    if "rewrite user questions" in system:
        return json.dumps({"intent": "policy_lookup", "rewritten_query": "enterprise refund window",
                           "needs_retrieval": True})
    if "subset of the knowledge base" in system:
        return json.dumps({"equals": {}, "confidence": 0.5})
    if "post-process" in system:
        return json.dumps({"answer": "Enterprise customers have a 30-day refund window [1].",
                           "claims": [{"text": "enterprise refund window is 30 days", "cite": [1]}]})
    if "numbered context" in system:  # the answer generator
        return "Enterprise customers have a 30-day refund window [1]."
    return "ok"


async def test_pipeline_uses_small_model_for_simple_query(sample_index):
    store, features = sample_index
    cache_mod.get_cache.cache_clear()
    reset_semantic_cache()
    set_role_client("llm_small", FakeLLM(handler=_handler))
    large = FakeLLM(response="SHOULD-NOT-BE-CALLED")
    set_role_client("llm_large", large)

    result = await answer("what is the enterprise refund window?", User(entitlements=["public"]),
                          store=store, features=features)
    assert result.grounded
    assert large.calls == []  # simple query routed to the small model; large never called
    reset_clients()
    reset_semantic_cache()
    cache_mod.get_cache.cache_clear()


async def test_pipeline_semantic_cache_serves_paraphrase(sample_index):
    store, features = sample_index
    cache_mod.get_cache.cache_clear()
    reset_semantic_cache()
    # the deterministic test embedder is crude; use a lower threshold so the mechanism is exercised
    import ragnarok.optimization.semantic_cache as sc

    sc._cache = SemanticResponseCache(threshold=0.6)

    gen_calls = {"n": 0}

    def handler(messages, schema):
        if "numbered context" in messages[0]["content"].lower():
            gen_calls["n"] += 1
        return _handler(messages, schema)

    set_role_client("llm_small", FakeLLM(handler=handler))
    set_role_client("llm_large", FakeLLM(response="x"))

    user = User(entitlements=["public"])
    await answer("what is the enterprise refund window?", user, store=store, features=features)
    # a paraphrase with overlapping vocabulary should hit the semantic cache (no new generation)
    r2 = await answer("enterprise refund window duration?", user, store=store, features=features)
    assert r2.cache_hit is True
    assert gen_calls["n"] == 1  # generator ran only for the first query
    reset_clients()
    reset_semantic_cache()
    cache_mod.get_cache.cache_clear()
