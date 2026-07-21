"""Tests for query pre-processing (Step 14)."""

from __future__ import annotations

import json

import ragnarok.cache as cache_mod
from ragnarok.providers import FakeLLM, reset_clients, set_role_client
from ragnarok.retrieval.preprocess import optimize_query, preprocess
from ragnarok.user import User


def _plan(**kw):
    base = {"intent": "policy_lookup", "rewritten_query": "enterprise refund window",
            "sub_queries": [], "expansions": ["SLA"], "needs_retrieval": True}
    base.update(kw)
    return json.dumps(base)


async def test_optimizer_rewrites_and_caches():
    reset_clients()
    cache_mod.get_cache.cache_clear()
    calls = {"n": 0}

    def handler(messages, schema):
        calls["n"] += 1
        return _plan(rewritten_query="What is the refund window for enterprise customers?")

    set_role_client("llm_small", FakeLLM(handler=handler))
    p1 = await optimize_query("and for enterprise?", ["What's the refund window?"])
    p2 = await optimize_query("and for enterprise?", ["What's the refund window?"])
    assert "enterprise" in p1.rewritten_query.lower()
    assert calls["n"] == 1  # second call served from cache
    reset_clients()
    cache_mod.get_cache.cache_clear()


async def test_preprocess_runs_both_agents_when_retrieval_needed():
    reset_clients()
    cache_mod.get_cache.cache_clear()

    def handler(messages, schema):
        text = json.dumps(messages)
        if "subset of the knowledge base" in text or "corpus facets" in text:
            return json.dumps({"equals": {"audience": ["enterprise"]}, "confidence": 0.8})
        return _plan()

    set_role_client("llm_small", FakeLLM(handler=handler))
    result = await preprocess("enterprise refunds?", [], User(entitlements=["public"]),
                              facets={"audience": ["enterprise", "consumer"]})
    assert not result.skip_retrieval
    assert result.source is not None
    assert result.source.equals["audience"] == ["enterprise"]
    reset_clients()
    cache_mod.get_cache.cache_clear()


async def test_chitchat_gate_skips_retrieval():
    reset_clients()
    cache_mod.get_cache.cache_clear()
    set_role_client("llm_small", FakeLLM(response=_plan(intent="chitchat", needs_retrieval=False,
                                                        rewritten_query="hi")))
    result = await preprocess("hey there", [], User())
    assert result.skip_retrieval is True
    assert result.source is None
    reset_clients()
    cache_mod.get_cache.cache_clear()
