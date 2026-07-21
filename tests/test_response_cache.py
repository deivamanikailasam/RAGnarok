"""Tests for the response cache (Step 27)."""

from __future__ import annotations

import json

import ragnarok.cache as cache_mod
from ragnarok.observability import metrics
from ragnarok.pipeline import answer
from ragnarok.providers import FakeLLM, reset_clients, set_role_client
from ragnarok.user import User


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
    return "ok"


async def test_identical_query_served_from_cache(sample_index):
    store, features = sample_index
    cache_mod.get_cache.cache_clear()
    metrics.collector().reset()

    # count generation calls on whichever model serves them (adaptive routing may pick either)
    gen_calls = {"n": 0}

    def handler(messages, schema):
        if "numbered context" in messages[0]["content"].lower():
            gen_calls["n"] += 1
            return "Enterprise customers have a 30-day refund window [1]."
        return _small(messages, schema)

    set_role_client("llm_small", FakeLLM(handler=handler))
    set_role_client("llm_large", FakeLLM(handler=handler))

    user = User(entitlements=["public"])
    r1 = await answer("What is the enterprise refund window?", user, store=store, features=features)
    r2 = await answer("what is the ENTERPRISE   refund window?", user, store=store, features=features)

    assert r1.grounded and r1.text == r2.text
    assert r2.cache_hit is True           # normalized query hits the exact cache
    assert gen_calls["n"] == 1            # generator not called the second time
    assert r2.citations == r1.citations   # citations round-trip through the cache
    reset_clients()
    cache_mod.get_cache.cache_clear()


async def test_cache_scoped_by_entitlements(sample_index):
    store, features = sample_index
    cache_mod.get_cache.cache_clear()
    set_role_client("llm_small", FakeLLM(handler=_small))
    set_role_client("llm_large", FakeLLM(response="Enterprise customers have a 30-day refund window [1]."))

    await answer("enterprise refund window?", User(entitlements=["public"]), store=store, features=features)
    # a different entitlement set must NOT hit the same cache entry (isolation/security)
    r = await answer("enterprise refund window?", User(entitlements=["secret"]), store=store, features=features)
    assert r.cache_hit is False
    reset_clients()
    cache_mod.get_cache.cache_clear()
