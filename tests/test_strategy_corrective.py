"""Tests for the Corrective RAG (CRAG) strategy (Step 32)."""

from __future__ import annotations

from ragnarok.config import Settings, reset_settings, set_settings
from ragnarok.providers import FakeLLM, reset_clients, set_role_client
from ragnarok.retrieval.preprocess import PreprocessResult, QueryPlan, SourcePlan
from ragnarok.strategies import get_strategy
from ragnarok.strategies.base import StrategyContext
from ragnarok.strategies.corrective import set_knowledge_search
from ragnarok.user import User


def _ctx(store, features, query="enterprise refund window"):
    pre = PreprocessResult(plan=QueryPlan(rewritten_query=query, intent="policy_lookup"),
                           source=SourcePlan(confidence=0.5))
    return StrategyContext(query=query, pre=pre, user=User(entitlements=["public"]),
                           store=store, features=features)


async def test_correct_grade_uses_results_as_is(sample_index):
    store, features = sample_index
    # a strong query needs no correction (grade above the low default min)
    res = await get_strategy("corrective").run(_ctx(store, features))
    assert res.notes["action"] == "correct"
    assert res.results


async def test_weak_grade_triggers_refine(sample_index):
    store, features = sample_index
    reset_clients()
    # force a high grade threshold so the initial retrieval is judged weak -> refine
    base = Settings.model_validate({"models": {"llm_large": {"model": "x"}, "llm_small": {"model": "y"}}})
    base.rag.corrective_grade_min = 999.0  # nothing passes -> always refine, then fallback
    set_settings(base)
    set_role_client("llm_small", FakeLLM(response="enterprise annual refund policy window"))

    res = await get_strategy("corrective").run(_ctx(store, features))
    assert res.notes["action"] in {"refine", "fallback"}  # correction path was taken
    reset_settings()
    reset_clients()


async def test_external_knowledge_hook_used_on_incorrect(sample_index):
    store, features = sample_index
    reset_clients()
    base = Settings.model_validate({"models": {"llm_large": {"model": "x"}, "llm_small": {"model": "y"}}})
    base.rag.corrective_grade_min = 999.0
    set_settings(base)
    set_role_client("llm_small", FakeLLM(response="totally unrelated aerospace query"))

    called = {"n": 0}

    def fake_web(q):
        called["n"] += 1
        return []

    set_knowledge_search(fake_web)
    res = await get_strategy("corrective").run(_ctx(store, features, "aerospace telemetry"))
    assert res.notes["action"] == "fallback"
    assert called["n"] == 1  # external knowledge search was consulted
    set_knowledge_search(None)
    reset_settings()
    reset_clients()
