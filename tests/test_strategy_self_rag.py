"""Tests for the Self-RAG strategy (Step 33)."""

from __future__ import annotations

from ragnarok.config import Settings, reset_settings, set_settings
from ragnarok.retrieval.preprocess import PreprocessResult, QueryPlan, SourcePlan
from ragnarok.strategies import get_strategy
from ragnarok.strategies.base import StrategyContext
from ragnarok.user import User


def _ctx(store, features, query="enterprise refund window"):
    pre = PreprocessResult(plan=QueryPlan(rewritten_query=query, intent="policy_lookup"),
                           source=SourcePlan(confidence=0.5))
    return StrategyContext(query=query, pre=pre, user=User(entitlements=["public"]),
                           store=store, features=features)


async def test_self_rag_filters_irrelevant_chunks(sample_index):
    store, features = sample_index
    # high relevance threshold -> most chunks dropped, best kept as a hedge
    base = Settings.model_validate({"models": {"llm_large": {"model": "x"}, "llm_small": {"model": "y"}}})
    base.rag.self_rag_relevance_min = 999.0
    set_settings(base)

    res = await get_strategy("self_rag").run(_ctx(store, features))
    assert res.strategy == "self_rag"
    assert res.notes["reflected"] is True
    assert len(res.results) == 1  # nothing passes the (extreme) threshold -> keep best only
    reset_settings()


async def test_self_rag_keeps_relevant_chunks(sample_index):
    store, features = sample_index
    base = Settings.model_validate({"models": {"llm_large": {"model": "x"}, "llm_small": {"model": "y"}}})
    base.rag.self_rag_relevance_min = 0.0  # everything relevant
    set_settings(base)

    res = await get_strategy("self_rag").run(_ctx(store, features))
    assert res.notes["dropped"] == 0
    assert res.results
    reset_settings()
