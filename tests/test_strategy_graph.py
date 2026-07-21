"""Tests for the Graph RAG strategy + knowledge graph (Step 34)."""

from __future__ import annotations

from ragnarok.retrieval.preprocess import PreprocessResult, QueryPlan, SourcePlan
from ragnarok.stores.factory import get_graph_store
from ragnarok.stores.graph import KnowledgeGraph
from ragnarok.strategies import get_strategy
from ragnarok.strategies.base import StrategyContext
from ragnarok.user import User


def test_knowledge_graph_entity_and_cooccurrence():
    g = KnowledgeGraph()
    g.add_chunk("c1", {"entities": ["Enterprise", "SSO"], "access_tags": ["public"]})
    g.add_chunk("c2", {"entities": ["SSO", "SAML"], "access_tags": ["public"]})
    assert g.stats()["entities"] == 3
    # SSO co-occurs with Enterprise and SAML -> both are 1-hop neighbors
    assert "saml" in g.neighbors("sso", hops=1)
    assert "enterprise" in g.neighbors("sso", hops=1)
    # 2 hops from Enterprise reaches SAML via SSO
    assert "saml" in g.neighbors("enterprise", hops=2)


def test_graph_query_returns_chunks_mentioning_entities():
    g = KnowledgeGraph()
    g.add_chunk("c1", {"entities": ["Enterprise", "Refund"], "access_tags": ["public"], "text": "..."})
    g.add_chunk("c2", {"entities": ["SSO"], "access_tags": ["public"], "text": "..."})
    hits = g.query("what is the enterprise refund policy", hops=1)
    assert hits and hits[0].chunk_id == "c1"


async def test_graph_strategy_end_to_end(sample_index):
    store, features = sample_index
    graph = get_graph_store()  # populated by the fixture's ingest
    assert graph.stats()["entities"] > 0

    pre = PreprocessResult(plan=QueryPlan(rewritten_query="enterprise refund", intent="policy_lookup"),
                           source=SourcePlan(confidence=0.5))
    ctx = StrategyContext(query=pre.plan.rewritten_query, pre=pre, user=User(entitlements=["public"]),
                          store=store, features=features, graph=graph)
    res = await get_strategy("graph").run(ctx)
    assert res.strategy == "graph"
    assert res.results  # entity-linked chunks retrieved


async def test_graph_strategy_falls_back_when_no_entities(sample_index):
    store, features = sample_index
    pre = PreprocessResult(plan=QueryPlan(rewritten_query="zzz nonexistent xyzzy", intent="other"),
                           source=SourcePlan(confidence=0.5))
    ctx = StrategyContext(query=pre.plan.rewritten_query, pre=pre, user=User(entitlements=["public"]),
                          store=store, features=features, graph=get_graph_store())
    res = await get_strategy("graph").run(ctx)
    assert res.notes.get("fallback") == "hybrid"
