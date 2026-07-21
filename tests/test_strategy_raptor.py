"""Tests for RAPTOR hierarchical index + strategy (Step 37)."""

from __future__ import annotations

from ragnarok.ingestion.chunking import Chunk
from ragnarok.ingestion.raptor import build_raptor_chunks, build_raptor_index
from ragnarok.providers import FakeLLM, reset_clients, set_role_client
from ragnarok.retrieval.preprocess import PreprocessResult, QueryPlan, SourcePlan
from ragnarok.stores.vector import InMemoryVectorStore
from ragnarok.strategies import get_strategy
from ragnarok.strategies.base import StrategyContext
from ragnarok.user import User


def _leaf(cid, section, text):
    return Chunk(chunk_id=cid, doc_id="d", text=text, token_count=5, content_hash=cid,
                 metadata={"section": section, "title": "Policy", "access_tags": ["public"],
                           "doc_type": "policy"})


async def test_build_raptor_chunks_creates_levels():
    reset_clients()
    set_role_client("llm_large", FakeLLM(response="Summary: enterprise 30 day, consumer 14 day refunds."))
    leaves = [
        _leaf("a", "Enterprise", "enterprise annual refund window is 30 days"),
        _leaf("b", "Consumer", "consumer monthly refund window is 14 days"),
    ]
    summaries = await build_raptor_chunks(leaves, doc_id="d", title="Policy", levels=2)
    # one level-1 summary per section (2) + one level-2 rollup
    levels = sorted(c.metadata["level"] for c in summaries)
    assert levels == [1, 1, 2]
    assert all(c.chunk_type == "summary" for c in summaries)
    reset_clients()


def test_build_raptor_index_upserts_summary_nodes():
    reset_clients()
    set_role_client("llm_large", FakeLLM(response="Refund policy summary with 30 and 14 day windows."))
    store = InMemoryVectorStore()
    leaves = [_leaf("a", "Enterprise", "enterprise refund 30 days"),
              _leaf("b", "Consumer", "consumer refund 14 days")]
    added = build_raptor_index(leaves, doc_id="d", title="Policy", store=store, levels=2)
    assert added and store.count() == len(added)
    assert any(c.metadata.get("raptor") for c in added)
    reset_clients()


async def test_raptor_strategy_reports_levels(sample_index):
    store, features = sample_index
    pre = PreprocessResult(plan=QueryPlan(rewritten_query="refund window", intent="policy_lookup"),
                           source=SourcePlan(confidence=0.5))
    ctx = StrategyContext(query=pre.plan.rewritten_query, pre=pre, user=User(entitlements=["public"]),
                          store=store, features=features)
    res = await get_strategy("raptor").run(ctx)
    assert res.strategy == "raptor"
    assert "levels" in res.notes
