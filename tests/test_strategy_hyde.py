"""Tests for the HyDE strategy (Step 30)."""

from __future__ import annotations

from ragnarok.providers import FakeLLM, reset_clients, set_role_client
from ragnarok.retrieval.preprocess import PreprocessResult, QueryPlan, SourcePlan
from ragnarok.strategies import get_strategy
from ragnarok.strategies.base import StrategyContext
from ragnarok.user import User


async def test_hyde_generates_hypothetical_and_retrieves(sample_index):
    store, features = sample_index
    # the HyDE LLM writes a hypothetical passage that mentions the answer vocabulary
    set_role_client("llm_small", FakeLLM(
        response="Enterprise customers on an annual contract have a 30-day refund window from invoice."))

    pre = PreprocessResult(plan=QueryPlan(rewritten_query="how long to get money back as an enterprise customer",
                                          intent="policy_lookup"),
                           source=SourcePlan(confidence=0.5))
    ctx = StrategyContext(query=pre.plan.rewritten_query, pre=pre, user=User(entitlements=["public"]),
                          store=store, features=features)

    res = await get_strategy("hyde").run(ctx)
    assert res.strategy == "hyde"
    assert "hypothetical" in res.notes and "30-day" in res.notes["hypothetical"]
    assert res.results  # retrieval succeeded using the hypothetical passage
    # the refund policy should be retrieved into the candidate set via the hypothetical passage
    joined = " ".join(
        str(r.payload.get(k) or "") for r in res.results for k in ("text", "table_markdown", "title")
    ).lower()
    assert "refund" in joined or "30 days" in joined
    reset_clients()
