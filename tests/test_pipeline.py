"""End-to-end online pipeline test (Step 23)."""

from __future__ import annotations

import json

from ragnarok.pipeline import answer
from ragnarok.providers import FakeLLM, reset_clients, set_role_client
from ragnarok.user import User


def _small_handler(messages, schema):
    """Route small-LLM calls to the right structured response by prompt content."""
    system = messages[0]["content"].lower()
    if "rewrite user questions" in system:  # query optimizer
        return json.dumps({"intent": "policy_lookup", "rewritten_query": "enterprise refund window",
                           "sub_queries": [], "expansions": [], "needs_retrieval": True})
    if "subset of the knowledge base" in system:  # source identifier
        return json.dumps({"equals": {"doc_type": ["policy"]}, "confidence": 0.8})
    if "post-process" in system:  # post-processor
        return json.dumps({
            "answer": "Enterprise customers have a 30-day refund window [1].",
            "claims": [{"text": "enterprise refund window is 30 days", "cite": [1]}],
            "followups": ["What about the Pro plan?"], "self_reported_confidence": 0.9,
        })
    return "ok"


async def test_end_to_end_answer_is_grounded_and_cited(sample_index):
    store, features = sample_index
    set_role_client("llm_small", FakeLLM(handler=_small_handler))
    set_role_client("llm_large", FakeLLM(response="Enterprise customers have a 30-day refund window [1]."))

    result = await answer("and for enterprise?", User(entitlements=["public"]),
                          store=store, features=features,
                          history=["what's the refund window?"],
                          facets={"doc_type": ["policy", "runbook"]})

    assert result.retrieved > 0
    assert result.grounded is True
    assert "30-day" in result.text
    assert result.citations  # resolved sources present
    assert result.trace_id


async def test_input_guardrail_blocks_pii(sample_index):
    store, features = sample_index
    result = await answer("refund my card 4111 1111 1111 1111", User(id="x"),
                          store=store, features=features)
    assert result.blocked is True


async def test_chitchat_skips_retrieval(sample_index):
    store, features = sample_index
    set_role_client("llm_small", FakeLLM(handler=lambda m, s: json.dumps(
        {"intent": "chitchat", "rewritten_query": "hi", "needs_retrieval": False})
        if s else "Hello! How can I help?"))
    result = await answer("hey there", User(), store=store, features=features)
    assert result.retrieved == 0
    assert not result.blocked
    reset_clients()
