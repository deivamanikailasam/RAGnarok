"""Tests for observability: metrics, cost, tracing (Step 26)."""

from __future__ import annotations

import json

from ragnarok.observability import metrics
from ragnarok.observability.cost import cost_of
from ragnarok.observability.trace import record_usage, span, trace
from ragnarok.pipeline import answer
from ragnarok.providers import FakeLLM, Usage, reset_clients, set_role_client
from ragnarok.user import User


def test_cost_of_scales_with_tokens():
    cheap = cost_of("qwen2.5:7b-instruct", Usage(1000, 1000))
    dear = cost_of("gpt-4o", Usage(1000, 1000))
    assert dear > cheap > 0


def test_trace_accumulates_tokens_and_cost():
    metrics.collector().reset()
    with trace("ask", trace_id="t1") as tr:
        with span("generate"):
            record_usage("generate", "llm_large", "qwen2.5:32b-instruct", Usage(500, 200))
    assert tr.total_tokens == 700
    assert tr.total_cost > 0
    snap = metrics.collector().snapshot()
    assert any("tokens_total" in k for k in snap)


async def test_pipeline_emits_metrics(sample_index):
    store, features = sample_index
    metrics.collector().reset()

    def small(messages, schema):
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

    set_role_client("llm_small", FakeLLM(handler=small))
    set_role_client("llm_large", FakeLLM(response="Enterprise customers have a 30-day refund window [1]."))
    result = await answer("enterprise refund?", User(entitlements=["public"]), store=store, features=features)

    snap = metrics.collector().snapshot()
    assert snap.get("answers_total", 0) >= 1
    assert any("grounding_score" in k for k in snap)  # grounding recorded
    assert result.trace_id
    reset_clients()
