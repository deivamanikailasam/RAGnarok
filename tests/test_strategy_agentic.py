"""Tests for Agentic RAG: tools, memory, ReAct loop, strategy (Step 39)."""

from __future__ import annotations

import json

from ragnarok.agent.memory import LongTermMemory, ShortTermMemory
from ragnarok.agent.tools import CalculatorTool, build_registry, register_tool
from ragnarok.providers import FakeLLM, reset_clients, set_role_client
from ragnarok.retrieval.preprocess import PreprocessResult, QueryPlan, SourcePlan
from ragnarok.strategies import get_strategy
from ragnarok.strategies.base import StrategyContext
from ragnarok.user import User


def _ctx(store, features, query="enterprise refund window"):
    pre = PreprocessResult(plan=QueryPlan(rewritten_query=query, intent="other"),
                           source=SourcePlan(confidence=0.5))
    return StrategyContext(query=query, pre=pre, user=User(id="u1", entitlements=["public"]),
                           store=store, features=features)


async def test_calculator_tool_is_safe():
    tr = await CalculatorTool().run("14 * 30 / 2", ctx=None)
    assert "210" in tr.observation
    bad = await CalculatorTool().run("__import__('os').system('ls')", ctx=None)
    assert "error" in bad.observation.lower()


def test_long_term_memory_recall():
    m = LongTermMemory()
    m.remember("u1", "The customer prefers annual billing.")
    m.remember("u1", "Their region is EU.")
    assert any("annual" in f for f in m.recall("u1", "what billing do they use"))


def test_short_term_memory_render():
    s = ShortTermMemory()
    s.add("need policy", "retrieve", "refund", "found 30-day window")
    assert "Action: retrieve(refund)" in s.render()


def test_registry_supports_pluggable_tool_servers():
    class DummyServer:
        name = "search"
        description = "web search"

        async def run(self, tool_input, ctx):  # pragma: no cover - not invoked here
            from ragnarok.agent.tools import ToolResult
            return ToolResult(observation="ok")

    register_tool(DummyServer())
    assert "search" in build_registry().names()


async def test_agentic_loop_gathers_evidence_then_finishes(sample_index):
    store, features = sample_index
    reset_clients()

    # planner: first retrieve, then finish (bounded loop)
    calls = {"n": 0}

    def planner(messages, schema):
        calls["n"] += 1
        if calls["n"] == 1:
            return json.dumps({"thought": "find policy", "tool": "retrieve",
                               "tool_input": "enterprise refund window"})
        return json.dumps({"thought": "have enough", "tool": "finish", "tool_input": ""})

    set_role_client("llm_small", FakeLLM(handler=planner))
    res = await get_strategy("agentic").run(_ctx(store, features))
    assert res.strategy == "agentic"
    assert res.results  # evidence gathered via the retrieve tool
    assert res.notes["tool_calls"] >= 2  # retrieve + finish
    reset_clients()


async def test_agentic_falls_back_when_no_evidence(sample_index):
    store, features = sample_index
    reset_clients()
    # planner immediately finishes without retrieving -> hybrid fallback
    set_role_client("llm_small", FakeLLM(response=json.dumps(
        {"thought": "done", "tool": "finish", "tool_input": ""})))
    res = await get_strategy("agentic").run(_ctx(store, features))
    assert res.notes.get("fallback") == "hybrid"
    assert res.results
    reset_clients()
