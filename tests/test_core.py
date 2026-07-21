"""Tests for the Step 3 core: config, providers, resilience, prompts, structured output."""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from ragnarok import resilience
from ragnarok.config import get_settings, load_settings
from ragnarok.prompts import prompts
from ragnarok.providers import FakeLLM, reset_clients, role, set_role_client
from ragnarok.structured import generate_structured


def test_settings_load_and_env_interpolation(monkeypatch):
    monkeypatch.setenv("LLM_LARGE_BASE_URL", "http://example:9/v1")
    s = load_settings()  # reads settings.example.yaml
    assert s.models.llm_large.base_url == "http://example:9/v1"
    assert s.retrieval.rerank_top_n == 8
    assert 0.0 <= s.guardrails.output.grounding_min <= 1.0


def test_get_settings_singleton():
    assert get_settings() is get_settings()


def test_prompt_registry_renders_messages():
    msgs = prompts().render(
        "query_optimizer", "latest", query="and for enterprise?", chat_history=["refund window?"]
    )
    assert msgs[0]["role"] == "system"
    assert msgs[-1]["role"] == "user"
    # few-shot examples present as alternating user/assistant turns
    assert any(m["role"] == "assistant" for m in msgs)
    assert prompts().label("query_optimizer").startswith("query_optimizer@v")


class _Plan(BaseModel):
    intent: str
    needs_retrieval: bool


async def test_structured_output_with_fake_llm():
    reset_clients()
    set_role_client(
        "llm_small",
        FakeLLM(response=json.dumps({"intent": "policy_lookup", "needs_retrieval": True})),
    )
    plan = await generate_structured("llm_small", [{"role": "user", "content": "x"}], _Plan)
    assert plan.intent == "policy_lookup"
    assert plan.needs_retrieval is True
    reset_clients()


async def test_structured_output_repairs_invalid_json():
    reset_clients()
    calls = {"n": 0}

    def handler(messages, schema):
        calls["n"] += 1
        if calls["n"] == 1:
            return "not json"
        return json.dumps({"intent": "howto", "needs_retrieval": False})

    set_role_client("llm_small", FakeLLM(handler=handler))
    plan = await generate_structured("llm_small", [{"role": "user", "content": "x"}], _Plan)
    assert plan.intent == "howto"
    assert calls["n"] == 2  # one repair round
    reset_clients()


async def test_resilience_falls_back_and_labels_degraded():
    reset_clients()

    class Boom:
        role_name = "llm_large"
        model = "boom"

        async def complete(self, messages, *, response_schema=None, **kw):
            raise TimeoutError("simulated")

        async def stream(self, messages, **kw):  # pragma: no cover
            raise TimeoutError

    set_role_client("llm_large", Boom())
    set_role_client("llm_small", FakeLLM(response="ok"))
    resp = await resilience.call(
        "llm_large", [{"role": "user", "content": "hi"}], fallback_role="llm_small", base_delay=0
    )
    assert resp.degraded is True
    assert resp.content == "ok"
    reset_clients()


async def test_resilience_raises_when_non_retryable_and_no_fallback():
    reset_clients()

    class Bad:
        role_name = "llm_small"
        model = "bad"

        async def complete(self, messages, *, response_schema=None, **kw):
            raise ValueError("bad request")

        async def stream(self, messages, **kw):  # pragma: no cover
            raise ValueError

    set_role_client("llm_small", Bad())
    with pytest.raises(ValueError):
        await resilience.call("llm_small", [{"role": "user", "content": "x"}], base_delay=0)
    reset_clients()
