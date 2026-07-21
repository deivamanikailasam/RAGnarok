"""Tests for generation + post-processing (Step 20)."""

from __future__ import annotations

import json

from ragnarok.generation.context import build_context
from ragnarok.generation.generate import generate_answer, generate_stream
from ragnarok.generation.postprocess import postprocess
from ragnarok.providers import FakeLLM, reset_clients, set_role_client
from ragnarok.retrieval.hybrid import RetrievalResult


def _ctx():
    r = RetrievalResult(chunk_id="c1", payload={"text": "Enterprise annual refund window is 30 days.",
                                                "title": "Refund Policy", "section": "Enterprise",
                                                "doc_id": "gdoc://refund", "uri": "gdoc://refund"},
                        rerank_score=0.9, final_score=0.9)
    return build_context([r])


async def test_generate_answer_uses_context():
    reset_clients()
    set_role_client("llm_large", FakeLLM(response="Enterprise customers have a 30-day refund window [1]."))
    ans = await generate_answer("enterprise refund window?", _ctx())
    assert "30-day" in ans and "[1]" in ans
    reset_clients()


async def test_generate_stream_yields_tokens():
    reset_clients()
    set_role_client("llm_large", FakeLLM(response="Enterprise refund window is 30 days [1]"))
    chunks = [c async for c in generate_stream("q", _ctx())]
    assert "".join(chunks).strip().startswith("Enterprise")
    reset_clients()


async def test_postprocess_extracts_claims_and_resolves_citations():
    reset_clients()
    pp = {
        "answer": "Enterprise customers have a 30-day refund window [1].",
        "claims": [{"text": "Enterprise refund window is 30 days", "cite": [1]}],
        "followups": ["What about the Pro plan?"],
        "self_reported_confidence": 0.9,
    }
    set_role_client("llm_small", FakeLLM(response=json.dumps(pp)))
    ctx = _ctx()
    result = await postprocess("enterprise refund window?", pp["answer"], ctx)
    assert result.claims[0].cite == [1]
    assert result.cited_indices() == {1}
    # [1] resolves to the real source citation
    assert result.resolved_citations
    assert result.resolved_citations[0].title == "Refund Policy"
    reset_clients()
