"""Tests for context assembly (Step 19)."""

from __future__ import annotations

from ragnarok.config import GenerationCfg
from ragnarok.generation.context import build_context
from ragnarok.retrieval.hybrid import RetrievalResult


def _r(cid, text, score, **payload):
    return RetrievalResult(chunk_id=cid, payload={"text": text, "title": "T", "doc_id": "d", **payload},
                           rerank_score=score, final_score=score)


def test_numbered_citations_align_with_results():
    ctx = build_context([_r("a", "alpha", 0.9), _r("b", "beta", 0.8)])
    assert ctx.n_chunks == 2
    assert "[1]" in ctx.text and "[2]" in ctx.text
    assert len(ctx.citations) == 2
    assert ctx.citations[0].index == 1
    # citation i aligns with results[i-1]
    assert ctx.citations[0].chunk_id == ctx.results[0].chunk_id


def test_budget_limits_chunks():
    big = "word " * 500
    results = [_r(str(i), big, 1.0 - i * 0.01) for i in range(10)]
    ctx = build_context(results, cfg=None)
    # default budget 3500 tokens -> far fewer than 10 huge blocks
    assert ctx.n_chunks < 10


def test_tables_are_included_verbatim():
    ctx = build_context([_r("t", "desc", 0.9, table_markdown="| Tier | Refund |\n|---|---|\n| Ent | 30 |")])
    assert "| Tier | Refund |" in ctx.text  # exact cell values available to the generator


def test_attention_ordering_places_best_at_ends():
    results = [_r(str(i), f"chunk {i}", 1.0 - i * 0.1) for i in range(5)]
    ctx = build_context(results, cfg=GenerationCfg(max_context_chunks=5, context_budget_tokens=10000))
    # strongest (index 0 by score) should be first; second-strongest last
    assert ctx.results[0].chunk_id == "0"
    assert ctx.results[-1].chunk_id == "1"
