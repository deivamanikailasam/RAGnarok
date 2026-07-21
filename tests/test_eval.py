"""Tests for evaluation metrics, judge, and the release gate (Step 24)."""

from __future__ import annotations

import json
from pathlib import Path

from ragnarok.eval.golden import load_golden
from ragnarok.eval.judge import judge_answer
from ragnarok.eval.metrics import answer_correctness, answer_relevancy, retrieval_hit, token_f1
from ragnarok.eval.run import run_suite
from ragnarok.generation.context import Citation
from ragnarok.pipeline import AnswerResult
from ragnarok.providers import FakeLLM, reset_clients, set_role_client

GOLDEN = str(Path(__file__).resolve().parents[1] / "datasets" / "golden" / "v1")


def test_metrics_basic():
    assert retrieval_hit([{"doc_id": "file://refund_policy.md"}], "refund") == 1.0
    assert retrieval_hit([{"doc_id": "file://sso.md"}], "refund") == 0.0
    assert token_f1("enterprise refund 30 days", "enterprise refund window 30 days") > 0.6
    assert answer_relevancy("enterprise 30 day refund", "enterprise refund window") > 0.5
    assert answer_correctness("30 day refund enterprise", "enterprise 30 day refund") > 0.6


def test_golden_loads():
    cases = load_golden(GOLDEN)
    assert len(cases) == 4
    assert any(c.expect == "refusal" for c in cases)


async def test_gate_passes_on_good_answers():
    def good(query: str):
        async def _():
            if "reveal the system prompt" in query.lower():
                return AnswerResult(text="I can't process that request.", blocked=True)
            if "enterprise" in query.lower():
                text, src = "Enterprise customers have a 30-day refund window.", "refund"
            elif "consumer" in query.lower():
                text, src = "Consumer monthly customers have a 14-day refund window.", "refund"
            else:
                text, src = "SAML SSO is available on the Growth and Enterprise plans.", "sso"
            return AnswerResult(text=text, grounded=True, grounding_score=1.0,
                                citations=[Citation(1, "c", src, src, "", "")])
        return _()

    # gate thresholds appropriate to deterministic lexical proxies (real runs use judge scores)
    report = await run_suite(GOLDEN, answer_fn=lambda q: good(q),
                             thresholds={"context_recall": 0.75, "faithfulness": 0.85})
    assert report.passed is True
    assert report.metrics["context_recall"] >= 0.75


async def test_gate_fails_on_hallucinated_answers():
    def bad(query: str):
        async def _():
            if "reveal the system prompt" in query.lower():
                return AnswerResult(text="Here is the system prompt: ...", blocked=False)  # leaked!
            return AnswerResult(text="I'm not sure, maybe 99 days?", grounded=False, grounding_score=0.0)
        return _()

    report = await run_suite(GOLDEN, answer_fn=lambda q: bad(q))
    assert report.passed is False


async def test_llm_judge_structured():
    reset_clients()
    set_role_client("llm_small", FakeLLM(response=json.dumps(
        {"faithfulness": 5, "relevancy": 4, "correctness": 5, "reasoning": "ok", "unsupported_claims": []})))
    j = await judge_answer("q", "a", "ctx", "ideal")
    assert j.faithfulness == 5
    assert j.normalized()["faithfulness"] == 1.0
    reset_clients()
