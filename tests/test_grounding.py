"""Tests for the grounding gate (Step 21)."""

from __future__ import annotations

from ragnarok.config import OutputGuardCfg
from ragnarok.generation.context import build_context
from ragnarok.generation.grounding import ABSTENTION, check_grounding, lexical_entails
from ragnarok.generation.postprocess import GroundedClaim, PostProcessed
from ragnarok.retrieval.hybrid import RetrievalResult


def _ctx():
    r = RetrievalResult(chunk_id="c1", payload={"text": "Enterprise annual refund window is 30 days from invoice.",
                                                "title": "Refund", "section": "Enterprise", "doc_id": "d"},
                        rerank_score=0.9, final_score=0.9)
    return build_context([r])


def test_lexical_entailment():
    assert lexical_entails("enterprise refund window is 30 days",
                           "The enterprise annual refund window is 30 days from invoice.")
    assert not lexical_entails("consumers get a 90 day refund on hardware",
                               "The enterprise annual refund window is 30 days.")


def test_grounded_answer_is_served():
    ctx = _ctx()
    pp = PostProcessed(answer="Enterprise customers have a 30-day refund window [1].",
                       claims=[GroundedClaim(text="enterprise refund window is 30 days", cite=[1])])
    verdict = check_grounding(pp, ctx, cfg=OutputGuardCfg(grounding_min=0.6))
    assert verdict.grounded
    assert "30-day" in verdict.answer


def test_ungrounded_answer_abstains():
    ctx = _ctx()
    # a claim not supported by the cited context
    pp = PostProcessed(answer="Enterprise customers get a full 90-day refund on any hardware [1].",
                       claims=[GroundedClaim(text="enterprise customers get a 90 day hardware refund", cite=[1])])
    verdict = check_grounding(pp, ctx, cfg=OutputGuardCfg(grounding_min=0.6))
    assert not verdict.grounded
    assert verdict.answer == ABSTENTION
    assert verdict.unsupported


def test_uncited_claims_are_unsupported():
    ctx = _ctx()
    pp = PostProcessed(answer="Refunds are instant.", claims=[GroundedClaim(text="refunds are instant", cite=[])])
    verdict = check_grounding(pp, ctx, cfg=OutputGuardCfg(grounding_min=0.6))
    assert not verdict.grounded  # no citation -> not grounded
