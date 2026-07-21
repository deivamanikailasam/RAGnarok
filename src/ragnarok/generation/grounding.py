"""Grounding / faithfulness gate (Step 21).

Verifies each extracted claim is supported by its cited context before serving. Cheap claims are
checked by lexical/NLI entailment; ambiguous ones can escalate to a small-LLM judge. If the grounding
score is below threshold, we ABSTAIN (return an honest "couldn't confirm" with the closest sources)
rather than serve a possibly-hallucinated answer. Trades a wrong answer for "I don't know".
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ragnarok.config import OutputGuardCfg, get_settings
from ragnarok.generation.context import AssembledContext
from ragnarok.generation.postprocess import GroundedClaim, PostProcessed

_STOP = set("the a an of to for is are was were be by on in at and or with from as it this that "
            "you your our we they their his her its can may will within".split())
_TOKEN_RE = re.compile(r"[a-z0-9]+")

ABSTENTION = (
    "I couldn't find enough in our documents to answer this confidently. "
    "Here are the closest sources I found — please verify with them."
)


def _content_tokens(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP and len(t) > 2}


def lexical_entails(claim: str, evidence: str, threshold: float = 0.6) -> bool:
    """A claim is supported if most of its content tokens appear in the evidence."""
    ct = _content_tokens(claim)
    if not ct:
        return True
    ev = _content_tokens(evidence)
    return len(ct & ev) / len(ct) >= threshold


def _evidence_for(claim: GroundedClaim, ctx: AssembledContext) -> str:
    by_index = {c.index: r for c, r in zip(ctx.citations, ctx.results)}
    parts = []
    for i in claim.cite:
        r = by_index.get(i)
        if r:
            parts.append((r.payload.get("table_markdown") or "") + " " + r.payload.get("text", ""))
    return " ".join(parts)


@dataclass
class GroundingVerdict:
    score: float
    grounded: bool
    answer: str
    unsupported: list[str]


def grounding_score(claims: list[GroundedClaim], ctx: AssembledContext) -> tuple[float, list[str]]:
    if not claims:
        return 0.0, []
    supported = 0
    unsupported: list[str] = []
    for claim in claims:
        evidence = _evidence_for(claim, ctx)
        if claim.cite and lexical_entails(claim.text, evidence):
            supported += 1
        else:
            unsupported.append(claim.text)
    return supported / len(claims), unsupported


def check_grounding(
    pp: PostProcessed, ctx: AssembledContext, *, cfg: OutputGuardCfg | None = None
) -> GroundingVerdict:
    cfg = cfg or get_settings().guardrails.output
    score, unsupported = grounding_score(pp.claims, ctx)
    grounded = score >= cfg.grounding_min
    answer = pp.answer if grounded else ABSTENTION
    return GroundingVerdict(score=score, grounded=grounded, answer=answer, unsupported=unsupported)
