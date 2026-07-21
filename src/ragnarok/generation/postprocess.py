"""Post-processing (Step 20).

A cheap small-LLM pass over the generated answer: resolve [i] markers to real sources, extract atomic
claims each tied to supporting chunks (feeds the grounding gate, Step 21), format for the channel, and
suggest follow-ups. Doing these on the large model would cost ~10x the tokens for no quality gain.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ragnarok.generation.context import AssembledContext, Citation
from ragnarok.prompts import prompts
from ragnarok.structured import generate_structured


class GroundedClaim(BaseModel):
    text: str
    cite: list[int] = Field(default_factory=list)  # context [i] indices supporting the claim


class PostProcessed(BaseModel):
    answer: str
    claims: list[GroundedClaim] = Field(default_factory=list)
    followups: list[str] = Field(default_factory=list, max_length=3)
    self_reported_confidence: float = 0.5
    resolved_citations: list[Citation] = Field(default_factory=list)

    def cited_indices(self) -> set[int]:
        return {i for c in self.claims for i in c.cite}


async def postprocess(question: str, answer: str, ctx: AssembledContext) -> PostProcessed:
    msgs = prompts().render(
        "post_processor", "latest", question=question, answer=answer, context=ctx.text
    )
    result = await generate_structured("llm_small", msgs, PostProcessed)
    # resolve [i] -> real Citation objects from the assembled context
    by_index = {c.index: c for c in ctx.citations}
    used = sorted(result.cited_indices())
    result.resolved_citations = [by_index[i] for i in used if i in by_index]
    if not result.answer:
        result.answer = answer
    return result
