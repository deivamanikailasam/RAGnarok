"""Online answer pipeline (Step 23) — the agentic graph tying every stage together.

guard-in -> preprocess -> retrieve -> context -> generate -> post-process -> grounding gate ->
guard-out. This is the single implementation shared by Slack, the API, and the CLI so behavior never
drifts across surfaces. Observability (Step 26) wraps each stage; here the control flow is explicit
and deterministic (no open-ended agent loop).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Optional

from ragnarok.generation.context import Citation, build_context
from ragnarok.generation.generate import generate_answer
from ragnarok.generation.grounding import check_grounding
from ragnarok.generation.postprocess import postprocess
from ragnarok.guardrails.input import check_input
from ragnarok.guardrails.output import check_output
from ragnarok.providers import role
from ragnarok.retrieval.orchestrator import retrieve
from ragnarok.retrieval.preprocess import QueryPlan, preprocess
from ragnarok.stores.factory import get_feature_store, get_vector_store
from ragnarok.stores.features import FeatureStore
from ragnarok.stores.vector import VectorStore
from ragnarok.user import User

_REFUSALS = {
    "query_too_long": "That question is too long — please shorten it.",
    "rate_limited": "You're sending requests too quickly. Please wait a moment and try again.",
    "pii_in_query": "Your message contained sensitive personal data, so I didn't process it. "
    "Please remove it and try again.",
    "prompt_injection": "I can't process that request.",
}


@dataclass
class AnswerResult:
    text: str
    citations: list[Citation] = field(default_factory=list)
    followups: list[str] = field(default_factory=list)
    grounded: bool = True
    grounding_score: float = 1.0
    abstained: bool = False
    blocked: bool = False
    degraded: bool = False
    retrieved: int = 0
    trace_id: str = ""


async def _direct_answer(plan: QueryPlan) -> str:
    """Chitchat / no-retrieval path: answer briefly without context."""
    resp = await role("llm_small").complete(
        [
            {"role": "system", "content": "You are a concise, friendly assistant for internal staff."},
            {"role": "user", "content": plan.rewritten_query},
        ]
    )
    return resp.content


async def answer(
    query: str,
    user: Optional[User] = None,
    *,
    store: Optional[VectorStore] = None,
    features: Optional[FeatureStore] = None,
    history: Optional[list[str]] = None,
    facets: Optional[dict[str, list[str]]] = None,
    collection: str = "chunks",
) -> AnswerResult:
    user = user or User()
    store = store or get_vector_store()
    features = features or get_feature_store()
    history = history or []
    trace_id = uuid.uuid4().hex

    # 1. input guardrails
    iv = check_input(query, user)
    if not iv.allowed:
        return AnswerResult(
            text=_REFUSALS.get(iv.reason, "I can't process that request."),
            blocked=True, trace_id=trace_id,
        )

    # 2. pre-process (query optimizer + source identifier)
    pre = await preprocess(iv.sanitized, history, user, facets)

    # 2b. chitchat gate: skip retrieval + heavy generation entirely
    if pre.skip_retrieval:
        text = await _direct_answer(pre.plan)
        ov = check_output(text)
        return AnswerResult(text=ov.answer if ov.allowed else "I can't answer that.", trace_id=trace_id)

    # 3. retrieval (filter -> hybrid -> rerank -> signals)
    results = retrieve(pre, user, store, features, collection=collection)
    ctx = build_context(results)

    # 4. generation + post-processing
    raw = await generate_answer(pre.plan.rewritten_query, ctx)
    pp = await postprocess(pre.plan.rewritten_query, raw, ctx)

    # 5. grounding gate (abstain if unsupported)
    gv = check_grounding(pp, ctx)

    # 6. output guardrails
    ov = check_output(gv.answer)
    if not ov.allowed:
        return AnswerResult(text="I can't share that answer.", blocked=True,
                            retrieved=len(results), trace_id=trace_id)

    return AnswerResult(
        text=ov.answer,
        citations=pp.resolved_citations,
        followups=pp.followups,
        grounded=gv.grounded,
        grounding_score=gv.score,
        abstained=not gv.grounded,
        retrieved=len(results),
        trace_id=trace_id,
    )


async def answer_once(query: str) -> AnswerResult:
    """CLI entrypoint (`ragnarok ask`). Uses the configured stores (Qdrant/Feast in prod)."""
    return await answer(query)
