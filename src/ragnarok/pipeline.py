"""Online answer pipeline (Step 23) — the agentic graph tying every stage together.

guard-in -> preprocess -> retrieve -> context -> generate -> post-process -> grounding gate ->
guard-out. This is the single implementation shared by Slack, the API, and the CLI so behavior never
drifts across surfaces. Observability (Step 26) wraps each stage; here the control flow is explicit
and deterministic (no open-ended agent loop).
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field

from ragnarok.cache import get_cache, get_json, make_key, set_json
from ragnarok.config import get_settings
from ragnarok.generation.context import Citation, build_context
from ragnarok.generation.context import Citation as _Citation
from ragnarok.generation.generate import generate_answer
from ragnarok.generation.grounding import check_grounding
from ragnarok.generation.postprocess import postprocess
from ragnarok.guardrails.input import check_input
from ragnarok.guardrails.output import check_output
from ragnarok.observability import metrics
from ragnarok.observability.trace import span, trace
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
    cache_hit: bool = False


def _cache_key(query: str, user: User, collection: str) -> str:
    norm = " ".join(query.lower().split())  # normalization widens hit rate
    ents = ",".join(sorted(user.entitlements))
    return make_key("ans", collection, hash((norm, ents)))


def _serialize(r: AnswerResult) -> dict:
    d = asdict(r)
    return d


def _deserialize(d: dict) -> AnswerResult:
    d = dict(d)
    d["citations"] = [_Citation(**c) for c in d.get("citations", [])]
    return AnswerResult(**d)


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
    user: User | None = None,
    *,
    store: VectorStore | None = None,
    features: FeatureStore | None = None,
    history: list[str] | None = None,
    facets: dict[str, list[str]] | None = None,
    collection: str = "chunks",
) -> AnswerResult:
    user = user or User()
    store = store or get_vector_store()
    features = features or get_feature_store()
    history = history or []
    trace_id = uuid.uuid4().hex

    with trace("ask", trace_id=trace_id, user=user.id) as tr:
        # 1. input guardrails
        with span("guard.input") as s:
            iv = check_input(query, user)
            s.set(allowed=iv.allowed, reason=iv.reason, flags=iv.flags)
        if not iv.allowed:
            metrics.record_answer(abstained=False, blocked=True)
            return AnswerResult(
                text=_REFUSALS.get(iv.reason, "I can't process that request."),
                blocked=True, trace_id=trace_id,
            )

        # Response cache (Step 27): identical question + entitlements served instantly.
        # Only consulted when there's no chat history (follow-ups depend on context).
        cache = get_cache()
        ckey = _cache_key(iv.sanitized, user, collection)
        if not history and (cached := get_json(cache, ckey)) is not None:
            metrics.collector().inc("response_cache_hit")
            result = _deserialize(cached)
            result.cache_hit = True
            result.trace_id = trace_id
            return result

        # 2. pre-process (query optimizer + source identifier)
        with span("preprocess"):
            pre = await preprocess(iv.sanitized, history, user, facets)

        # 2b. chitchat gate: skip retrieval + heavy generation entirely
        if pre.skip_retrieval:
            text = await _direct_answer(pre.plan)
            ov = check_output(text)
            metrics.record_answer(abstained=False, blocked=not ov.allowed)
            return AnswerResult(
                text=ov.answer if ov.allowed else "I can't answer that.", trace_id=trace_id
            )

        # 3. retrieval (filter -> hybrid -> rerank -> signals)
        with span("retrieve") as s:
            results = retrieve(pre, user, store, features, collection=collection)
            s.set(returned=len(results))
        ctx = build_context(results)

        # 4. generation + post-processing
        with span("generate"):
            raw = await generate_answer(pre.plan.rewritten_query, ctx)
        with span("postprocess"):
            pp = await postprocess(pre.plan.rewritten_query, raw, ctx)

        # 5. grounding gate (abstain if unsupported)
        with span("grounding") as s:
            gv = check_grounding(pp, ctx)
            s.set(score=gv.score, grounded=gv.grounded)
        metrics.record_grounding(gv.score)

        # 6. output guardrails
        with span("guard.output"):
            ov = check_output(gv.answer)
        if not ov.allowed:
            metrics.record_answer(abstained=not gv.grounded, blocked=True)
            return AnswerResult(text="I can't share that answer.", blocked=True,
                                retrieved=len(results), trace_id=trace_id)

        metrics.record_answer(abstained=not gv.grounded, blocked=False)
        tr.root.set(tokens=tr.total_tokens, cost=tr.total_cost)
        result = AnswerResult(
            text=ov.answer,
            citations=pp.resolved_citations,
            followups=pp.followups,
            grounded=gv.grounded,
            grounding_score=gv.score,
            abstained=not gv.grounded,
            retrieved=len(results),
            trace_id=trace_id,
        )
        # cache only confident, grounded answers (never abstentions/blocks)
        if gv.grounded and not history:
            set_json(cache, ckey, _serialize(result), ttl_s=get_settings().caching.response_ttl_s)
        return result


async def answer_once(query: str) -> AnswerResult:
    """CLI entrypoint (`ragnarok ask`). Uses the configured stores (Qdrant/Feast in prod)."""
    return await answer(query)
