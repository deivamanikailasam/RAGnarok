"""Query pre-processing (Step 14): query optimizer + source identifier.

Two small-LLM agents that fix the *input* to retrieval — the highest-ROI place to spend a small
model. The optimizer resolves thread context, rewrites, expands, decomposes, and gates chitchat.
The source identifier emits metadata filters so retrieval searches the right subset. Both are
guided-JSON, temperature 0, and cached; they run concurrently.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field

from ragnarok.cache import get_cache, get_json, make_key, set_json
from ragnarok.config import get_settings
from ragnarok.prompts import prompts
from ragnarok.structured import generate_structured
from ragnarok.user import User


class QueryPlan(BaseModel):
    intent: str = "other"
    rewritten_query: str
    sub_queries: list[str] = Field(default_factory=list)
    expansions: list[str] = Field(default_factory=list)
    needs_retrieval: bool = True
    hyde_hint: str | None = None

    def retrieval_queries(self) -> list[str]:
        return self.sub_queries or [self.rewritten_query]


class SourcePlan(BaseModel):
    equals: dict[str, list[str]] = Field(default_factory=dict)  # field -> allowed values
    must_access_tags: list[str] = Field(default_factory=list)  # model suggestion (Step 15 gates it)
    boost_recent: bool = False
    confidence: float = 0.5


@dataclass
class PreprocessResult:
    plan: QueryPlan
    source: SourcePlan | None = None
    skip_retrieval: bool = False


async def optimize_query(query: str, history: list[str]) -> QueryPlan:
    cache = get_cache()
    version = prompts().label("query_optimizer")
    key = make_key("qopt", version, hash((query, tuple(history))))
    if (cached := get_json(cache, key)) is not None:
        return QueryPlan.model_validate(cached)
    msgs = prompts().render("query_optimizer", "latest", query=query, chat_history=history)
    plan = await generate_structured("llm_small", msgs, QueryPlan)
    if not plan.rewritten_query:
        plan.rewritten_query = query
    set_json(cache, key, plan.model_dump(), ttl_s=get_settings().caching.rewrite_ttl_s)
    return plan


async def identify_sources(plan: QueryPlan, user: User, facets: dict[str, list[str]]) -> SourcePlan:
    msgs = prompts().render(
        "source_identifier", "latest", query=plan.rewritten_query, intent=plan.intent, facets=facets
    )
    return await generate_structured("llm_small", msgs, SourcePlan)


async def preprocess(
    query: str, history: list[str], user: User, facets: dict[str, list[str]] | None = None
) -> PreprocessResult:
    plan = await optimize_query(query, history)
    if not plan.needs_retrieval:
        return PreprocessResult(plan=plan, skip_retrieval=True)
    source = await identify_sources(plan, user, facets or {})
    return PreprocessResult(plan=plan, source=source)
