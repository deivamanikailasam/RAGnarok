"""Answer generation (Step 20).

The large LLM generates a streamed answer constrained to "answer only from the numbered context,
cite with [i], don't recompute table numbers, abstain if insufficient." Streaming is essential to
perceived latency (first token < 1s). A fallback to llm_small keeps answers flowing under load
(labelled degraded).
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from ragnarok import resilience
from ragnarok.generation.context import AssembledContext
from ragnarok.prompts import prompts
from ragnarok.providers import role


def _messages(question: str, ctx: AssembledContext) -> list[dict[str, str]]:
    context_text = ctx.text or "(no context retrieved)"
    return prompts().render("answer_generator", "latest", question=question, context=context_text)


async def generate_answer(question: str, ctx: AssembledContext, *, role_name: str = "llm_large") -> str:
    resp = await resilience.call(role_name, _messages(question, ctx), fallback_role="llm_small")
    return resp.content


async def generate_stream(
    question: str, ctx: AssembledContext, *, role_name: str = "llm_large"
) -> AsyncIterator[str]:
    gen = await role(role_name).stream(_messages(question, ctx))
    async for delta in gen:
        yield delta
