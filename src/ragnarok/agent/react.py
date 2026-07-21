"""Bounded ReAct loop (Step 39).

Thought -> Action -> Observation, repeated up to a step budget. The planner (a small LLM with guided
JSON) picks the next tool; the tool runs; the observation feeds the scratchpad. Retrieval tools
accumulate evidence (RetrievalResults) that the shared generator turns into a grounded, cited answer
(so Agentic RAG still passes the grounding gate and guardrails). Bounded — never an open-ended loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel

from ragnarok.agent.memory import ShortTermMemory, long_term_memory
from ragnarok.agent.tools import ToolRegistry, build_registry
from ragnarok.config import get_settings
from ragnarok.prompts import prompts
from ragnarok.retrieval.hybrid import RetrievalResult
from ragnarok.structured import generate_structured


class AgentAction(BaseModel):
    thought: str = ""
    tool: Literal["retrieve", "calculator", "finish"] = "retrieve"
    tool_input: str = ""


@dataclass
class AgentTrace:
    steps: list[dict] = field(default_factory=list)
    results: list[RetrievalResult] = field(default_factory=list)


async def run_react(ctx, *, registry: ToolRegistry | None = None) -> AgentTrace:
    registry = registry or build_registry()
    cfg = get_settings().rag
    short = ShortTermMemory()
    trace = AgentTrace()
    question = ctx.pre.plan.rewritten_query
    memory = "\n".join(long_term_memory().recall(ctx.user.id, question)) or "(none)"
    seen: set[str] = set()

    for _ in range(max(cfg.agentic_max_steps, 1)):
        msgs = prompts().render(
            "agent_react", "latest", question=question, memory=memory,
            tools=registry.describe(), scratchpad=short.render() or "(empty)",
        )
        action = await generate_structured("llm_small", msgs, AgentAction)
        trace.steps.append(action.model_dump())
        if action.tool == "finish":
            break
        tool = registry.get(action.tool)
        if tool is None:
            short.add(action.thought, action.tool, action.tool_input, "unknown tool")
            continue
        result = await tool.run(action.tool_input, ctx)
        for r in result.results:  # accumulate evidence, deduped
            if r.chunk_id not in seen:
                seen.add(r.chunk_id)
                trace.results.append(r)
        short.add(action.thought, action.tool, action.tool_input, result.observation)

    return trace
