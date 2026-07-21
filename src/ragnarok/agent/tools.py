"""Agent tools + registry (Step 39).

Tools are the agent's actions. The registry models the diagram's "MCP Servers" (local data, search,
cloud) as pluggable named tools: `retrieve` and `calculator` ship by default; search/cloud/local
tool servers register via ``register_tool`` (e.g. an MCP client adapter). Every tool returns a
ToolResult with a human-readable observation (for the ReAct scratchpad) and any RetrievalResults to
accumulate as evidence for the shared generator.
"""

from __future__ import annotations

import ast
import operator
from dataclasses import dataclass, field
from typing import Protocol

from ragnarok.retrieval.hybrid import RetrievalResult
from ragnarok.retrieval.orchestrator import retrieve
from ragnarok.retrieval.preprocess import PreprocessResult, QueryPlan


@dataclass
class ToolResult:
    observation: str
    results: list[RetrievalResult] = field(default_factory=list)


class Tool(Protocol):
    name: str
    description: str

    async def run(self, tool_input: str, ctx: object) -> ToolResult: ...


class RetrieveTool:
    name = "retrieve"
    description = "Search the internal knowledge base for passages relevant to a query."

    async def run(self, tool_input: str, ctx: object) -> ToolResult:
        plan = QueryPlan(rewritten_query=tool_input, intent="other")
        pre = PreprocessResult(plan=plan, source=ctx.pre.source)  # type: ignore[attr-defined]
        results = retrieve(pre, ctx.user, ctx.store, ctx.features, collection=ctx.collection)  # type: ignore[attr-defined]
        obs = "; ".join((r.payload.get("text") or r.payload.get("table_markdown") or "")[:120]
                        for r in results[:3]) or "no results"
        return ToolResult(observation=obs, results=results)


_ALLOWED_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg,
    ast.Mod: operator.mod,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("unsupported expression")


class CalculatorTool:
    name = "calculator"
    description = "Evaluate an arithmetic expression, e.g. '14 * 30 / 2'."

    async def run(self, tool_input: str, ctx: object) -> ToolResult:
        try:
            value = _safe_eval(ast.parse(tool_input, mode="eval").body)
            return ToolResult(observation=f"{tool_input} = {value}")
        except Exception as e:  # noqa: BLE001
            return ToolResult(observation=f"calculator error: {e}")


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def describe(self) -> str:
        return "\n".join(f"- {t.name}: {t.description}" for t in self._tools.values())


def default_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(RetrieveTool())
    reg.register(CalculatorTool())
    return reg


_extra_tools: list[Tool] = []


def register_tool(tool: Tool) -> None:
    """Register a tool server (local/search/cloud, MCP-style) available to the agent."""
    _extra_tools.append(tool)


def build_registry() -> ToolRegistry:
    reg = default_registry()
    for t in _extra_tools:
        reg.register(t)
    return reg
