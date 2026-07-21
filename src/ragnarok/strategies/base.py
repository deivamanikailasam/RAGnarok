"""RAG strategy framework (Step 29).

Every RAG architecture (Naive, Hybrid, HyDE, Fusion, Corrective, Self-RAG, Graph, Hybrid+Graph,
Multimodal, RAPTOR, Adaptive, Agentic) implements one interface, so they are interchangeable,
individually testable, and comparable head-to-head through the eval harness (Step 24). The online
pipeline (Step 23) runs the selected strategy, then applies the SHARED generate -> post-process ->
grounding-gate -> guardrail path — so every strategy inherits citations, abstention, guardrails,
adaptive budgets, and tracing for free.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ragnarok.retrieval.hybrid import RetrievalResult
from ragnarok.retrieval.preprocess import PreprocessResult
from ragnarok.stores.features import FeatureStore
from ragnarok.stores.vector import VectorStore
from ragnarok.user import User


@dataclass
class StrategyContext:
    query: str  # sanitized user query
    pre: PreprocessResult  # plan (rewrite/intent/sub-queries) + source filters
    user: User
    store: VectorStore
    features: FeatureStore
    collection: str = "chunks"
    graph: Any | None = None  # KnowledgeGraph for graph strategies (Step 34)
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyResult:
    results: list[RetrievalResult]
    strategy: str
    direct_answer: str | None = None  # set for the no-retrieval path (Adaptive/Self-RAG)
    notes: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class RagStrategy(Protocol):
    name: str

    async def run(self, ctx: StrategyContext) -> StrategyResult: ...
