"""RAPTOR retrieval strategy (Step 37).

RAPTOR is primarily an *indexing* method (a summary tree built at ingest, see
``ragnarok.ingestion.raptor``). Retrieval is standard collapsed-tree retrieval: hybrid search over a
collection that contains both leaf chunks and summary nodes, so an answer can draw on fine detail and
high-level context together. This strategy runs hybrid retrieval and reports the tree-level mix.
"""

from __future__ import annotations

from collections import Counter

from ragnarok.retrieval.orchestrator import retrieve
from ragnarok.strategies import register
from ragnarok.strategies.base import StrategyContext, StrategyResult


@register
class RaptorStrategy:
    name = "raptor"

    async def run(self, ctx: StrategyContext) -> StrategyResult:
        results = retrieve(ctx.pre, ctx.user, ctx.store, ctx.features, collection=ctx.collection)
        levels = Counter(r.payload.get("level", 0) for r in results)
        return StrategyResult(results, self.name, notes={"levels": dict(levels)})
