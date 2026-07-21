"""Multimodal RAG strategy (Step 36).

CLIP/ColPali-style multimodal retrieval. Images/diagrams are captioned at ingest (Step 36) and
embedded alongside text, so a single hybrid search retrieves across modalities. The generator
receives the caption plus the asset reference (payload ``asset_uri``) so it can cite/link the image.
This strategy runs hybrid retrieval and reports the modality mix; a production build embeds images
with a true multimodal model and can balance modalities per query.
"""

from __future__ import annotations

from collections import Counter

from ragnarok.retrieval.orchestrator import retrieve
from ragnarok.strategies import register
from ragnarok.strategies.base import StrategyContext, StrategyResult


@register
class MultimodalStrategy:
    name = "multimodal"

    async def run(self, ctx: StrategyContext) -> StrategyResult:
        results = retrieve(ctx.pre, ctx.user, ctx.store, ctx.features, collection=ctx.collection)
        mix = Counter(r.payload.get("chunk_type", "text") for r in results)
        return StrategyResult(results, self.name, notes={"modalities": dict(mix)})
