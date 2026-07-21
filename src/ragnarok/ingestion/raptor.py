"""RAPTOR — Recursive Abstractive Processing for Tree-Organized Retrieval (Step 37).

Sarthi et al. 2024 (arXiv:2401.18059). Long documents lose high-level context when chunked flat.
RAPTOR clusters related chunks, summarizes each cluster into a higher-level node, and recurses,
forming a tree. Retrieval over the *collapsed tree* (all levels indexed together) surfaces both fine
detail (leaf chunks) and high-level context (summary nodes).

RAGnarok's implementation: level-1 clusters by section (deterministic; embedding clustering can
swap in), summarized by the large LLM; higher levels summarize the previous level's summaries. The
summary nodes are embedded and added to the same collection, so normal hybrid retrieval (Steps 16-18)
does collapsed-tree retrieval for free. Run as an explicit offline index augmentation.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict

from ragnarok import resilience
from ragnarok.ingestion.chunking import Chunk
from ragnarok.ingestion.embedding import EmbeddedChunk, embed_chunks
from ragnarok.ingestion.models import sha256_of
from ragnarok.prompts import prompts
from ragnarok.stores.vector import VectorStore
from ragnarok.tokenization import count_tokens


async def _summarize(passages: list[str]) -> str:
    text = "\n\n---\n\n".join(passages)
    msgs = prompts().render("raptor_summary", "latest", passages=text)
    resp = await resilience.call("llm_large", msgs, fallback_role="llm_small", stage="raptor")
    return resp.content.strip()


def _summary_chunk(doc_id: str, title: str, text: str, level: int, base_meta: dict, pos: int) -> Chunk:
    meta = {**base_meta, "level": level, "raptor": True}
    return Chunk(
        chunk_id=sha256_of(doc_id, "raptor", level, pos, text),
        doc_id=doc_id,
        text=text,
        contextual_prefix=f"Document: {title} (summary, level {level}).",
        chunk_type="summary",
        position=10_000 * level + pos,
        token_count=count_tokens(text),
        metadata=meta,
        content_hash=sha256_of(text),
    )


async def build_raptor_chunks(leaf_chunks: list[Chunk], *, doc_id: str, title: str, levels: int = 2) -> list[Chunk]:
    """Build summary nodes (levels 1..N) from leaf chunks by clustering on section."""
    if not leaf_chunks or levels < 1:
        return []
    summaries: list[Chunk] = []

    # level 1: one summary per section cluster
    clusters: dict[str, list[Chunk]] = defaultdict(list)
    for c in leaf_chunks:
        clusters[c.metadata.get("section", "")].append(c)
    base_meta = {k: leaf_chunks[0].metadata.get(k) for k in ("title", "uri", "doc_type", "audience", "access_tags", "authority")}

    level1_texts = await asyncio.gather(*[_summarize([c.text for c in group]) for group in clusters.values()])
    for pos, summary in enumerate(level1_texts):
        summaries.append(_summary_chunk(doc_id, title, summary, 1, base_meta, pos))

    # higher levels: summarize the previous level's summaries (doc-level rollup)
    prev = level1_texts
    for level in range(2, levels + 1):
        if len(prev) <= 1:
            break
        rollup = await _summarize(list(prev))
        summaries.append(_summary_chunk(doc_id, title, rollup, level, base_meta, 0))
        prev = [rollup]
    return summaries


def build_raptor_index(leaf_chunks: list[Chunk], *, doc_id: str, title: str, store: VectorStore,
                       levels: int = 2, collection: str = "chunks") -> list[EmbeddedChunk]:
    """Sync entry point: build + embed + upsert RAPTOR summary nodes into the collection."""
    summaries = asyncio.run(build_raptor_chunks(leaf_chunks, doc_id=doc_id, title=title, levels=levels))
    embedded = embed_chunks(summaries)
    if embedded:
        store.upsert(embedded, collection=collection)
    return embedded
