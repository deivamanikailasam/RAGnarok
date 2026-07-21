"""Hybrid retrieval: dense + sparse search fused with RRF (Step 16).

Dense embeddings catch semantics/paraphrase; sparse/BM25 catches exact terms (plan names, SKUs,
codes). Reciprocal Rank Fusion combines them by *rank* (no fragile score normalization). Sub-queries
from decomposition are each retrieved and folded into one fusion pool. If a narrow filter starves
retrieval, fall back to an access-scoped unfiltered search (never relaxing security).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ragnarok.config import RetrievalCfg, get_settings
from ragnarok.ingestion.embedding import get_embedding_client
from ragnarok.retrieval.preprocess import QueryPlan
from ragnarok.stores.vector import MetadataFilter, SearchHit, VectorStore


@dataclass
class RetrievalResult:
    chunk_id: str
    payload: dict[str, Any] = field(default_factory=dict)
    dense_score: float = 0.0
    sparse_score: float = 0.0
    fused_score: float = 0.0
    rerank_score: float = 0.0
    final_score: float = 0.0

    @property
    def doc_id(self) -> str:
        return self.payload.get("doc_id", "")


def rrf_fuse(hit_lists: list[list[SearchHit]], k: int = 60) -> dict[str, float]:
    scores: dict[str, float] = defaultdict(float)
    for ranked in hit_lists:
        for rank, hit in enumerate(ranked):
            scores[hit.id] += 1.0 / (k + rank)
    return scores


def _search_once(
    store: VectorStore, plan: QueryPlan, flt: MetadataFilter | None, cfg: RetrievalCfg, collection: str
) -> tuple[list[list[SearchHit]], dict[str, SearchHit], dict[str, float], dict[str, float]]:
    embedder = get_embedding_client()
    queries = plan.retrieval_queries()
    embeds = embedder.embed_texts(queries)
    hit_lists: list[list[SearchHit]] = []
    by_id: dict[str, SearchHit] = {}
    dense_best: dict[str, float] = defaultdict(float)
    sparse_best: dict[str, float] = defaultdict(float)
    for e in embeds:
        dense = store.search_dense(e.dense, filter=flt, limit=cfg.top_k_dense, collection=collection)
        sparse = store.search_sparse(e.sparse, filter=flt, limit=cfg.top_k_sparse, collection=collection)
        hit_lists.append(dense)
        hit_lists.append(sparse)
        for h in dense:
            by_id[h.id] = h
            dense_best[h.id] = max(dense_best[h.id], h.score)
        for h in sparse:
            by_id.setdefault(h.id, h)
            sparse_best[h.id] = max(sparse_best[h.id], h.score)
    return hit_lists, by_id, dense_best, sparse_best


def hybrid_search(
    plan: QueryPlan,
    flt: MetadataFilter | None,
    store: VectorStore,
    *,
    cfg: RetrievalCfg | None = None,
    collection: str = "chunks",
    relax_fn: Callable[[], MetadataFilter] | None = None,
) -> list[RetrievalResult]:
    cfg = cfg or get_settings().retrieval
    hit_lists, by_id, dense_best, sparse_best = _search_once(store, plan, flt, cfg, collection)
    fused = rrf_fuse(hit_lists, k=cfg.rrf_k)

    # Fallback: a narrow filter starved retrieval -> relax to an access-scoped search (Step 15).
    if len(fused) < cfg.filter_fallback_min_candidates and relax_fn is not None:
        hit_lists, by_id, dense_best, sparse_best = _search_once(
            store, plan, relax_fn(), cfg, collection
        )
        fused = rrf_fuse(hit_lists, k=cfg.rrf_k)

    results = [
        RetrievalResult(
            chunk_id=cid,
            payload=by_id[cid].payload,
            dense_score=dense_best.get(cid, 0.0),
            sparse_score=sparse_best.get(cid, 0.0),
            fused_score=score,
        )
        for cid, score in fused.items()
    ]
    results.sort(key=lambda r: r.fused_score, reverse=True)
    return results
