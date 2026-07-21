"""Cross-encoder reranking (Step 17).

We retrieve broad (high recall) then rerank narrow (high precision): a cross-encoder reads the query
and each passage *together*, far sharper than bi-encoder similarity. The generator then sees a small,
dense, highly-relevant context -> better answers AND fewer input tokens (Steps 19-20). A score floor
drops weak chunks to prevent context dilution.

Backends by env: http (BGE-reranker service, prod) / local (LexicalReranker, no GPU).
"""

from __future__ import annotations

import json
import os
import urllib.request
from functools import lru_cache

from ragnarok.config import RetrievalCfg, get_settings
from ragnarok.retrieval.hybrid import RetrievalResult


class _LocalReranker:
    name = "local-lexical"

    def __init__(self) -> None:
        from ragnarok._local_backends import LexicalReranker

        self._rr = LexicalReranker()

    def score(self, query: str, passages: list[str]) -> list[float]:
        return self._rr.score_many(query, passages)


class _HttpReranker:  # pragma: no cover - requires the reranker service
    name = "http"

    def __init__(self, url: str) -> None:
        self.url = url.rstrip("/") + "/rerank"

    def score(self, query: str, passages: list[str]) -> list[float]:
        req = urllib.request.Request(
            self.url,
            data=json.dumps({"query": query, "passages": passages}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as r:  # noqa: S310
            return json.loads(r.read())["scores"]


@lru_cache
def get_reranker():
    backend = os.environ.get("RAGNAROK_RERANK_BACKEND", "local").lower()
    if backend == "http":
        return _HttpReranker(get_settings().models.reranker.url)
    return _LocalReranker()


def _passage_text(r: RetrievalResult) -> str:
    parts = [r.payload.get("contextual_prefix", ""), r.payload.get("text", "")]
    if r.payload.get("table_markdown"):
        parts.append(r.payload["table_markdown"])
    return "\n".join(p for p in parts if p)


def rerank(
    query: str, results: list[RetrievalResult], *, cfg: RetrievalCfg | None = None
) -> list[RetrievalResult]:
    cfg = cfg or get_settings().retrieval
    if not results:
        return []
    candidates = results[: max(cfg.top_k_dense + cfg.top_k_sparse, cfg.rerank_top_n)]
    scores = get_reranker().score(query, [_passage_text(r) for r in candidates])
    for r, s in zip(candidates, scores):
        r.rerank_score = float(s)
        r.final_score = float(s)  # Step 18 layers business signals on top of this
    candidates.sort(key=lambda r: r.rerank_score, reverse=True)
    kept = [r for r in candidates if r.rerank_score >= cfg.min_rerank_score]
    # never return empty purely due to the floor: keep the single best as a hedge
    return (kept or candidates[:1])[: cfg.rerank_top_n]
