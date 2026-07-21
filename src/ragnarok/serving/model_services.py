"""Embedding + reranking microservice (Step 2).

Why a separate service (not in-process): a slow generation must never block a fast embed, and
embed/rerank scale independently of the LLMs (per-role endpoints, Step 2/3). This wraps the real
models (BGE-M3 dense+sparse embeddings, BGE-reranker cross-encoder) behind a tiny HTTP API.

Backends, selected by env (never hardcoded):
  - RAGNAROK_MODELS_BACKEND=flag  -> FlagEmbedding on GPU (production)
  - RAGNAROK_MODELS_BACKEND=local -> DeterministicEmbedder/LexicalReranker (no GPU; dev/CI)

Run:
    RAGNAROK_MODELS_BACKEND=local uvicorn ragnarok.serving.model_services:app --port 7997
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from ragnarok._local_backends import DeterministicEmbedder, LexicalReranker


class _LocalBackend:
    """Deterministic backend (no external weights). Serves both embed and rerank."""

    name = "local-deterministic"

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim
        self._emb = DeterministicEmbedder(dim=dim)
        self._rr = LexicalReranker()

    def embed(self, texts: list[str]) -> list[dict[str, Any]]:
        out = []
        for dense, sparse in self._emb.embed_many(texts):
            out.append(
                {
                    "dense": dense,
                    "sparse": {"indices": list(sparse.keys()), "values": list(sparse.values())},
                }
            )
        return out

    def rerank(self, query: str, passages: list[str]) -> list[float]:
        return self._rr.score_many(query, passages)


class _FlagBackend:  # pragma: no cover - requires GPU + FlagEmbedding weights
    """Production backend: BGE-M3 (dense+sparse) + BGE-reranker-v2-m3 cross-encoder."""

    name = "bge-m3 + bge-reranker-v2-m3"

    def __init__(self) -> None:
        from FlagEmbedding import BGEM3FlagModel, FlagReranker

        self._emb = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
        self._rr = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)

    def embed(self, texts: list[str]) -> list[dict[str, Any]]:
        res = self._emb.encode(texts, return_dense=True, return_sparse=True)
        out = []
        for dense, lex in zip(res["dense_vecs"], res["lexical_weights"]):
            out.append(
                {
                    "dense": [float(x) for x in dense],
                    "sparse": {
                        "indices": [int(k) for k in lex],
                        "values": [float(v) for v in lex.values()],
                    },
                }
            )
        return out

    def rerank(self, query: str, passages: list[str]) -> list[float]:
        scores = self._rr.compute_score([[query, p] for p in passages], normalize=True)
        return [float(s) for s in (scores if isinstance(scores, list) else [scores])]


@lru_cache
def get_backend() -> Any:
    backend = os.environ.get("RAGNAROK_MODELS_BACKEND", "local").lower()
    if backend == "flag":
        return _FlagBackend()
    return _LocalBackend(dim=int(os.environ.get("RAGNAROK_EMBED_DIM", "256")))


def _build_app() -> Any:  # pragma: no cover - exercised via integration, not unit tests
    from fastapi import FastAPI
    from pydantic import BaseModel

    class EmbedRequest(BaseModel):
        texts: list[str]

    class RerankRequest(BaseModel):
        query: str
        passages: list[str]

    app = FastAPI(title="RAGnarok model service")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "backend": get_backend().name}

    @app.post("/embed")
    def embed(req: EmbedRequest) -> dict[str, Any]:
        return {"data": get_backend().embed(req.texts)}

    @app.post("/rerank")
    def rerank(req: RerankRequest) -> dict[str, Any]:
        return {"scores": get_backend().rerank(req.query, req.passages)}

    return app


# `app` is created lazily so importing this module never requires FastAPI installed.
def __getattr__(name: str) -> Any:
    if name == "app":
        return _build_app()
    raise AttributeError(name)
