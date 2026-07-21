"""Embedding generation (Step 9).

Embeds chunks with a hybrid model (BGE-M3: dense + sparse in one pass) so hybrid retrieval needs no
second system. Two backends chosen by env (never hardcoded):
  RAGNAROK_EMBED_BACKEND=http  -> the embed service (Step 2), production
  RAGNAROK_EMBED_BACKEND=local -> DeterministicEmbedder, no GPU (dev/CI/offline demos)

Batched for throughput and content-hash cached so re-embedding an unchanged corpus is ~free.
"""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

from ragnarok.cache import get_cache, get_json, make_key, set_json
from ragnarok.config import get_settings
from ragnarok.ingestion.chunking import Chunk


@dataclass
class EmbedResult:
    dense: list[float]
    sparse: dict[int, float]


class EmbeddedChunk(Chunk):
    dense_vector: list[float] = []
    sparse_vector: dict[int, float] = {}
    embedding_model: str = ""
    embedding_version: str = ""


def _batched(items: list, size: int) -> Iterable[list]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


class _LocalBackend:
    name = "local-deterministic"

    def __init__(self, dim: int = 256) -> None:
        from ragnarok._local_backends import DeterministicEmbedder

        self._emb = DeterministicEmbedder(dim=dim)

    def embed(self, texts: list[str]) -> list[EmbedResult]:
        return [EmbedResult(dense=d, sparse=s) for d, s in self._emb.embed_many(texts)]


class _HttpBackend:  # pragma: no cover - requires the running embed service
    name = "http"

    def __init__(self, base_url: str) -> None:
        self.url = base_url.rstrip("/") + "/embed"

    def embed(self, texts: list[str]) -> list[EmbedResult]:
        req = urllib.request.Request(
            self.url,
            data=json.dumps({"texts": texts}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as r:  # noqa: S310
            data = json.loads(r.read())
        out = []
        for item in data["data"]:
            sparse = dict(zip(item["sparse"]["indices"], item["sparse"]["values"]))
            out.append(EmbedResult(dense=item["dense"], sparse=sparse))
        return out


class EmbeddingClient:
    def __init__(self, backend: str | None = None) -> None:
        cfg = get_settings().models.embedding
        chosen = backend or os.environ.get("RAGNAROK_EMBED_BACKEND", "local")
        if chosen == "http":
            self.backend = _HttpBackend(cfg.base_url)
        else:
            self.backend = _LocalBackend(dim=int(os.environ.get("RAGNAROK_EMBED_DIM", "256")))
        self.model = cfg.model
        self.version = f"{cfg.model}@{self.backend.name}"

    def embed_texts(self, texts: list[str], *, use_cache: bool = True, batch_size: int = 64) -> list[EmbedResult]:
        cache = get_cache()
        results: list[EmbedResult | None] = [None] * len(texts)
        misses: list[tuple[int, str]] = []
        for i, t in enumerate(texts):
            key = make_key("emb", self.version, _hash(t))
            if use_cache and (cached := get_json(cache, key)) is not None:
                results[i] = EmbedResult(cached["dense"], {int(k): v for k, v in cached["sparse"].items()})
            else:
                misses.append((i, t))

        for batch in _batched(misses, batch_size):
            embedded = self.backend.embed([t for _, t in batch])
            for (idx, text), res in zip(batch, embedded):
                results[idx] = res
                if use_cache:
                    set_json(
                        cache,
                        make_key("emb", self.version, _hash(text)),
                        {"dense": res.dense, "sparse": res.sparse},
                    )
        return [r for r in results if r is not None]


def _hash(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode()).hexdigest()


@lru_cache
def get_embedding_client() -> EmbeddingClient:
    return EmbeddingClient()


def embed_chunks(chunks: list[Chunk]) -> list[EmbeddedChunk]:
    client = get_embedding_client()
    results = client.embed_texts([c.embedding_text() for c in chunks])
    out = []
    for chunk, res in zip(chunks, results):
        out.append(
            EmbeddedChunk(
                **chunk.model_dump(),
                dense_vector=res.dense,
                sparse_vector=res.sparse,
                embedding_model=client.model,
                embedding_version=client.version,
            )
        )
    return out
