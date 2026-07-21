"""Store factories (Step 10/11). Backend chosen by env, never hardcoded."""

from __future__ import annotations

import os
from functools import lru_cache

from ragnarok.config import get_settings
from ragnarok.stores.vector import InMemoryVectorStore, VectorStore


@lru_cache
def get_vector_store() -> VectorStore:
    backend = os.environ.get("RAGNAROK_VECTOR_STORE", "memory").lower()
    if backend == "qdrant":  # pragma: no cover - requires a running Qdrant
        from ragnarok.stores.vector import QdrantVectorStore

        cfg = get_settings()
        return QdrantVectorStore(
            cfg.stores.qdrant_url,
            dim=cfg.models.embedding.dim,
            quantization=cfg.stores.vector_quantization,
        )
    return InMemoryVectorStore()


def reset_stores() -> None:
    get_vector_store.cache_clear()
