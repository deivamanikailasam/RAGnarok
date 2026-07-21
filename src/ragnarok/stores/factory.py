"""Store factories (Step 10/11). Backend chosen by env, never hardcoded."""

from __future__ import annotations

import os
from functools import lru_cache

from ragnarok.config import get_settings
from ragnarok.stores.features import FeatureStore, InMemoryFeatureStore
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


@lru_cache
def get_feature_store() -> FeatureStore:
    backend = os.environ.get("RAGNAROK_FEATURE_STORE", "memory").lower()
    if backend == "feast":  # pragma: no cover - requires Feast + Postgres/Redis
        from ragnarok.stores.features import FeastFeatureStore

        return FeastFeatureStore(os.environ.get("FEAST_REPO", "feature_repo"))
    return InMemoryFeatureStore()


@lru_cache
def get_graph_store():  # -> KnowledgeGraph (Step 34)
    from ragnarok.stores.graph import KnowledgeGraph

    return KnowledgeGraph()


def reset_stores() -> None:
    get_vector_store.cache_clear()
    get_feature_store.cache_clear()
    get_graph_store.cache_clear()
