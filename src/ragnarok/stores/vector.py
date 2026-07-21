"""Vector store (Step 10).

Holds each chunk with dense + sparse vectors and a rich payload used for filtering, security, and
reranking signals. Two backends behind one interface:
  - QdrantVectorStore: production (native sparse vectors + payload filtering + int8 quantization ->
    hybrid in one collection, one round trip; filtered search stays fast).
  - InMemoryVectorStore: dev/CI/offline — same semantics, no server.

Metadata filtering (incl. access_tags) happens *inside* search, so an unauthorized chunk is never
scored (Steps 15, 22).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

from pydantic import BaseModel, Field

from ragnarok.ingestion.embedding import EmbeddedChunk


class MetadataFilter(BaseModel):
    """AND across fields; MatchAny within a field. access_tags is the security intersection."""

    equals: dict[str, list[str]] = Field(default_factory=dict)  # field -> allowed values
    access_tags: list[str] = Field(default_factory=list)  # payload must overlap these
    freshness_after: Optional[str] = None  # ISO date lower bound

    def matches(self, payload: dict[str, Any]) -> bool:
        for field_name, allowed in self.equals.items():
            value = payload.get(field_name)
            values = value if isinstance(value, list) else [value]
            if not set(map(str, values)) & set(map(str, allowed)):
                return False
        if self.access_tags:
            tags = payload.get("access_tags") or []
            if not set(tags) & set(self.access_tags):
                return False
        if self.freshness_after:
            fd = payload.get("freshness_date")
            if fd is not None and str(fd) < self.freshness_after:
                return False
        return True


@dataclass
class SearchHit:
    id: str
    score: float
    payload: dict[str, Any] = field(default_factory=dict)


def chunk_to_payload(c: EmbeddedChunk) -> dict[str, Any]:
    p = {
        "doc_id": c.doc_id,
        "text": c.text,
        "table_markdown": c.table_markdown,
        "chunk_type": c.chunk_type,
        "contextual_prefix": c.contextual_prefix,
        "position": c.position,
        "content_hash": c.content_hash,
        "embedding_version": c.embedding_version,
    }
    p.update(c.metadata)  # title, section, doc_type, audience, topics, authority, access_tags, ...
    return p


class VectorStore(Protocol):
    def upsert(self, chunks: list[EmbeddedChunk], collection: str) -> None: ...
    def search_dense(
        self, vector: list[float], *, filter: MetadataFilter | None, limit: int, collection: str
    ) -> list[SearchHit]: ...
    def search_sparse(
        self, sparse: dict[int, float], *, filter: MetadataFilter | None, limit: int, collection: str
    ) -> list[SearchHit]: ...
    def delete_by_doc(self, doc_id: str, collection: str) -> int: ...
    def count(self, collection: str) -> int: ...
    def switch_alias(self, alias: str, target: str) -> None: ...
    def get_alias_target(self, alias: str) -> str: ...
    def list_collections(self) -> list[str]: ...


# --------------------------------------------------------------------------- in-memory


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def _sparse_dot(a: dict[int, float], b: dict[int, float]) -> float:
    small, large = (a, b) if len(a) < len(b) else (b, a)
    return sum(v * large.get(k, 0.0) for k, v in small.items())


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._c: dict[str, dict[str, tuple[EmbeddedChunk, dict]]] = {}
        self._aliases: dict[str, str] = {}

    # --- alias support for blue/green reindex (Step 12) ---
    def _resolve(self, name: str) -> str:
        return self._aliases.get(name, name)

    def switch_alias(self, alias: str, target: str) -> None:
        self._aliases[alias] = target

    def get_alias_target(self, alias: str) -> str:
        return self._aliases.get(alias, alias)

    def list_collections(self) -> list[str]:
        return [c for c in self._c if c]

    def _coll(self, name: str) -> dict:
        return self._c.setdefault(self._resolve(name), {})

    def upsert(self, chunks: list[EmbeddedChunk], collection: str = "chunks") -> None:
        coll = self._coll(collection)
        for c in chunks:
            coll[c.chunk_id] = (c, chunk_to_payload(c))  # id = content-derived -> idempotent

    def _filtered(self, collection: str, flt: MetadataFilter | None):
        for _id, (chunk, payload) in self._coll(collection).items():
            if flt is None or flt.matches(payload):
                yield chunk, payload

    def search_dense(self, vector, *, filter=None, limit=40, collection="chunks"):
        scored = [
            SearchHit(c.chunk_id, _cosine(vector, c.dense_vector), p)
            for c, p in self._filtered(collection, filter)
        ]
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:limit]

    def search_sparse(self, sparse, *, filter=None, limit=40, collection="chunks"):
        scored = [
            SearchHit(c.chunk_id, _sparse_dot(sparse, c.sparse_vector), p)
            for c, p in self._filtered(collection, filter)
        ]
        scored.sort(key=lambda h: h.score, reverse=True)
        return [h for h in scored if h.score > 0][:limit]

    def delete_by_doc(self, doc_id: str, collection: str = "chunks") -> int:
        coll = self._coll(collection)
        ids = [i for i, (c, _) in coll.items() if c.doc_id == doc_id]
        for i in ids:
            coll.pop(i, None)
        return len(ids)

    def count(self, collection: str = "chunks") -> int:
        return len(self._coll(collection))


# --------------------------------------------------------------------------- qdrant


class QdrantVectorStore:  # pragma: no cover - requires a running Qdrant
    def __init__(self, url: str, dim: int = 1024, quantization: str = "int8") -> None:
        from qdrant_client import QdrantClient

        self._client = QdrantClient(url=url)
        self.dim = dim
        self.quantization = quantization

    def ensure_collection(self, collection: str) -> None:
        from qdrant_client import models as qm

        if self._client.collection_exists(collection):
            return
        quant = (
            qm.ScalarQuantization(scalar=qm.ScalarQuantizationConfig(type=qm.ScalarType.INT8))
            if self.quantization == "int8"
            else None
        )
        self._client.create_collection(
            collection,
            vectors_config={"dense": qm.VectorParams(size=self.dim, distance=qm.Distance.COSINE, quantization_config=quant)},
            sparse_vectors_config={"sparse": qm.SparseVectorParams()},
        )
        for f in ["doc_type", "audience", "authority", "topics", "access_tags", "entities"]:
            self._client.create_payload_index(collection, f, field_schema="keyword")
        self._client.create_payload_index(collection, "freshness_date", field_schema="datetime")

    def _to_filter(self, flt: MetadataFilter | None):
        from qdrant_client import models as qm

        if flt is None:
            return None
        must = []
        for field_name, allowed in flt.equals.items():
            must.append(qm.FieldCondition(key=field_name, match=qm.MatchAny(any=allowed)))
        if flt.access_tags:
            must.append(qm.FieldCondition(key="access_tags", match=qm.MatchAny(any=flt.access_tags)))
        if flt.freshness_after:
            must.append(qm.FieldCondition(key="freshness_date", range=qm.DatetimeRange(gte=flt.freshness_after)))
        return qm.Filter(must=must) if must else None

    def upsert(self, chunks: list[EmbeddedChunk], collection: str = "chunks") -> None:
        from qdrant_client import models as qm

        self.ensure_collection(collection)
        points = []
        for c in chunks:
            points.append(
                qm.PointStruct(
                    id=c.chunk_id,
                    vector={
                        "dense": c.dense_vector,
                        "sparse": qm.SparseVector(
                            indices=list(c.sparse_vector.keys()),
                            values=list(c.sparse_vector.values()),
                        ),
                    },
                    payload=chunk_to_payload(c),
                )
            )
        self._client.upsert(collection, points=points)

    def search_dense(self, vector, *, filter=None, limit=40, collection="chunks"):
        res = self._client.query_points(
            collection, query=vector, using="dense", query_filter=self._to_filter(filter), limit=limit
        )
        return [SearchHit(str(p.id), p.score, p.payload or {}) for p in res.points]

    def search_sparse(self, sparse, *, filter=None, limit=40, collection="chunks"):
        from qdrant_client import models as qm

        res = self._client.query_points(
            collection,
            query=qm.SparseVector(indices=list(sparse.keys()), values=list(sparse.values())),
            using="sparse",
            query_filter=self._to_filter(filter),
            limit=limit,
        )
        return [SearchHit(str(p.id), p.score, p.payload or {}) for p in res.points]

    def delete_by_doc(self, doc_id: str, collection: str = "chunks") -> int:
        from qdrant_client import models as qm

        self._client.delete(
            collection,
            points_selector=qm.Filter(must=[qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id))]),
        )
        return -1  # count not returned by delete

    def count(self, collection: str = "chunks") -> int:
        return self._client.count(collection).count

    def switch_alias(self, alias: str, target: str) -> None:
        from qdrant_client import models as qm

        self._client.update_collection_aliases(
            change_aliases_operations=[
                qm.CreateAliasOperation(
                    create_alias=qm.CreateAlias(collection_name=target, alias_name=alias)
                )
            ]
        )

    def get_alias_target(self, alias: str) -> str:
        for a in self._client.get_aliases().aliases:
            if a.alias_name == alias:
                return a.collection_name
        return alias

    def list_collections(self) -> list[str]:
        return [c.name for c in self._client.get_collections().collections]
