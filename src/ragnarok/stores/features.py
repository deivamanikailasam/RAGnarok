"""Feature store (Step 11).

Document-level signals that change over time (authority, freshness, popularity, feedback) and are
reused across stages (source routing Step 14, reranking Step 18, eval Step 24). Storing these in a
feature store gives O(docs) updates and one definition reused everywhere — duplicating them into
every chunk payload would be O(chunks) and race with search.

Backends behind one interface: InMemory (dev/CI) and Feast (prod: Postgres offline + Redis online).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Protocol


@dataclass
class DocumentFeatures:
    doc_id: str
    authority: str = "draft"  # official | draft | deprecated
    freshness_days: int = 9999  # days since the doc's effective date (lower = fresher)
    popularity: float = 0.0  # normalized usage
    feedback_score: float = 0.0  # rolling avg of thumbs feedback, [-1, 1]
    doc_type: str = "other"


def _days_since(iso_date: str | None) -> int:
    if not iso_date:
        return 9999
    try:
        d = datetime.fromisoformat(iso_date).date()
        return max((date.today() - d).days, 0)
    except ValueError:
        return 9999


class FeatureStore(Protocol):
    def upsert_document(self, features: DocumentFeatures) -> None: ...
    def get_online(self, doc_ids: list[str]) -> dict[str, DocumentFeatures]: ...
    def record_feedback(self, doc_id: str, delta: float) -> None: ...
    def record_hit(self, doc_id: str) -> None: ...


class InMemoryFeatureStore:
    def __init__(self) -> None:
        self._db: dict[str, DocumentFeatures] = {}
        self._hits: dict[str, int] = {}

    def upsert_document(self, features: DocumentFeatures) -> None:
        existing = self._db.get(features.doc_id)
        if existing:  # preserve accumulated usage/feedback across re-ingests
            features.popularity = existing.popularity
            features.feedback_score = existing.feedback_score
        self._db[features.doc_id] = features

    def get_online(self, doc_ids: list[str]) -> dict[str, DocumentFeatures]:
        return {d: self._db.get(d, DocumentFeatures(doc_id=d)) for d in doc_ids}

    def record_feedback(self, doc_id: str, delta: float) -> None:
        f = self._db.setdefault(doc_id, DocumentFeatures(doc_id=doc_id))
        # exponential moving average, clamped
        f.feedback_score = max(-1.0, min(1.0, 0.8 * f.feedback_score + 0.2 * delta))

    def record_hit(self, doc_id: str) -> None:
        self._hits[doc_id] = self._hits.get(doc_id, 0) + 1
        total = max(sum(self._hits.values()), 1)
        f = self._db.setdefault(doc_id, DocumentFeatures(doc_id=doc_id))
        f.popularity = self._hits[doc_id] / total


class FeastFeatureStore:  # pragma: no cover - requires Feast + Postgres/Redis
    def __init__(self, repo_path: str) -> None:
        from feast import FeatureStore as _Feast

        self._store = _Feast(repo_path=repo_path)
        self._fallback = InMemoryFeatureStore()  # for write-through / usage counters

    def upsert_document(self, features: DocumentFeatures) -> None:
        self._fallback.upsert_document(features)  # materialized to Feast via batch job

    def get_online(self, doc_ids: list[str]) -> dict[str, DocumentFeatures]:
        rows = self._store.get_online_features(
            features=[
                "document_stats:authority",
                "document_stats:freshness_days",
                "document_stats:popularity",
                "document_stats:feedback_score",
                "document_stats:doc_type",
            ],
            entity_rows=[{"doc_id": d} for d in doc_ids],
        ).to_dict()
        out = {}
        for i, d in enumerate(doc_ids):
            out[d] = DocumentFeatures(
                doc_id=d,
                authority=rows["authority"][i] or "draft",
                freshness_days=rows["freshness_days"][i] or 9999,
                popularity=rows["popularity"][i] or 0.0,
                feedback_score=rows["feedback_score"][i] or 0.0,
                doc_type=rows["doc_type"][i] or "other",
            )
        return out

    def record_feedback(self, doc_id: str, delta: float) -> None:
        self._fallback.record_feedback(doc_id, delta)

    def record_hit(self, doc_id: str) -> None:
        self._fallback.record_hit(doc_id)


def features_from_enriched(doc) -> DocumentFeatures:
    """Build the document-level feature row from an EnrichedDocument (Step 6 output)."""
    m = doc.enrichment.custom_metadata
    return DocumentFeatures(
        doc_id=doc.doc_id,
        authority=m.authority,
        freshness_days=_days_since(m.freshness_date),
        doc_type=m.doc_type,
    )
