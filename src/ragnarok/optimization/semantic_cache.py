"""Semantic response cache (Step 28).

The exact response cache (Step 27) only hits on identical normalized strings. Real users paraphrase
("enterprise refund window?" vs "how long can enterprise customers get a refund?"). This cache keys
on the query *embedding* and serves a stored answer when cosine similarity to a prior query is high
AND the caller's entitlements match (never cross-tenant/tier). Big latency/cost win on the long tail
of paraphrased repeats, at the cost of one cheap (cached) query embedding.

In-memory ring buffer by default (reference + tests); a Qdrant-backed variant scales it in prod.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


@dataclass
class _Entry:
    embedding: list[float]
    entitlements_key: str
    payload: Any


class SemanticResponseCache:
    def __init__(self, *, threshold: float = 0.90, max_entries: int = 2000) -> None:
        self.threshold = threshold
        self._entries: deque[_Entry] = deque(maxlen=max_entries)

    @staticmethod
    def _key(entitlements: list[str]) -> str:
        return ",".join(sorted(entitlements))

    def lookup(self, embedding: list[float], entitlements: list[str]) -> Any | None:
        key = self._key(entitlements)
        best: tuple[float, Any] | None = None
        for e in self._entries:
            if e.entitlements_key != key:  # never serve across entitlement scopes (security)
                continue
            sim = _cosine(embedding, e.embedding)
            if sim >= self.threshold and (best is None or sim > best[0]):
                best = (sim, e.payload)
        return best[1] if best else None

    def store(self, embedding: list[float], entitlements: list[str], payload: Any) -> None:
        self._entries.append(_Entry(embedding, self._key(entitlements), payload))

    def clear(self) -> None:
        self._entries.clear()


_cache: SemanticResponseCache | None = None


def get_semantic_cache() -> SemanticResponseCache:
    global _cache
    if _cache is None:
        from ragnarok.config import get_settings

        cfg = get_settings().optimization
        _cache = SemanticResponseCache(
            threshold=cfg.semantic_cache_threshold, max_entries=cfg.semantic_cache_max_entries
        )
    return _cache


def reset_semantic_cache() -> None:
    global _cache
    _cache = None
