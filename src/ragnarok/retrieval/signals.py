"""Business ranking signals (Step 13; applied in Step 18).

Two passages can be equally relevant but one is the *current, official* policy and the other a
deprecated draft. Freshness/authority/popularity encode "which source should win". These are a
BOUNDED adjustment on top of semantic relevance (the reranker score stays primary) — they break
ties and demote stale content, they never override relevance. This prevents "popular-but-wrong"
from beating "relevant-and-correct".
"""

from __future__ import annotations

import math

from ragnarok.stores.features import DocumentFeatures

# Weights are intentionally small so relevance dominates. Tunable by eval (Step 24).
W_RECENCY = 0.05
W_AUTHORITY = 0.05
W_POPULARITY = 0.03
DEPRECATED_PENALTY = 0.20

_AUTHORITY = {"official": 1.0, "draft": 0.4, "deprecated": 0.0}


def recency_boost(freshness_days: int) -> float:
    """~1.0 for fresh docs, decaying over ~a year. Always in [0, 1]."""
    return math.exp(-max(freshness_days, 0) / 365.0)


def authority_boost(authority: str) -> float:
    return _AUTHORITY.get(authority, 0.3)


def is_deprecated(authority: str) -> bool:
    return authority == "deprecated"


def business_score(rerank_score: float, feats: DocumentFeatures) -> float:
    """Combine relevance with bounded business signals. Deprecated docs are hard-demoted."""
    score = rerank_score
    score += W_RECENCY * recency_boost(feats.freshness_days)
    score += W_AUTHORITY * authority_boost(feats.authority)
    score += W_POPULARITY * max(min(feats.popularity + feats.feedback_score, 1.0), 0.0)
    if is_deprecated(feats.authority):
        score -= DEPRECATED_PENALTY
    return score
