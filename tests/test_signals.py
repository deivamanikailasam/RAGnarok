"""Tests for business ranking signals (Step 13)."""

from __future__ import annotations

from ragnarok.retrieval.signals import (
    authority_boost,
    business_score,
    is_deprecated,
    recency_boost,
)
from ragnarok.stores.features import DocumentFeatures


def test_recency_boost_monotonic():
    assert recency_boost(0) > recency_boost(180) > recency_boost(3650)
    assert 0.0 <= recency_boost(10_000) <= 1.0


def test_authority_ordering():
    assert authority_boost("official") > authority_boost("draft") > authority_boost("deprecated")
    assert is_deprecated("deprecated") and not is_deprecated("official")


def test_current_official_beats_deprecated_on_equal_relevance():
    rel = 0.7  # identical semantic relevance
    current = business_score(rel, DocumentFeatures("a", authority="official", freshness_days=10))
    stale = business_score(rel, DocumentFeatures("b", authority="deprecated", freshness_days=1200))
    assert current > stale


def test_signals_are_bounded_and_do_not_override_relevance():
    # a much more relevant deprecated doc should still beat a barely-relevant official one
    strong_deprecated = business_score(0.95, DocumentFeatures("a", authority="deprecated", freshness_days=1000))
    weak_official = business_score(0.30, DocumentFeatures("b", authority="official", freshness_days=1))
    assert strong_deprecated > weak_official
