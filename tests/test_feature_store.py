"""Tests for the feature store (Step 11)."""

from __future__ import annotations

from ragnarok.stores.features import DocumentFeatures, InMemoryFeatureStore, features_from_enriched


class _Meta:
    authority = "official"
    freshness_date = "2026-01-01"
    doc_type = "policy"


class _Enr:
    class enrichment:  # noqa: N801
        class custom_metadata:  # noqa: N801
            authority = "official"
            freshness_date = "2026-01-01"
            doc_type = "policy"

    doc_id = "gdoc://x"


def test_features_from_enriched_maps_fields():
    f = features_from_enriched(_Enr())
    assert f.authority == "official"
    assert f.doc_type == "policy"
    assert f.freshness_days >= 0


def test_online_lookup_returns_defaults_for_unknown():
    fs = InMemoryFeatureStore()
    feats = fs.get_online(["unknown"])
    assert feats["unknown"].authority == "draft"  # safe default


def test_feedback_and_popularity_accumulate_and_survive_reingest():
    fs = InMemoryFeatureStore()
    fs.upsert_document(DocumentFeatures(doc_id="d", authority="official"))
    fs.record_feedback("d", 1.0)
    fs.record_hit("d")
    before = fs.get_online(["d"])["d"]
    assert before.feedback_score > 0
    assert before.popularity > 0
    # re-ingest (upsert) must preserve accumulated usage/feedback
    fs.upsert_document(DocumentFeatures(doc_id="d", authority="official", freshness_days=1))
    after = fs.get_online(["d"])["d"]
    assert after.feedback_score == before.feedback_score
    assert after.popularity == before.popularity
