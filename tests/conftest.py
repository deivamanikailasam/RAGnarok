"""Shared fixtures: a populated in-memory index built from the sample corpus (Steps 16-21).

Uses the deterministic local embedder (no GPU) and a FakeLLM enricher that returns sensible,
content-aware metadata so retrieval/generation tests exercise the real pipeline offline.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ragnarok.ingestion.embedding import get_embedding_client
from ragnarok.ingestion.registry import InMemoryRegistry
from ragnarok.ingestion.wiring import build_pipeline
from ragnarok.providers import FakeLLM, reset_clients, set_role_client
from ragnarok.stores.features import InMemoryFeatureStore
from ragnarok.stores.vector import InMemoryVectorStore

SAMPLE = str(Path(__file__).resolve().parents[1] / "datasets" / "sample")


def _enrichment_for(messages) -> str:
    text = json.dumps(messages).lower()
    if "sso" in text or "saml" in text:
        return json.dumps({
            "doc_summary": "How to enable SSO/SAML by plan.",
            "section_summaries": [],
            "table_descriptions": [],
            "custom_metadata": {"doc_type": "runbook", "audience": ["enterprise", "growth"],
                                "authority": "official", "freshness_date": "2026-02-01",
                                "topics": ["sso", "saml"], "entities": ["Growth", "Enterprise"],
                                "access_tags": ["public"]},
            "keywords": ["sso", "saml", "scim"],
        })
    return json.dumps({
        "doc_summary": "Refund policy with tier-dependent windows.",
        "section_summaries": [{"heading": "Enterprise Exceptions", "summary": "Enterprise 30-day window."}],
        "table_descriptions": [{"section_heading": "Refund Windows by Tier",
                                "description": "Refund windows by tier: enterprise annual 30 days, consumer monthly 14 days."}],
        "custom_metadata": {"doc_type": "policy", "audience": ["enterprise", "pro", "consumer"],
                            "authority": "official", "freshness_date": "2026-03-01",
                            "topics": ["refunds", "billing"], "entities": ["Enterprise", "Consumer"],
                            "access_tags": ["public"]},
        "keywords": ["refund", "enterprise", "annual"],
    })


@pytest.fixture
def sample_index():
    import ragnarok.cache as cache_mod
    from ragnarok.optimization.semantic_cache import reset_semantic_cache

    cache_mod.get_cache.cache_clear()
    get_embedding_client.cache_clear()
    reset_semantic_cache()
    reset_clients()
    set_role_client("llm_large", FakeLLM(handler=lambda m, s: _enrichment_for(m)))

    store = InMemoryVectorStore()
    features = InMemoryFeatureStore()
    pipe = build_pipeline(registry=InMemoryRegistry(), store=store, feature_store=features)
    summary = pipe.run([SAMPLE])
    assert summary.processed >= 2

    yield store, features

    reset_clients()
    cache_mod.get_cache.cache_clear()
    get_embedding_client.cache_clear()
    reset_semantic_cache()
