"""Tests for content enrichment (Step 6)."""

from __future__ import annotations

import json
from pathlib import Path

from ragnarok.ingestion.connectors.files import load_file
from ragnarok.ingestion.enrich import enrich
from ragnarok.ingestion.normalize import normalize
from ragnarok.providers import FakeLLM, reset_clients, set_role_client

SAMPLE = Path(__file__).resolve().parents[1] / "datasets" / "sample"

_FAKE_ENRICHMENT = {
    "doc_summary": "Refund policy with tier-dependent windows.",
    "section_summaries": [{"heading": "Enterprise Exceptions", "summary": "Enterprise 30-day window."}],
    "table_descriptions": [
        {"section_heading": "Refund Windows by Tier",
         "description": "Refund windows by tier: enterprise annual is 30 days, consumer monthly is 14 days."}
    ],
    "custom_metadata": {
        "topics": ["refunds", "billing"],
        "entities": ["Enterprise", "Consumer", "Pro"],
        "doc_type": "policy",
        "audience": ["enterprise", "pro", "consumer"],
        "freshness_date": "2026-03-01",
        "authority": "official",
        "access_tags": [],
    },
    "keywords": ["refund", "enterprise", "annual"],
}


async def test_enrich_produces_metadata_and_table_desc():
    reset_clients()
    set_role_client("llm_large", FakeLLM(response=json.dumps(_FAKE_ENRICHMENT)))

    norm = normalize(load_file(SAMPLE / "refund_policy.md"))
    enriched = await enrich(norm)

    assert enriched.enrichment.custom_metadata.doc_type == "policy"
    assert enriched.enrichment.custom_metadata.authority == "official"
    assert "enterprise" in enriched.enrichment.custom_metadata.audience
    # table description is retrievable text mentioning the exact fact
    assert "30 days" in enriched.table_desc_for("Refund Windows by Tier")
    # ACLs flow into access_tags when the LLM leaves them empty (security default)
    assert enriched.enrichment.custom_metadata.access_tags == norm.acl_tags
    assert enriched.enrichment_version.startswith("content_enricher@v")
    reset_clients()


async def test_enrich_summary_fallback_to_doc_summary():
    reset_clients()
    set_role_client("llm_large", FakeLLM(response=json.dumps(_FAKE_ENRICHMENT)))
    enriched = await enrich(normalize(load_file(SAMPLE / "refund_policy.md")))
    # a heading with no explicit section summary falls back to the doc summary
    assert enriched.summary_for("Nonexistent Heading") == "Refund policy with tier-dependent windows."
    reset_clients()
