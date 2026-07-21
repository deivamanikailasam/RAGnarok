"""Tests for the cache, dedup, and enrichment caching (Step 7)."""

from __future__ import annotations

import json
from pathlib import Path

from ragnarok.cache import InMemoryCache, get_json, make_key, set_json
from ragnarok.ingestion.connectors.files import load_file
from ragnarok.ingestion.dedup import NearDupIndex, hamming, simhash
from ragnarok.ingestion.enrich import enrich
from ragnarok.ingestion.normalize import normalize
from ragnarok.ingestion.pipeline import IngestionPipeline
from ragnarok.ingestion.registry import InMemoryRegistry
from ragnarok.providers import FakeLLM, reset_clients, set_role_client

SAMPLE = Path(__file__).resolve().parents[1] / "datasets" / "sample"


def test_cache_versioned_keys_and_ttl_expiry():
    c = InMemoryCache()
    set_json(c, make_key("emb", "v1", "abc"), {"x": 1})
    assert get_json(c, make_key("emb", "v1", "abc")) == {"x": 1}
    assert get_json(c, make_key("emb", "v2", "abc")) is None  # version bump invalidates
    assert c.delete_namespace("emb:v1:") == 1


def test_simhash_near_duplicate_detection():
    a = "The enterprise refund window is 30 days from the invoice date."
    b = "The enterprise refund window is 30 days from the invoice date!!"  # trivially changed
    c = "Kubernetes autoscaling uses horizontal pod metrics."
    assert hamming(simhash(a), simhash(b)) <= 3
    assert hamming(simhash(a), simhash(c)) > 3

    idx = NearDupIndex(max_distance=3)
    idx.add("doc-a", a)
    assert idx.find_duplicate(b) == "doc-a"
    assert idx.find_duplicate(c) is None


def test_pipeline_skips_near_duplicates(tmp_path):
    # two files with essentially identical content
    (tmp_path / "one.md").write_text("# Policy\nEnterprise refund window is 30 days from invoice.")
    (tmp_path / "two.md").write_text("# Policy\nEnterprise refund window is 30 days from invoice!!")
    pipe = IngestionPipeline(registry=InMemoryRegistry())
    summary = pipe.run([str(tmp_path)])
    assert summary.processed == 1
    assert summary.duplicates == 1


async def test_enrichment_is_cached(monkeypatch):
    reset_clients()
    monkeypatch.setenv("RAGNAROK_CACHE", "memory")
    # fresh cache instance
    import ragnarok.cache as cache_mod

    cache_mod.get_cache.cache_clear()

    calls = {"n": 0}

    def handler(messages, schema):
        calls["n"] += 1
        return json.dumps({"doc_summary": "s", "custom_metadata": {"doc_type": "policy"}})

    set_role_client("llm_large", FakeLLM(handler=handler))
    norm = normalize(load_file(SAMPLE / "sso_runbook.md"))
    await enrich(norm)
    await enrich(norm)  # identical content -> cache hit, no second LLM call
    assert calls["n"] == 1
    reset_clients()
    cache_mod.get_cache.cache_clear()
