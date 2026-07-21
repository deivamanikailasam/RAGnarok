"""Tests for embedding generation (Step 9)."""

from __future__ import annotations

import ragnarok.cache as cache_mod
from ragnarok.ingestion.chunking import Chunk
from ragnarok.ingestion.embedding import EmbeddingClient, embed_chunks, get_embedding_client


def _chunk(text: str) -> Chunk:
    return Chunk(chunk_id="c" + str(abs(hash(text)) % 1000), doc_id="d", text=text,
                 contextual_prefix="Document: T.", token_count=5, content_hash=str(hash(text)))


def test_embed_produces_dense_and_sparse():
    client = EmbeddingClient(backend="local")
    res = client.embed_texts(["enterprise refund window is 30 days"])
    assert len(res) == 1
    assert len(res[0].dense) == 256
    assert len(res[0].sparse) > 0


def test_embedding_cache_avoids_recompute(monkeypatch):
    cache_mod.get_cache.cache_clear()
    client = EmbeddingClient(backend="local")

    calls = {"n": 0}
    orig = client.backend.embed

    def counting(texts):
        calls["n"] += 1
        return orig(texts)

    client.backend.embed = counting
    client.embed_texts(["hello world"])
    client.embed_texts(["hello world"])  # cache hit -> no second backend call
    assert calls["n"] == 1
    cache_mod.get_cache.cache_clear()


def test_embed_chunks_attaches_vectors_and_version():
    get_embedding_client.cache_clear()
    chunks = [_chunk("enterprise refund window"), _chunk("sso saml enablement")]
    embedded = embed_chunks(chunks)
    assert len(embedded) == 2
    for e in embedded:
        assert e.dense_vector and e.sparse_vector
        assert e.embedding_version
    get_embedding_client.cache_clear()
