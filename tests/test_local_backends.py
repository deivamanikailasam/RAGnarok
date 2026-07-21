"""Tests for the deterministic local model backends (Step 2)."""

from __future__ import annotations

import math

from ragnarok._local_backends import DeterministicEmbedder, LexicalReranker


def _cosine(a, b):
    return sum(x * y for x, y in zip(a, b)) / (
        (math.sqrt(sum(x * x for x in a)) or 1) * (math.sqrt(sum(y * y for y in b)) or 1)
    )


def test_embedder_is_deterministic_and_normalized():
    emb = DeterministicEmbedder(dim=128)
    d1, s1 = emb.embed("enterprise refund window policy")
    d2, _ = emb.embed("enterprise refund window policy")
    assert d1 == d2  # deterministic
    assert abs(math.sqrt(sum(v * v for v in d1)) - 1.0) < 1e-6  # L2-normalized
    assert len(s1) > 0  # sparse terms present


def test_related_texts_score_higher_than_unrelated():
    emb = DeterministicEmbedder(dim=256)
    q, _ = emb.embed("refund window for enterprise customers")
    related, _ = emb.embed("enterprise customers refund window is 30 days")
    unrelated, _ = emb.embed("kubernetes pod autoscaling metrics")
    assert _cosine(q, related) > _cosine(q, unrelated)


def test_reranker_rewards_query_term_overlap():
    rr = LexicalReranker()
    query = "how long is the enterprise refund window"
    good = rr.score(query, "The enterprise refund window is 30 days.")
    bad = rr.score(query, "SSO can be enabled from the admin console.")
    assert good > bad
    assert 0.0 <= good <= 1.0
