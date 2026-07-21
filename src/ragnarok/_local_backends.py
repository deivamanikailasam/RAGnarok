"""Deterministic, dependency-free local model backends (Step 2).

These are *fallback* implementations of the embedding and reranking models. In production the
embed/rerank service (``ragnarok.serving.model_services``) uses BGE-M3 / BGE-reranker on a GPU;
but a fully-local system must also be runnable and testable on a laptop with no GPU and no model
weights. These backends provide that: they are pure-Python, deterministic, and good enough that
retrieval/rerank behave sensibly in tests and demos (lexical overlap dominates, which is the
correct qualitative behavior for exact-term queries).

They are intentionally NOT a quality substitute for real models — the provider abstraction
(Step 3) points at the real service in any real deployment. Selection is by config/env, never
hardcoded, so swapping to real models is a one-line change.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _hash_index(token: str, dim: int) -> int:
    h = hashlib.blake2b(token.encode(), digest_size=8).digest()
    return int.from_bytes(h, "big") % dim


class DeterministicEmbedder:
    """Hashing embedder producing L2-normalized dense vectors + tf-weighted sparse vectors.

    Cosine similarity between two texts rises with shared vocabulary, so semantically/lexically
    related chunks score higher — enough for the pipeline to exhibit correct behavior offline.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def embed(self, text: str) -> tuple[list[float], dict[int, float]]:
        tokens = tokenize(text)
        dense = [0.0] * self.dim
        counts = Counter(tokens)
        for tok, tf in counts.items():
            idx = _hash_index(tok, self.dim)
            # signed hashing keeps dimensions from all-positive collapse
            sign = 1.0 if _hash_index(tok + "#", 2) == 0 else -1.0
            dense[idx] += sign * (1.0 + math.log(tf))
        norm = math.sqrt(sum(v * v for v in dense)) or 1.0
        dense = [v / norm for v in dense]
        # sparse: idf-agnostic tf weights keyed by a large hashed vocabulary space
        sparse = {_hash_index(tok, 2**20): 1.0 + math.log(tf) for tok, tf in counts.items()}
        return dense, sparse

    def embed_many(self, texts: list[str]) -> list[tuple[list[float], dict[int, float]]]:
        return [self.embed(t) for t in texts]


class LexicalReranker:
    """Cross-encoder stand-in: scores a (query, passage) pair by weighted lexical overlap.

    Approximates what a real cross-encoder rewards — query terms actually present in the passage —
    so the broad-retrieve / narrow-rerank behavior (Step 17) is observable without a GPU.
    """

    def score(self, query: str, passage: str) -> float:
        q = Counter(tokenize(query))
        p = Counter(tokenize(passage))
        if not q or not p:
            return 0.0
        overlap = sum(min(q[t], p[t]) for t in q)
        # normalize by query length; light length penalty to avoid rewarding huge passages
        coverage = overlap / sum(q.values())
        length_penalty = 1.0 / (1.0 + math.log1p(len(p)) / 10.0)
        return round(coverage * length_penalty, 6)

    def score_many(self, query: str, passages: list[str]) -> list[float]:
        return [self.score(query, p) for p in passages]
