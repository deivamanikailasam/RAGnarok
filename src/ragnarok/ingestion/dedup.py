"""Near-duplicate detection (Step 7).

Copy-pasted / re-exported docs bloat the index, waste embedding cost, and cause the
"five identical chunks" retrieval failure. SimHash over token shingles gives a 64-bit fingerprint;
near-duplicates have small Hamming distance. Pure-Python and deterministic (uses ``datasketch``
MinHash when available for larger corpora, but the SimHash path needs no dependency).
"""

from __future__ import annotations

import hashlib
import re
from typing import Optional

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _shingles(text: str, k: int = 3) -> list[str]:
    tokens = _TOKEN_RE.findall(text.lower())
    if len(tokens) < k:
        return tokens
    return [" ".join(tokens[i : i + k]) for i in range(len(tokens) - k + 1)]


def _hash64(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big")


def simhash(text: str) -> int:
    bits = [0] * 64
    for sh in _shingles(text) or [text]:
        h = _hash64(sh)
        for i in range(64):
            bits[i] += 1 if (h >> i) & 1 else -1
    out = 0
    for i in range(64):
        if bits[i] > 0:
            out |= 1 << i
    return out


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


class NearDupIndex:
    """Tracks fingerprints of seen docs; reports the id of a near-duplicate if one exists."""

    def __init__(self, max_distance: int = 3) -> None:
        self.max_distance = max_distance
        self._seen: list[tuple[str, int]] = []

    def find_duplicate(self, text: str) -> Optional[str]:
        fp = simhash(text)
        for doc_id, seen_fp in self._seen:
            if hamming(fp, seen_fp) <= self.max_distance:
                return doc_id
        return None

    def add(self, doc_id: str, text: str) -> None:
        self._seen.append((doc_id, simhash(text)))
