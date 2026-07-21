"""Token counting (Steps 8, 19).

Chunk sizes and the generation context budget must be measured in *tokens*, not characters, so they
are exact for the model that will read them. Uses tiktoken when available; falls back to a
word-based estimate (~1.3 tokens/word) so the package works without the optional dependency.
Counts are cheap and cached per string.
"""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def _encoder():  # pragma: no cover - depends on optional tiktoken
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


@lru_cache(maxsize=8192)
def count_tokens(text: str) -> int:
    enc = _encoder()
    if enc is not None:  # pragma: no cover - optional path
        return len(enc.encode(text))
    # fallback estimate: words * 1.3, rounded up
    words = len(text.split())
    return int(words * 1.3) + 1
