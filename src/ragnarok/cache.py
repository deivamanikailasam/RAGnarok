"""Version-namespaced cache used across the system (Step 7, reused by Steps 9/14/20/27).

One abstraction, two backends: in-memory (tests/dev) and Redis (prod). Keys are namespaced by a
*version* (model/prompt/index version) so a version bump invalidates correctly — a stale cache hit
is worse than a miss. Backend chosen by ``RAGNAROK_CACHE`` env (``memory`` default), never hardcoded.
"""

from __future__ import annotations

import json
import os
import time
from functools import lru_cache
from typing import Any, Protocol


def make_key(namespace: str, version: str, *parts: Any) -> str:
    tail = ":".join(str(p) for p in parts)
    return f"{namespace}:{version}:{tail}"


class Cache(Protocol):
    def get(self, key: str) -> str | None: ...
    def set(self, key: str, value: str, ttl_s: int | None = None) -> None: ...
    def delete_namespace(self, prefix: str) -> int: ...


class InMemoryCache:
    def __init__(self) -> None:
        self._db: dict[str, tuple[str, float | None]] = {}
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> str | None:
        item = self._db.get(key)
        if item is None:
            self.misses += 1
            return None
        value, expiry = item
        if expiry is not None and expiry < time.monotonic():
            self._db.pop(key, None)
            self.misses += 1
            return None
        self.hits += 1
        return value

    def set(self, key: str, value: str, ttl_s: int | None = None) -> None:
        expiry = time.monotonic() + ttl_s if ttl_s else None
        self._db[key] = (value, expiry)

    def delete_namespace(self, prefix: str) -> int:
        keys = [k for k in self._db if k.startswith(prefix)]
        for k in keys:
            self._db.pop(k, None)
        return len(keys)


class RedisCache:  # pragma: no cover - requires a running Redis
    def __init__(self, url: str) -> None:
        import redis

        self._r = redis.from_url(url, decode_responses=True)

    def get(self, key: str) -> str | None:
        return self._r.get(key)

    def set(self, key: str, value: str, ttl_s: int | None = None) -> None:
        self._r.set(key, value, ex=ttl_s)

    def delete_namespace(self, prefix: str) -> int:
        deleted = 0
        for key in self._r.scan_iter(match=prefix + "*"):
            self._r.delete(key)
            deleted += 1
        return deleted


# JSON convenience -----------------------------------------------------------


def get_json(cache: Cache, key: str) -> Any | None:
    raw = cache.get(key)
    return json.loads(raw) if raw is not None else None


def set_json(cache: Cache, key: str, value: Any, ttl_s: int | None = None) -> None:
    cache.set(key, json.dumps(value, default=str), ttl_s)


@lru_cache
def get_cache() -> Cache:
    backend = os.environ.get("RAGNAROK_CACHE", "memory").lower()
    if backend == "redis":
        from ragnarok.config import get_settings

        return RedisCache(get_settings().stores.redis_url)
    return InMemoryCache()
