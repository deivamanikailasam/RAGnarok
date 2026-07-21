"""Blue/green reindex via collection aliasing (Step 12).

Changing the embedding model or chunker must never serve a half-rebuilt index. We build a new
concrete collection (``chunks_vN+1``) alongside the live one, validate it, then atomically repoint
the ``chunks`` alias. Rollback is repointing the alias — the old version is retained.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from ragnarok.stores.vector import VectorStore


def _next_version_name(alias: str, existing: list[str]) -> str:
    pattern = re.compile(rf"^{re.escape(alias)}_v(\d+)$")
    versions = [int(m.group(1)) for c in existing if (m := pattern.match(c))]
    nxt = (max(versions) + 1) if versions else 1
    return f"{alias}_v{nxt}"


class BlueGreenReindexer:
    def __init__(self, store: VectorStore, alias: str = "chunks") -> None:
        self.store = store
        self.alias = alias

    def current_target(self) -> str:
        return self.store.get_alias_target(self.alias)

    def reindex(
        self,
        build_into: Callable[[str], None],
        *,
        validate: Callable[[str], bool] | None = None,
    ) -> str:
        """Build a fresh collection, validate it, then swap the alias. Returns the new collection.

        On validation failure the alias is left untouched (old index keeps serving).
        """
        new_collection = _next_version_name(self.alias, self.store.list_collections())
        build_into(new_collection)  # ingest the corpus into the new collection
        if validate is not None and not validate(new_collection):
            raise RuntimeError(f"validation failed for {new_collection}; alias unchanged")
        self.store.switch_alias(self.alias, new_collection)
        return new_collection

    def rollback(self, to_collection: str) -> None:
        self.store.switch_alias(self.alias, to_collection)
