"""Document registry for idempotent, incremental ingestion (Step 5).

Tracks every document's content hash, versions, and status so a nightly re-ingest of an unchanged
corpus is ~100% skips (hash-diff), not a full re-embed. Two backends: in-memory (tests) and SQLite
(local default; Postgres in prod via the same SQL). The registry is the system of record that
drives incremental work and, later, cascade deletes (Step 12).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol


@dataclass
class DocRecord:
    doc_id: str
    content_hash: str
    status: str = "pending"  # pending | ingested | failed
    enrichment_version: str = ""
    embedding_version: str = ""


class Registry(Protocol):
    def get(self, doc_id: str) -> Optional[DocRecord]: ...
    def upsert(self, record: DocRecord) -> None: ...
    def delete(self, doc_id: str) -> None: ...
    def all_ids(self) -> list[str]: ...

    def is_unchanged(self, doc_id: str, content_hash: str) -> bool:
        rec = self.get(doc_id)
        return rec is not None and rec.content_hash == content_hash and rec.status == "ingested"


class InMemoryRegistry:
    def __init__(self) -> None:
        self._db: dict[str, DocRecord] = {}

    def get(self, doc_id: str) -> Optional[DocRecord]:
        return self._db.get(doc_id)

    def upsert(self, record: DocRecord) -> None:
        self._db[record.doc_id] = record

    def delete(self, doc_id: str) -> None:
        self._db.pop(doc_id, None)

    def all_ids(self) -> list[str]:
        return list(self._db)

    is_unchanged = Registry.is_unchanged


class SqliteRegistry:
    def __init__(self, path: str | Path = ".ragnarok/registry.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS documents (
                   doc_id TEXT PRIMARY KEY,
                   content_hash TEXT NOT NULL,
                   status TEXT NOT NULL,
                   enrichment_version TEXT,
                   embedding_version TEXT
               )"""
        )
        self._conn.commit()

    def get(self, doc_id: str) -> Optional[DocRecord]:
        row = self._conn.execute(
            "SELECT doc_id, content_hash, status, enrichment_version, embedding_version "
            "FROM documents WHERE doc_id = ?",
            (doc_id,),
        ).fetchone()
        return DocRecord(*row) if row else None

    def upsert(self, record: DocRecord) -> None:
        self._conn.execute(
            """INSERT INTO documents VALUES (?,?,?,?,?)
               ON CONFLICT(doc_id) DO UPDATE SET
                 content_hash=excluded.content_hash, status=excluded.status,
                 enrichment_version=excluded.enrichment_version,
                 embedding_version=excluded.embedding_version""",
            (
                record.doc_id,
                record.content_hash,
                record.status,
                record.enrichment_version,
                record.embedding_version,
            ),
        )
        self._conn.commit()

    def delete(self, doc_id: str) -> None:
        self._conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
        self._conn.commit()

    def all_ids(self) -> list[str]:
        return [r[0] for r in self._conn.execute("SELECT doc_id FROM documents").fetchall()]

    is_unchanged = Registry.is_unchanged
