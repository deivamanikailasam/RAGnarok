"""Google Docs connector (Step 4).

Uses the Docs API to preserve *structure* — headings and, crucially, tables — instead of a flat
text export. A pricing/SLA table exported to flat text is nearly useless for retrieval; the API
gives structured rows we keep as table Blocks (Step 8 / Step 20).

The googleapiclient import is lazy so the package (and tests) load without Google libs installed.
ACLs are captured at read time (Step 4/22) — you can't retrofit "who may see this" reliably later.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from ragnarok.ingestion.models import Block, SourceDocument


def _heading_level(paragraph: dict[str, Any]) -> int:
    style = (paragraph.get("paragraphStyle") or {}).get("namedStyle", "")
    if style.startswith("HEADING_"):
        try:
            return int(style.split("_")[1])
        except (IndexError, ValueError):
            return 1
    if style == "TITLE":
        return 1
    return 0


def _para_text(element: dict[str, Any]) -> str:
    runs = (element.get("paragraph") or {}).get("elements", [])
    return "".join(r.get("textRun", {}).get("content", "") for r in runs).strip()


def _parse_table(table: dict[str, Any]) -> Block:
    rows: list[list[str]] = []
    for row in table.get("tableRows", []):
        cells = []
        for cell in row.get("tableCells", []):
            text = " ".join(
                _para_text(c) for c in cell.get("content", []) if "paragraph" in c
            ).strip()
            cells.append(text)
        rows.append(cells)
    headers = rows[0] if rows else []
    return Block(kind="table", headers=headers, rows=rows[1:] if len(rows) > 1 else [])


def _fetch_acls(doc_id: str) -> list[str]:  # pragma: no cover - requires Drive API
    try:
        from googleapiclient.discovery import build

        drive = build("drive", "v3")
        perms = drive.permissions().list(fileId=doc_id, fields="permissions(type,emailAddress,role)")
        result = perms.execute()
        tags = []
        for p in result.get("permissions", []):
            if p.get("type") == "domain":
                tags.append("domain:all")
            elif p.get("emailAddress"):
                tags.append(f"user:{p['emailAddress']}")
        return tags or ["restricted"]
    except Exception:
        return ["restricted"]


def extract_gdoc(doc_id: str, *, service: Any | None = None) -> SourceDocument:
    """Fetch a Google Doc into a structured SourceDocument.

    ``service`` may be injected (tests); otherwise the Docs API client is built lazily.
    """
    if service is None:  # pragma: no cover - requires google auth
        from googleapiclient.discovery import build

        service = build("docs", "v1")

    doc = service.documents().get(documentId=doc_id).execute()
    blocks: list[Block] = []
    for el in doc.get("body", {}).get("content", []):
        if "paragraph" in el:
            text = _para_text(el)
            if text:
                blocks.append(Block(kind="text", text=text, heading_level=_heading_level(el["paragraph"])))
        elif "table" in el:
            blocks.append(_parse_table(el["table"]))

    return SourceDocument(
        doc_id=f"gdoc://{doc_id}",
        uri=f"gdoc://{doc_id}",
        source_type="gdoc",
        title=doc.get("title", ""),
        blocks=blocks,
        acl_tags=_fetch_acls(doc_id),
        fetched_at=datetime.now(timezone.utc).isoformat(),
    ).with_hash()
