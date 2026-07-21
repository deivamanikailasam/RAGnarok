"""Local file connectors (Step 4).

Markdown/text/HTML loaders that preserve heading hierarchy and parse Markdown tables into table
Blocks. Same ``SourceDocument`` output as the GDoc connector, so an air-gapped install uses the
identical downstream pipeline. PDF/Office loaders (unstructured/PyMuPDF) plug in here the same way.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

from ragnarok.ingestion.models import Block, SourceDocument

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*$")
_IMAGE_RE = re.compile(r"^!\[([^\]]*)\]\(([^)]+)\)\s*$")  # ![alt](src)  (Step 36)

TEXT_SUFFIXES = {".md", ".markdown", ".txt"}


def _split_row(line: str) -> list[str]:
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    return cells


def _parse_markdown(text: str) -> list[Block]:
    lines = text.splitlines()
    blocks: list[Block] = []
    buf: list[str] = []

    def flush_text() -> None:
        if buf:
            joined = "\n".join(buf).strip()
            if joined:
                blocks.append(Block(kind="text", text=joined))
            buf.clear()

    i = 0
    while i < len(lines):
        line = lines[i]
        heading = _HEADING_RE.match(line)
        # table: a header row followed by a separator row
        is_table_start = (
            "|" in line
            and i + 1 < len(lines)
            and _TABLE_SEP_RE.match(lines[i + 1])
        )
        image = _IMAGE_RE.match(line.strip())
        if heading:
            flush_text()
            blocks.append(Block(kind="text", text=heading.group(2).strip(), heading_level=len(heading.group(1))))
            i += 1
        elif image:
            flush_text()
            blocks.append(Block(kind="image", text=image.group(1).strip(), src=image.group(2).strip()))
            i += 1
        elif is_table_start:
            flush_text()
            headers = _split_row(line)
            rows = []
            i += 2  # skip header + separator
            while i < len(lines) and "|" in lines[i] and lines[i].strip():
                rows.append(_split_row(lines[i]))
                i += 1
            blocks.append(Block(kind="table", headers=headers, rows=rows))
        else:
            buf.append(line)
            i += 1
    flush_text()
    return blocks


def _strip_html(html: str) -> str:
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", html)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_file(path: Path, *, acl_tags: list[str] | None = None) -> SourceDocument:
    path = Path(path)
    raw = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() in {".html", ".htm"}:
        blocks = [Block(kind="text", text=_strip_html(raw))]
    else:
        blocks = _parse_markdown(raw)
    return SourceDocument(
        doc_id=f"file://{path.resolve()}",
        uri=f"file://{path.resolve()}",
        source_type="file",
        title=path.stem.replace("_", " ").replace("-", " ").strip(),
        blocks=blocks,
        acl_tags=acl_tags or ["public"],
        fetched_at=datetime.now(timezone.utc).isoformat(),
    ).with_hash()


def load_path(path: Path) -> list[SourceDocument]:
    path = Path(path)
    if path.is_file():
        return [load_file(path)]
    docs = []
    for p in sorted(path.rglob("*")):
        if p.is_file() and p.suffix.lower() in TEXT_SUFFIXES | {".html", ".htm"}:
            docs.append(load_file(p))
    return docs
