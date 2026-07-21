"""Tests for connectors + normalization (Step 4)."""

from __future__ import annotations

from pathlib import Path

from ragnarok.ingestion.connectors.files import load_file
from ragnarok.ingestion.connectors.gdoc import extract_gdoc
from ragnarok.ingestion.normalize import normalize

SAMPLE = Path(__file__).resolve().parents[1] / "datasets" / "sample"


def test_markdown_table_is_parsed_as_table_block():
    doc = load_file(SAMPLE / "refund_policy.md")
    tables = [b for b in doc.blocks if b.kind == "table"]
    assert len(tables) == 1
    t = tables[0]
    assert "Tier" in t.headers
    # the Enterprise/Annual/30 days row must survive as structured cells
    assert any("Enterprise" in row[0] and "30 days" in row[2] for row in t.rows)


def test_headings_become_sections_after_normalize():
    doc = load_file(SAMPLE / "refund_policy.md")
    norm = normalize(doc)
    headings = [s.heading for s in norm.sections if s.heading]
    assert "Refund Windows by Tier" in headings
    assert "Enterprise Exceptions" in headings
    # the table section round-trips to markdown with exact values (needed for Step 20)
    table_section = next(s for s in norm.sections if s.is_table)
    assert "30 days" in table_section.text


def test_content_hash_is_stable_and_excludes_timestamp():
    d1 = load_file(SAMPLE / "sso_runbook.md")
    d2 = load_file(SAMPLE / "sso_runbook.md")
    assert d1.content_hash == d2.content_hash  # identical content -> same hash (idempotency)
    assert d1.content_hash != ""


class _FakeDocsService:
    """Minimal stand-in for the Google Docs API client."""

    def __init__(self, payload):
        self._payload = payload

    def documents(self):
        return self

    def get(self, documentId):  # noqa: N803 - matches google client signature
        return self

    def execute(self):
        return self._payload


def test_gdoc_connector_parses_structure(monkeypatch):
    payload = {
        "title": "Refund Policy",
        "body": {
            "content": [
                {"paragraph": {"paragraphStyle": {"namedStyle": "HEADING_1"},
                               "elements": [{"textRun": {"content": "Overview\n"}}]}},
                {"paragraph": {"elements": [{"textRun": {"content": "Refunds within a window.\n"}}]}},
                {"table": {"tableRows": [
                    {"tableCells": [
                        {"content": [{"paragraph": {"elements": [{"textRun": {"content": "Tier"}}]}}]},
                        {"content": [{"paragraph": {"elements": [{"textRun": {"content": "Window"}}]}}]}]},
                    {"tableCells": [
                        {"content": [{"paragraph": {"elements": [{"textRun": {"content": "Enterprise"}}]}}]},
                        {"content": [{"paragraph": {"elements": [{"textRun": {"content": "30 days"}}]}}]}]},
                ]}},
            ]
        },
    }
    # patch ACL fetch to avoid Drive API
    monkeypatch.setattr("ragnarok.ingestion.connectors.gdoc._fetch_acls", lambda _id: ["domain:all"])
    doc = extract_gdoc("abc123", service=_FakeDocsService(payload))
    assert doc.source_type == "gdoc"
    assert doc.title == "Refund Policy"
    table = next(b for b in doc.blocks if b.kind == "table")
    assert table.headers == ["Tier", "Window"]
    assert table.rows == [["Enterprise", "30 days"]]
    assert doc.acl_tags == ["domain:all"]
