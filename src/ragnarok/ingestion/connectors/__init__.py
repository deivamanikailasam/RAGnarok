"""Source connectors. Every connector emits a ``SourceDocument`` (Step 4).

The rest of the pipeline is connector-agnostic, so adding a source (SharePoint, a DB, ...) is a new
connector, not a pipeline change.
"""

from __future__ import annotations

from pathlib import Path

from ragnarok.ingestion.models import SourceDocument

from .files import load_file, load_path
from .gdoc import extract_gdoc

__all__ = ["extract_gdoc", "load_file", "load_path", "load_any"]


def load_any(uri: str) -> list[SourceDocument]:
    """Dispatch a URI/path to the right connector."""
    if uri.startswith("gdoc://"):
        return [extract_gdoc(uri.removeprefix("gdoc://"))]
    return load_path(Path(uri))
