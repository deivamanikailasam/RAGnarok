"""PII detection & redaction (Step 22).

Regex detectors for common PII (email, phone, SSN, credit card) as a dependency-free default;
Presidio is used when installed for higher recall + more entity types. Used on both the input query
and the output answer so PII never reaches logs or users.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

_PATTERNS = {
    "EMAIL": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "PHONE": re.compile(r"\b(?:\+?\d{1,2}[\s-]?)?(?:\(?\d{3}\)?[\s-]?)\d{3}[\s-]?\d{4}\b"),
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "CREDIT_CARD": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
}


@dataclass
class PIISpan:
    entity: str
    text: str


def detect(text: str) -> list[PIISpan]:
    if os.environ.get("RAGNAROK_PII_BACKEND", "regex") == "presidio":  # pragma: no cover
        return _detect_presidio(text)
    spans: list[PIISpan] = []
    for entity, pattern in _PATTERNS.items():
        for m in pattern.finditer(text):
            spans.append(PIISpan(entity=entity, text=m.group()))
    return spans


def redact(text: str, spans: list[PIISpan] | None = None) -> str:
    spans = spans if spans is not None else detect(text)
    for span in spans:
        text = text.replace(span.text, f"[REDACTED_{span.entity}]")
    return text


def _detect_presidio(text: str) -> list[PIISpan]:  # pragma: no cover - requires presidio
    from presidio_analyzer import AnalyzerEngine

    analyzer = AnalyzerEngine()
    return [
        PIISpan(entity=r.entity_type, text=text[r.start : r.end])
        for r in analyzer.analyze(text=text, language="en")
    ]
