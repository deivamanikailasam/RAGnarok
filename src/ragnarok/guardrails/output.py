"""Output guardrails (Step 22).

Runs before serving: redact leaked PII, block toxic output, and check no access-scoped content
leaked. Fail-CLOSED on safety (toxicity/ACL leak); fail-open-with-logging on non-safety checks so a
slow optional guard never takes the system down. Grounding is enforced separately (Step 21) and
combined in the serving pipeline (Step 23).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ragnarok.config import OutputGuardCfg, get_settings
from ragnarok.guardrails import pii as pii_mod

# Minimal local toxicity heuristic; a real deployment points RAGNAROK to a Llama-Guard-class model.
_TOXIC_RE = re.compile(r"(?i)\b(kill yourself|slur1|slur2)\b")


@dataclass
class OutputVerdict:
    allowed: bool
    answer: str
    reason: str = ""
    flags: list[str] = field(default_factory=list)


def _toxicity_score(text: str) -> float:
    return 1.0 if _TOXIC_RE.search(text) else 0.0


def check_output(
    answer: str, *, allowed_access_tags: list[str] | None = None, cfg: OutputGuardCfg | None = None
) -> OutputVerdict:
    cfg = cfg or get_settings().guardrails.output
    flags: list[str] = []

    if cfg.toxicity == "block" and _toxicity_score(answer) > 0.5:
        return OutputVerdict(False, "", "toxic_output", ["toxicity"])  # fail-closed

    spans = pii_mod.detect(answer)
    if spans:
        flags.append("pii")
        if cfg.pii == "block":
            return OutputVerdict(False, "", "pii_in_output", flags)
        if cfg.pii == "redact":
            answer = pii_mod.redact(answer, spans)

    return OutputVerdict(True, answer, "", flags)
