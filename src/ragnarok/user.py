"""Caller identity and entitlements (Steps 14/15/22).

Entitlements come from the IdP/SSO group mapping and are the hard security boundary for retrieval:
the source identifier may narrow within them but never widen (Step 15).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class User:
    id: str = "anonymous"
    entitlements: list[str] = field(default_factory=lambda: ["public"])
    tier: str | None = None  # optional: the customer tier this agent is serving

    def scoped_tags(self) -> list[str]:
        return list(self.entitlements)
