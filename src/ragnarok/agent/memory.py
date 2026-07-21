"""Agent memory (Step 39).

Short-term memory = the ReAct scratchpad (this turn's thoughts/actions/observations). Long-term
memory = facts persisted across turns/sessions, recalled by keyword overlap. Both feed the planner's
prompt so the agent can build on prior context (the diagram's "Short Term / Long Term Memory").
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass
class ShortTermMemory:
    steps: list[str] = field(default_factory=list)

    def add(self, thought: str, tool: str, tool_input: str, observation: str) -> None:
        self.steps.append(
            f"Thought: {thought}\nAction: {tool}({tool_input})\nObservation: {observation}"
        )

    def render(self) -> str:
        return "\n".join(self.steps)


class LongTermMemory:
    def __init__(self) -> None:
        self._facts: dict[str, list[str]] = {}

    def remember(self, user_id: str, fact: str) -> None:
        self._facts.setdefault(user_id, []).append(fact)

    def recall(self, user_id: str, query: str, limit: int = 3) -> list[str]:
        q = set(_TOKEN_RE.findall(query.lower()))
        scored = [
            (len(q & set(_TOKEN_RE.findall(f.lower()))), f) for f in self._facts.get(user_id, [])
        ]
        scored.sort(reverse=True)
        return [f for score, f in scored if score > 0][:limit]


_long_term = LongTermMemory()


def long_term_memory() -> LongTermMemory:
    return _long_term
