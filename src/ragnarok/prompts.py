"""Versioned prompt configs (Step 3 — the "Prompt configs" box).

Prompts are config, not code: versioned YAML under ``config/prompts/``, hot-reloadable, individually
testable, and recorded in every trace so a prompt change is a reversible, gate-able release
(no redeploy). ``render()`` turns a prompt + input vars into chat messages.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Prompt:
    id: str
    version: int
    model_role: str
    system: str
    few_shot: list[dict[str, Any]]
    user_template: str | None = None
    output_schema_ref: str | None = None

    def render(self, **variables: Any) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [{"role": "system", "content": self.system}]
        for ex in self.few_shot:
            messages.append({"role": "user", "content": _as_text(ex.get("input"))})
            messages.append({"role": "assistant", "content": _as_text(ex.get("output"))})
        if self.user_template:
            user = self.user_template.format(**variables)
        else:
            user = _as_text(variables)
        messages.append({"role": "user", "content": user})
        return messages


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


class PromptRegistry:
    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root else Path(__file__).resolve().parents[2] / "config" / "prompts"
        self._by_id: dict[str, dict[int, Prompt]] = {}
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        for path in sorted(self.root.glob("*.yaml")):
            raw = yaml.safe_load(path.read_text()) or {}
            p = Prompt(
                id=raw["id"],
                version=int(raw.get("version", 1)),
                model_role=raw.get("model_role", "llm_small"),
                system=raw.get("system", ""),
                few_shot=raw.get("few_shot", []) or [],
                user_template=raw.get("user_template"),
                output_schema_ref=raw.get("output_schema_ref"),
            )
            self._by_id.setdefault(p.id, {})[p.version] = p
        self._loaded = True

    def get(self, prompt_id: str, version: int | str = "latest") -> Prompt:
        self._load()
        versions = self._by_id.get(prompt_id)
        if not versions:
            raise KeyError(f"no prompt '{prompt_id}' found under {self.root}")
        if version == "latest":
            version = max(versions)
        return versions[int(version)]

    def version(self, prompt_id: str) -> int:
        return self.get(prompt_id).version

    def render(self, prompt_id: str, version: int | str = "latest", **variables: Any) -> list[dict]:
        return self.get(prompt_id, version).render(**variables)

    def label(self, prompt_id: str, version: int | str = "latest") -> str:
        p = self.get(prompt_id, version)
        return f"{p.id}@v{p.version}"


_registry: PromptRegistry | None = None


def prompts() -> PromptRegistry:
    global _registry
    if _registry is None:
        _registry = PromptRegistry()
    return _registry
