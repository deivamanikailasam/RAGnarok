"""RAG strategy registry (Step 29; strategies added in Steps 30-39).

Strategies register themselves by name; ``get_strategy(name)`` returns an instance. The pipeline
resolves the name from ``rag.strategy`` (or the Adaptive router). See docs/15 for the catalog.
"""

from __future__ import annotations

from ragnarok.strategies.base import RagStrategy, StrategyContext, StrategyResult

_REGISTRY: dict[str, type] = {}


def register(cls: type) -> type:
    _REGISTRY[cls.name] = cls
    return cls


def get_strategy(name: str) -> RagStrategy:
    if name not in _REGISTRY:
        raise KeyError(f"unknown RAG strategy '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]()


def available() -> list[str]:
    return sorted(_REGISTRY)


# Import strategy modules so they self-register. Each is a separate step.
from ragnarok.strategies import hybrid as _hybrid  # noqa: E402,F401
from ragnarok.strategies import hyde as _hyde  # noqa: E402,F401
from ragnarok.strategies import naive as _naive  # noqa: E402,F401

__all__ = ["RagStrategy", "StrategyContext", "StrategyResult", "get_strategy", "available", "register"]
