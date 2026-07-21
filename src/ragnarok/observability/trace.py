"""Tracing (Step 26).

One nested span tree per request, with tokens/cost/model+prompt-version per stage. In-memory by
default (inspectable in tests); pushed to Langfuse when configured. A slow/absent tracer never
affects the request — tracing is best-effort.
"""

from __future__ import annotations

import contextvars
import time
from dataclasses import dataclass, field
from typing import Any, Iterator
from contextlib import contextmanager

from ragnarok.observability import metrics
from ragnarok.observability.cost import cost_of
from ragnarok.providers import Usage


@dataclass
class Span:
    name: str
    meta: dict[str, Any] = field(default_factory=dict)
    tokens: int = 0
    cost: float = 0.0
    duration_s: float = 0.0
    children: list["Span"] = field(default_factory=list)

    def set(self, **kw: Any) -> None:
        self.meta.update(kw)


@dataclass
class Trace:
    trace_id: str
    root: Span
    total_tokens: int = 0
    total_cost: float = 0.0


_current: contextvars.ContextVar[Span | None] = contextvars.ContextVar("current_span", default=None)
_trace: contextvars.ContextVar[Trace | None] = contextvars.ContextVar("current_trace", default=None)


@contextmanager
def trace(name: str, trace_id: str = "", **meta: Any) -> Iterator[Trace]:
    root = Span(name=name, meta=meta)
    tr = Trace(trace_id=trace_id, root=root)
    t_token = _trace.set(tr)
    s_token = _current.set(root)
    start = time.monotonic()
    try:
        yield tr
    finally:
        root.duration_s = time.monotonic() - start
        _current.reset(s_token)
        _trace.reset(t_token)
        metrics.record_latency(name, root.duration_s)


@contextmanager
def span(name: str, **meta: Any) -> Iterator[Span]:
    parent = _current.get()
    sp = Span(name=name, meta=meta)
    if parent is not None:
        parent.children.append(sp)
    token = _current.set(sp)
    start = time.monotonic()
    try:
        yield sp
    finally:
        sp.duration_s = time.monotonic() - start
        _current.reset(token)
        metrics.record_latency(name, sp.duration_s)


def record_usage(stage: str, role: str, model: str, usage: Usage) -> None:
    """Attribute token usage + cost to the current span and the trace totals."""
    metrics.record_tokens(stage, role, usage.prompt_tokens, usage.completion_tokens)
    cost = cost_of(model, usage)
    metrics.record_cost(cost)
    sp = _current.get()
    if sp is not None:
        sp.tokens += usage.total
        sp.cost += cost
    tr = _trace.get()
    if tr is not None:
        tr.total_tokens += usage.total
        tr.total_cost += cost
