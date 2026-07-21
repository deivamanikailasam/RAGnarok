"""Metrics — the RAG SLIs (Step 26).

Always-available in-memory collector (so metrics work in tests/offline), optionally mirrored to
Prometheus when prometheus_client is installed. Tracks the signals that matter for a RAG service:
TTFT, per-stage latency, tokens by stage/role/direction, cost, grounding, abstain rate.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MetricsCollector:
    counters: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    observations: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))

    def inc(self, name: str, value: float = 1.0, **labels: str) -> None:
        self.counters[_key(name, labels)] += value
        _mirror_counter(name, value, labels)

    def observe(self, name: str, value: float, **labels: str) -> None:
        self.observations[_key(name, labels)].append(value)
        _mirror_histogram(name, value, labels)

    def snapshot(self) -> dict[str, Any]:
        summary: dict[str, Any] = dict(self.counters)
        for k, vals in self.observations.items():
            summary[k] = {"count": len(vals), "avg": sum(vals) / len(vals) if vals else 0.0}
        return summary

    def reset(self) -> None:
        self.counters.clear()
        self.observations.clear()


def _key(name: str, labels: dict[str, str]) -> str:
    if not labels:
        return name
    return name + "{" + ",".join(f"{k}={v}" for k, v in sorted(labels.items())) + "}"


_collector = MetricsCollector()


def collector() -> MetricsCollector:
    return _collector


# --- domain helpers used across the pipeline ---


def record_tokens(stage: str, role: str, prompt: int, completion: int) -> None:
    _collector.inc("tokens_total", prompt, stage=stage, role=role, direction="in")
    _collector.inc("tokens_total", completion, stage=stage, role=role, direction="out")


def record_ttft(seconds: float) -> None:
    _collector.observe("ttft_seconds", seconds)


def record_latency(stage: str, seconds: float) -> None:
    _collector.observe("stage_latency_seconds", seconds, stage=stage)


def record_grounding(score: float) -> None:
    _collector.observe("grounding_score", score)


def record_answer(abstained: bool, blocked: bool) -> None:
    _collector.inc("answers_total")
    if abstained:
        _collector.inc("abstain_total")
    if blocked:
        _collector.inc("blocked_total")


def record_cost(usd: float) -> None:
    _collector.inc("query_cost_usd_total", usd)


# --- Prometheus mirroring (optional) ---

_prom_cache: dict[str, Any] = {}


def _prom():  # pragma: no cover - requires prometheus_client
    if "mod" not in _prom_cache:
        try:
            import prometheus_client

            _prom_cache["mod"] = prometheus_client
        except Exception:
            _prom_cache["mod"] = None
    return _prom_cache["mod"]


def _mirror_counter(name: str, value: float, labels: dict[str, str]) -> None:  # pragma: no cover
    prom = _prom()
    if prom is None:
        return
    key = "c_" + name
    if key not in _prom_cache:
        _prom_cache[key] = prom.Counter(name, name, list(labels))
    (_prom_cache[key].labels(**labels) if labels else _prom_cache[key]).inc(value)


def _mirror_histogram(name: str, value: float, labels: dict[str, str]) -> None:  # pragma: no cover
    prom = _prom()
    if prom is None:
        return
    key = "h_" + name
    if key not in _prom_cache:
        _prom_cache[key] = prom.Histogram(name, name, list(labels))
    (_prom_cache[key].labels(**labels) if labels else _prom_cache[key]).observe(value)


def metrics_response() -> Any:  # pragma: no cover - requires prometheus_client + fastapi
    from fastapi import Response
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
