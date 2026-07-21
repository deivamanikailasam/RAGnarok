"""Evaluation runner + release gate (Step 24).

Runs the pipeline over a golden suite, aggregates metrics, and compares to gate thresholds. CI calls
this; a change that regresses any gate does not ship. Post-prod (Step 25) reuses the same metrics on
sampled live traffic.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional

from ragnarok.config import get_settings
from ragnarok.eval.golden import GoldenCase, load_golden
from ragnarok.eval.metrics import answer_correctness, answer_relevancy, retrieval_hit
from ragnarok.pipeline import AnswerResult, answer
from ragnarok.stores.features import FeatureStore
from ragnarok.stores.vector import VectorStore
from ragnarok.user import User

AnswerFn = Callable[[str], Awaitable[AnswerResult]]


@dataclass
class CaseResult:
    id: str
    metrics: dict[str, float]
    passed_expectation: bool


@dataclass
class EvalReport:
    metrics: dict[str, float] = field(default_factory=dict)
    per_case: list[CaseResult] = field(default_factory=list)
    passed: bool = False

    def __str__(self) -> str:
        lines = [f"eval: {'PASS' if self.passed else 'FAIL'}"]
        for k, v in self.metrics.items():
            lines.append(f"  {k}: {v:.3f}")
        return "\n".join(lines)


async def _score_case(case: GoldenCase, result: AnswerResult) -> CaseResult:
    if case.expect == "refusal":
        ok = result.blocked or result.abstained
        return CaseResult(case.id, {"refusal_correct": 1.0 if ok else 0.0}, ok)

    payloads = [c.__dict__ for c in result.citations]
    metrics = {
        "context_recall": retrieval_hit(payloads, case.relevant_source_contains),
        "faithfulness": result.grounding_score if result.grounded else 0.0,
        "answer_relevancy": answer_relevancy(result.text, case.query),
        "answer_correctness": answer_correctness(result.text, case.ideal_answer),
    }
    return CaseResult(case.id, metrics, not result.blocked)


def _aggregate(cases: list[CaseResult]) -> dict[str, float]:
    keys: set[str] = set()
    for c in cases:
        keys |= set(c.metrics)
    out = {}
    for k in keys:
        vals = [c.metrics[k] for c in cases if k in c.metrics]
        out[k] = sum(vals) / len(vals) if vals else 0.0
    return out


async def run_suite(
    suite: Optional[str] = None,
    *,
    answer_fn: Optional[AnswerFn] = None,
    store: Optional[VectorStore] = None,
    features: Optional[FeatureStore] = None,
    facets: Optional[dict[str, list[str]]] = None,
    thresholds: Optional[dict[str, float]] = None,
) -> EvalReport:
    settings = get_settings()
    cases = load_golden(suite or settings.eval.golden_suite)

    async def _default_answer(q: str) -> AnswerResult:
        return await answer(q, User(entitlements=["public"]), store=store, features=features, facets=facets)

    fn = answer_fn or _default_answer
    results = await asyncio.gather(*[fn(c.query) for c in cases])
    per_case = [await _score_case(c, r) for c, r in zip(cases, results)]
    metrics = _aggregate(per_case)

    gate = thresholds or settings.eval.gate_thresholds
    passed = all(metrics.get(k, 0.0) >= thr for k, thr in gate.items())
    # refusal cases must also pass their expectation
    passed = passed and all(c.passed_expectation for c in per_case)
    report = EvalReport(metrics=metrics, per_case=per_case, passed=passed)
    print(report)
    return report
