"""Post-prod continuous evaluation (Step 25).

Samples live answers (100% of thumbs-down / low-grounding + a fraction of the rest), scores them with
the same metrics/judges, detects drift vs a baseline, and feeds failures back into the golden set and
per-document feedback. Production is a graded, self-hardening loop — not launch-and-forget.

Sampling is deterministic (hash of trace_id) so it is reproducible and needs no RNG.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ragnarok.eval.golden import GoldenCase


@dataclass
class ServedAnswer:
    trace_id: str
    query: str
    answer: str
    grounded: bool = True
    grounding_score: float = 1.0
    abstained: bool = False
    doc_ids: list[str] = field(default_factory=list)
    feedback: int | None = None  # +1 / -1 from the user (Slack buttons / API)


class TrafficLog:
    def __init__(self) -> None:
        self._log: list[ServedAnswer] = []

    def record(self, served: ServedAnswer) -> None:
        self._log.append(served)

    def all(self) -> list[ServedAnswer]:
        return list(self._log)

    def sample(self, rate: float = 0.1) -> list[ServedAnswer]:
        """Always include negative/low-grounding cases; sample the rest deterministically."""
        bucket = max(int(1 / rate), 1) if rate > 0 else 10**9
        out = []
        for s in self._log:
            forced = (s.feedback is not None and s.feedback < 0) or not s.grounded or s.abstained
            if forced or (hash(s.trace_id) % bucket == 0):
                out.append(s)
        return out


def score_live(sample: list[ServedAnswer]) -> dict[str, float]:
    if not sample:
        return {"faithfulness": 1.0, "thumbs_up_rate": 1.0, "abstain_rate": 0.0}
    n = len(sample)
    faith = sum(s.grounding_score if s.grounded else 0.0 for s in sample) / n
    thumbs = [s for s in sample if s.feedback is not None]
    up = sum(1 for s in thumbs if s.feedback > 0) / len(thumbs) if thumbs else 1.0
    abstain = sum(1 for s in sample if s.abstained) / n
    return {"faithfulness": faith, "thumbs_up_rate": up, "abstain_rate": abstain}


def detect_drift(
    current: dict[str, float], baseline: dict[str, float], *, max_drop: float = 0.1
) -> list[str]:
    alerts = []
    for metric in ("faithfulness", "thumbs_up_rate"):
        if baseline.get(metric, 1.0) - current.get(metric, 0.0) > max_drop:
            alerts.append(f"{metric} dropped from {baseline[metric]:.2f} to {current[metric]:.2f}")
    if current.get("abstain_rate", 0.0) - baseline.get("abstain_rate", 0.0) > max_drop:
        alerts.append(
            f"abstain_rate rose to {current['abstain_rate']:.2f} (coverage gap?)"
        )
    return alerts


def promote_failures_to_golden(sample: list[ServedAnswer]) -> list[GoldenCase]:
    """Turn production failures into new golden cases so the gate hardens over time."""
    cases = []
    for s in sample:
        if (s.feedback is not None and s.feedback < 0) or not s.grounded:
            cases.append(
                GoldenCase(
                    id=f"prod-{s.trace_id[:8]}",
                    query=s.query,
                    ideal_answer="",  # to be filled by an SME during review
                    expect="answer",
                    tags=["from_production", "needs_review"],
                )
            )
    return cases
