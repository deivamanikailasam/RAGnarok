"""Tests for post-prod continuous evaluation (Step 25)."""

from __future__ import annotations

from ragnarok.eval.postprod import (
    ServedAnswer,
    TrafficLog,
    detect_drift,
    promote_failures_to_golden,
    score_live,
)


def test_sampling_always_includes_negatives():
    log = TrafficLog()
    for i in range(50):
        log.record(ServedAnswer(trace_id=f"ok-{i}", query="q", answer="a", grounded=True))
    log.record(ServedAnswer(trace_id="bad", query="wrong?", answer="a", grounded=False))
    log.record(ServedAnswer(trace_id="td", query="down?", answer="a", feedback=-1))
    sample = log.sample(rate=0.1)
    ids = {s.trace_id for s in sample}
    assert "bad" in ids and "td" in ids  # forced in regardless of sampling


def test_score_live_and_drift_alert():
    sample = [
        ServedAnswer("1", "q", "a", grounded=True, grounding_score=0.9, feedback=1),
        ServedAnswer("2", "q", "a", grounded=False, grounding_score=0.0, feedback=-1, abstained=True),
    ]
    metrics = score_live(sample)
    assert 0.0 <= metrics["faithfulness"] <= 1.0
    assert metrics["thumbs_up_rate"] == 0.5

    baseline = {"faithfulness": 0.9, "thumbs_up_rate": 0.9, "abstain_rate": 0.05}
    alerts = detect_drift(metrics, baseline, max_drop=0.1)
    assert any("faithfulness" in a for a in alerts)


def test_failures_become_golden_cases():
    sample = [
        ServedAnswer("aaaaaaaa11", "why was I charged?", "a", grounded=False),
        ServedAnswer("bbbbbbbb22", "good one", "a", grounded=True, feedback=1),
    ]
    new_cases = promote_failures_to_golden(sample)
    assert len(new_cases) == 1
    assert new_cases[0].query == "why was I charged?"
    assert "needs_review" in new_cases[0].tags
