"""Evaluation metrics (Step 24).

Component metrics (retrieval) + generation metrics localize failures: a low answer score with high
context recall means the generator is at fault, not retrieval. Default implementations are
deterministic lexical proxies (no external service); Ragas plugs in for richer scores in CI, and the
LLM judge (judge.py) covers what lexical can't.
"""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP = set("the a an of to for is are was were be by on in at and or with from as it this that "
            "you your our we they their can may will have has do how long".split())


def _tokens(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP and len(t) > 2}


def retrieval_hit(retrieved_payloads: list[dict], relevant_source_contains: str) -> float:
    if not relevant_source_contains:
        return 1.0
    needle = relevant_source_contains.lower()
    for p in retrieved_payloads:
        hay = (str(p.get("doc_id", "")) + " " + str(p.get("title", ""))).lower()
        if needle in hay:
            return 1.0
    return 0.0


def token_f1(pred: str, gold: str) -> float:
    p, g = _tokens(pred), _tokens(gold)
    if not p or not g:
        return 1.0 if not p and not g else 0.0
    tp = len(p & g)
    if tp == 0:
        return 0.0
    precision, recall = tp / len(p), tp / len(g)
    return 2 * precision * recall / (precision + recall)


def answer_correctness(answer: str, ideal: str) -> float:
    return token_f1(answer, ideal)


def answer_relevancy(answer: str, query: str) -> float:
    q = _tokens(query)
    if not q:
        return 1.0
    return len(q & _tokens(answer)) / len(q)
