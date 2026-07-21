"""Deterministic, model-free CI gate (Step 27).

CI has no GPU/LLM, so this gate validates the parts that ARE deterministic with the local backends:
retrieval quality (context recall over the golden set) and the input guardrail (adversarial cases
are refused). The full generation gate (faithfulness/relevancy via the judge) runs in staging with
real models (Step 24). Exits non-zero on regression so a bad change never merges.
"""

from __future__ import annotations

import sys
from pathlib import Path

from ragnarok.eval.golden import load_golden
from ragnarok.eval.metrics import retrieval_hit
from ragnarok.guardrails.input import check_input
from ragnarok.ingestion.chunking import chunk_document
from ragnarok.ingestion.connectors import load_any
from ragnarok.ingestion.embedding import embed_chunks
from ragnarok.ingestion.enrich import EnrichedDocument, Enrichment
from ragnarok.ingestion.normalize import normalize
from ragnarok.retrieval.hybrid import hybrid_search
from ragnarok.retrieval.preprocess import QueryPlan
from ragnarok.stores.vector import InMemoryVectorStore, MetadataFilter
from ragnarok.user import User

ROOT = Path(__file__).resolve().parents[3]
SAMPLE = str(ROOT / "datasets" / "sample")
GOLDEN = str(ROOT / "datasets" / "golden" / "v1")
RECALL_THRESHOLD = 0.75


def _build_index() -> InMemoryVectorStore:
    """Ingest the sample corpus with a model-free heuristic enrichment."""
    store = InMemoryVectorStore()
    for src in load_any(SAMPLE):
        norm = normalize(src)
        # heuristic enrichment (no LLM): empty Enrichment; access_tags default to ACLs
        enr = EnrichedDocument.from_norm(norm, Enrichment(), model="heuristic", version="heuristic@0")
        store.upsert(embed_chunks(chunk_document(enr)))
    return store


def main() -> int:
    store = _build_index()
    cases = load_golden(GOLDEN)
    recalls: list[float] = []
    failures: list[str] = []

    for case in cases:
        if case.expect == "refusal":
            v = check_input(case.query, User(id="ci"))
            handled = (not v.allowed) or ("injection" in v.flags)
            if not handled:
                failures.append(f"{case.id}: adversarial query not caught by input guardrail")
            continue
        plan = QueryPlan(rewritten_query=case.query)
        results = hybrid_search(plan, MetadataFilter(access_tags=["public"]), store)
        payloads = [r.payload for r in results]
        hit = retrieval_hit(payloads, case.relevant_source_contains)
        recalls.append(hit)
        if hit < 1.0:
            failures.append(f"{case.id}: relevant source '{case.relevant_source_contains}' not retrieved")

    recall = sum(recalls) / len(recalls) if recalls else 0.0
    print(f"CI gate: context_recall={recall:.3f} (threshold {RECALL_THRESHOLD})")
    for f in failures:
        print("  FAIL:", f)

    passed = recall >= RECALL_THRESHOLD and not failures
    print("RESULT:", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
