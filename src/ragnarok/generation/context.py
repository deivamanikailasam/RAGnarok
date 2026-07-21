"""Context assembly (Step 19).

Packs reranked chunks into a token-budgeted context with numbered [i] citation markers. Tables are
handed over as exact markdown (so the generator reads real cell values). The best evidence is placed
at the start AND end ("lost-in-the-middle" mitigation) where models attend best. A fixed budget makes
latency/cost predictable and prevents context dilution.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ragnarok.config import GenerationCfg, get_settings
from ragnarok.retrieval.hybrid import RetrievalResult
from ragnarok.tokenization import count_tokens


@dataclass
class Citation:
    index: int
    chunk_id: str
    doc_id: str
    title: str
    section: str
    uri: str


@dataclass
class AssembledContext:
    text: str
    citations: list[Citation] = field(default_factory=list)
    results: list[RetrievalResult] = field(default_factory=list)  # in display order, aligned to [i]
    used_tokens: int = 0

    @property
    def n_chunks(self) -> int:
        return len(self.results)


def _block_text(index: int, r: RetrievalResult) -> str:
    p = r.payload
    header = f"[{index}] (source: {p.get('title', '?')} > {p.get('section', '')}, {p.get('freshness_date', 'n/a')})"
    body = p.get("table_markdown") or p.get("text", "")
    return f"{header}\n{body}"


def _reorder_for_attention(results: list[RetrievalResult]) -> list[RetrievalResult]:
    """Place the strongest evidence at the ends: [1st, 3rd, 5th, ... , 6th, 4th, 2nd]."""
    front: list[RetrievalResult] = []
    back: list[RetrievalResult] = []
    for i, r in enumerate(results):
        (front if i % 2 == 0 else back).append(r)
    return front + list(reversed(back))


def build_context(
    results: list[RetrievalResult], *, cfg: GenerationCfg | None = None
) -> AssembledContext:
    cfg = cfg or get_settings().generation
    # keep top-N by final score, within the token budget
    ranked = sorted(results, key=lambda r: r.final_score or r.rerank_score, reverse=True)
    kept: list[RetrievalResult] = []
    used = 0
    for r in ranked[: cfg.max_context_chunks]:
        block_tokens = count_tokens(_block_text(0, r))
        if kept and used + block_tokens > cfg.context_budget_tokens:
            break
        kept.append(r)
        used += block_tokens

    ordered = _reorder_for_attention(kept)
    citations: list[Citation] = []
    blocks: list[str] = []
    for i, r in enumerate(ordered, start=1):
        p = r.payload
        blocks.append(_block_text(i, r))
        citations.append(
            Citation(
                index=i,
                chunk_id=r.chunk_id,
                doc_id=r.doc_id,
                title=p.get("title", ""),
                section=p.get("section", ""),
                uri=p.get("uri", ""),
            )
        )
    return AssembledContext(
        text="\n\n".join(blocks), citations=citations, results=ordered, used_tokens=used
    )
