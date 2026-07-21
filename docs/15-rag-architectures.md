# 15 — RAG Architectures (catalog, verification, and how RAGnarok implements each)

This document catalogs the well-known RAG architectures, **verifies each is a genuine, recognized
method** (with its origin), and maps it to a **pluggable strategy** in RAGnarok. Every strategy is
selectable by config (`rag.strategy`) or chosen automatically by the **Adaptive** router, and every
one is comparable head-to-head through the evaluation harness ([docs/11](11-evaluation-and-golden-datasets.md), Step 24).

Implemented in Steps 29–39 (see [docs/IMPLEMENTATION.md](IMPLEMENTATION.md)). Code lives in
`src/ragnarok/strategies/`.

---

## 1. Are these genuine? (verification)

The 8 architectures in the reference diagram are all real and recognized; most have published papers.
RAGnarok also adds three widely-used methods the diagram omits.

| # | Architecture | Genuine | Origin / reference | RAGnarok strategy | Step |
|---|---|---|---|---|---|
| 1 | **Naive RAG** | ✅ | Canonical baseline (Lewis et al. 2020, "RAG") | `naive` | 29 |
| 2 | **Hybrid RAG (dense+sparse)** | ✅ | Standard IR practice (BM25 + vectors) | `hybrid` (default) | 16/29 |
| 3 | **HyDE** | ✅ | Gao et al. 2022, *Hypothetical Document Embeddings* (arXiv:2212.10496) | `hyde` | 30 |
| 4 | **RAG-Fusion** *(added)* | ✅ | Multi-query + Reciprocal Rank Fusion (Raudaschl 2023) | `fusion` | 31 |
| 5 | **Corrective RAG (CRAG)** | ✅ | Yan et al. 2024 (arXiv:2401.15884) | `corrective` | 32 |
| 6 | **Self-RAG** *(added)* | ✅ | Asai et al. 2023 (arXiv:2310.11511) | `self_rag` | 33 |
| 7 | **Graph RAG** | ✅ | Microsoft GraphRAG 2024; earlier KG-RAG work | `graph` | 34 |
| 8 | **Hybrid RAG (vector+graph)** | ✅ | Sarmah et al. 2024, *HybridRAG* (arXiv:2408.04948) | `hybrid_graph` | 35 |
| 9 | **Multimodal RAG** | ✅ | CLIP/ColPali-style multimodal retrieval | `multimodal` | 36 |
| 10 | **RAPTOR** *(added)* | ✅ | Sarthi et al. 2024 (arXiv:2401.18059) | `raptor` | 37 |
| 11 | **Adaptive RAG** | ✅ | Jeong et al. 2024 (arXiv:2403.14403) | `adaptive` | 38 |
| 12 | **Agentic RAG** | ✅ | Emerging pattern (ReAct: Yao et al. 2022; tools/MCP) | `agentic` | 39 |

**Terminology note.** The reference diagram uses "Hybrid RAG" to mean **Vector + Graph**. In general
IR usage, "hybrid retrieval" means **dense + sparse (BM25)** — which RAGnarok already does as its
*default* retrieval (Step 16). To avoid ambiguity we keep both: `hybrid` = dense+sparse,
`hybrid_graph` = vector+graph.

---

## 2. The strategy framework (Step 29)

All architectures implement one interface so they are interchangeable, individually testable, and
eval-comparable:

```python
class RagStrategy(Protocol):
    name: str
    async def run(self, ctx: StrategyContext) -> StrategyResult: ...
```

- `StrategyContext` carries the sanitized query, the `PreprocessResult` (plan + source filters), the
  caller `User`, the vector/feature/graph stores, and the collection.
- `StrategyResult` returns the retrieved `results` for the generator, plus optional `direct_answer`
  (the no-retrieval path) and `notes` (corrective actions, chosen sub-strategy, reflections).

The online pipeline ([docs/10](10-slack-interface-and-serving.md), Step 23) selects a strategy (config default or Adaptive), runs it,
then applies the **shared** generation → post-process → grounding-gate → output-guardrail path. So
every strategy inherits citations, the grounding/abstain safety net (Step 21), guardrails (Step 22),
adaptive budgets (Step 28), and observability (Step 26) for free.

---

## 3. Each architecture — what it is and how RAGnarok implements it

*(Filled in as each step lands; see the linked step for the why-not-alternatives rationale.)*

- **Naive** — dense-only retrieve → generate. Baseline for eval comparison; never the production default.
- **Hybrid (dense+sparse)** — the default; RRF fusion + cross-encoder rerank (Steps 16–18).
- **HyDE** — a small LLM writes a *hypothetical answer*; we embed that (it sits closer to real
  passages than a short question) and retrieve. Falls back to the plain query.
- **RAG-Fusion** — generate N query paraphrases, retrieve each, fuse with RRF; robust recall on
  under-specified queries.
- **Corrective RAG (CRAG)** — grade retrieved chunks; if the top grade is weak, take a corrective
  action (query rewrite + re-retrieve, then a local-knowledge/web fallback) before generating.
- **Self-RAG** — reflect on *whether* to retrieve, filter irrelevant chunks by a relevance reflection,
  and mark support (reusing the grounding gate) — reducing hallucination and needless retrieval.
- **Graph RAG** — a lightweight knowledge graph (entities + co-occurrence relations) built at ingest;
  retrieve entity neighborhoods and surface the chunks that mention them.
- **Hybrid RAG (vector+graph)** — fuse vector results with graph-expanded results (best for
  multi-hop / relationship questions the vector store alone answers poorly).
- **Multimodal RAG** — images/diagrams are captioned at ingest (a caption model / VLM hook) and
  embedded alongside text; retrieval is modality-aware and the generator receives the caption + a
  reference to the asset.
- **RAPTOR** — recursively cluster + summarize chunks into a tree at ingest; retrieve across tree
  levels so both fine detail and high-level context are available for long documents.
- **Adaptive RAG** — a complexity classifier routes each query to *no-retrieval* (direct answer),
  *single-step* (hybrid/HyDE), or *multi-step* (fusion/graph/agentic) — spending compute in
  proportion to difficulty.
- **Agentic RAG** — a bounded ReAct-style loop with **memory** (short/long term), **planning**, and a
  **tool registry** (retrieval, calculator, and pluggable local/search/cloud tool servers in the
  MCP style). The agent decides which tools to call, gathers evidence, and hands it to the generator.

---

## 4. Choosing a strategy

- **Config:** set `rag.strategy` in `settings.yaml` (default `hybrid`).
- **Automatic:** set `rag.strategy: adaptive` to let the router pick per query.
- **Per request:** the API/pipeline accepts a `strategy` override for A/B testing.
- **Which to use?** `hybrid` is the right default for most corpora. Reach for `graph`/`hybrid_graph`
  on relationship-heavy corpora, `raptor` on long documents, `multimodal` when images matter,
  `corrective`/`self_rag` when precision/faithfulness is paramount, and `agentic` when answering
  needs tools or multi-hop planning. Let **eval** ([docs/11](11-evaluation-and-golden-datasets.md)) decide — every strategy is scored on
  the same golden set.

---

_See also: [docs/IMPLEMENTATION.md](IMPLEMENTATION.md) Steps 29–39 · [docs/07 Hybrid Retrieval](07-hybrid-retrieval.md)._
