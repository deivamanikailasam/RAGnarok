# 00 — Overview & Architecture

This document is the map for everything that follows. It defines the system, its data flows, the
design principles that make it "production‑grade", and the reasoning behind each major technology
choice. Every later document (`01`–`14`) implements one box or cross‑cutting concern from the
architecture diagram.

---

## 1. What we are building

An **Agentic RAG** system: instead of the naive "embed query → top‑k → stuff into prompt" pipeline,
RAGnarok inserts small, cheap LLM‑driven **agents** at the points where they add the most quality
per token:

- **At ingestion**, a large LLM *enriches* content (summaries, table descriptions, custom metadata)
  so that what we index is richer than the raw text.
- **Before retrieval**, small LLMs *optimize the query* and *identify the right sources*
  (metadata filters), so retrieval starts from a better input.
- **During retrieval**, we combine dense vector search, sparse BM25 lexical search, and a
  cross‑encoder reranker — hybrid retrieval that no single method matches.
- **After retrieval**, a large LLM *generates* a grounded answer and a small LLM *post‑processes* it
  (citation formatting, guardrail checks, style).
- **Around everything**, guardrails, tracing, and continuous evaluation form the feedback loop that
  keeps quality from silently degrading.

"Agentic" here is deliberate and bounded. We do **not** hand a single autonomous agent an open
toolbelt and hope. Each agent has one job, a constrained output schema, a small model where a small
model suffices, and a large model only where reasoning depth pays off. This is what makes the
system both high‑quality **and** affordable — see [docs/14](14-optimization-playbook.md).

---

## 2. The two planes: Offline (build) and Online (serve)

The architecture cleanly separates work done **ahead of time** from work done **per query**.
This is the single most important production principle: push everything expensive offline.

### 2.1 Offline plane — "Enriched Document Processing"

Runs on a schedule or on document change. Produces the artifacts the online plane reads.

```
Knowledge Documents (GDocs / files)
        │
        ▼
[03] Content Extraction (GDoc API, file loaders)
        │
        ▼
[03] Content Enrichment  ── large LLM ──►  summaries, table descriptions, custom metadata
        │
        ▼
[04] Table‑aware Chunking
        │
        ▼
[04] Embedding Generation ── embedding model ──► dense + sparse vectors
        │
        ├──►  [05] Vector Store (Qdrant): chunk vectors + payload
        └──►  [05] Feature Store (Feast): Custom Metadata Attributes
```

### 2.2 Online plane — per query

Runs on every user question, must be fast and safe.

```
User ─ [10] Slack Interface ─► [09] Input Guardrails
                                       │
                                       ▼
                    [06] Pre‑Process:  query optimizer (small LLM)
                                       source identifier (small LLM) → metadata filters
                                       │
                                       ▼
                    [07] Retrieval:    vector search + BM25 search → fuse → rerank → top‑k chunks
                                       │
                                       ▼
                    [08] Answer Generation: large LLM generator → small LLM post‑processor
                                       │
                                       ▼
                    [09] Output Guardrails (grounding, PII, safety) ─► [10] Answer to user
```

Cross‑cutting on both planes: **[09] Guardrails**, **[12] Observability**, and **[11] Evaluation**
(pre‑prod against golden data, post‑prod against live traffic).

---

## 3. End‑to‑end request trace (the "happy path")

A concrete walk‑through of one Slack question, with the latency budget we design against
(local, single‑GPU box; see [docs/14](14-optimization-playbook.md) for how each number is achieved):

| Stage | Component | Typical latency | Notes |
|---|---|---|---|
| 1 | Slack event → API | 5–20 ms | Socket Mode, no public endpoint |
| 2 | Input guardrails | 10–40 ms | regex/Presidio PII, injection heuristics; LLM check only if flagged |
| 3 | Query optimizer (small LLM) | 120–300 ms | rewrites, expands, decomposes; cached |
| 4 | Source identifier (small LLM) | 80–200 ms | emits metadata filter JSON; often runs in parallel with 3 |
| 5 | Embed query | 10–30 ms | BGE‑M3, cached |
| 6 | Vector + BM25 search | 15–50 ms | Qdrant, parallel dense+sparse |
| 7 | Rerank top‑50 → top‑8 | 40–120 ms | BGE cross‑encoder, batched |
| 8 | Answer generation (large LLM) | 1.5–6 s | streamed to user token‑by‑token |
| 9 | Post‑processor (small LLM) | 150–400 ms | citations, guardrail, formatting; can overlap with grounding check |
| 10 | Output guardrails | 20–200 ms | grounding/faithfulness gate |

**Time‑to‑first‑token** is what users feel: stages 1–7 (~0.3–0.8 s) then streaming begins. Design
target: **first token < 1 s**, full answer < 8 s locally.

---

## 4. Design principles (the "production‑grade" checklist)

1. **Local‑first, provider‑agnostic.** Every model is addressed by a *role* (`llm_large`,
   `llm_small`, `embedding`, `reranker`), not a vendor. Roles resolve to an OpenAI‑compatible
   endpoint. Swap Ollama ↔ vLLM ↔ OpenAI by config. ([docs/02](02-configuration-and-model-providers.md))
2. **Everything expensive is offline.** Enrichment, embedding, and metadata extraction happen at
   ingest, never per query.
3. **Right‑sized models.** Small LLMs for narrow, schema‑bound tasks (query rewrite, source id,
   post‑process); large LLMs only for enrichment and final generation.
4. **Structured I/O everywhere.** Every agent emits validated JSON (Pydantic / JSON‑schema
   constrained decoding). No brittle string parsing. This is a correctness *and* a token win.
5. **Hybrid over monolithic retrieval.** Dense + sparse + rerank beats any single retriever,
   especially for names, codes, and rare terms that embeddings blur.
6. **Guardrails are non‑optional and fail‑closed** on safety, fail‑open (with logging) on latency.
7. **Observability is built‑in, not bolted on.** Every LLM call, retrieval, and guardrail decision
   is traced with token counts and cost. You cannot optimize what you cannot see.
8. **Continuous evaluation.** A golden dataset gates releases (pre‑prod); sampled live traffic is
   scored continuously (post‑prod). Regressions surface before users complain.
9. **Idempotent, versioned, reproducible.** Content, chunks, embeddings, and prompts are all
   content‑hashed and versioned so any answer can be reproduced and any change can be A/B tested.
10. **Graceful degradation.** If the reranker is down, fall back to fusion scores; if the large LLM
    is saturated, queue or fall back to a smaller model with a labelled "degraded" response.

---

## 5. Key technology choices and *why*

| Choice | Why this, why not the obvious alternative |
|---|---|
| **OpenAI‑compatible protocol as the lingua franca** | Ollama, vLLM, LM Studio, OpenAI, Together, Azure all speak it. One client, zero vendor lock‑in. |
| **Qdrant** as vector store | First‑class **payload filtering** (needed for source‑identifier metadata filters), native **sparse vectors** (BM25/BM42 in the same store → one round trip for hybrid), Rust performance, runs in a single container. Simpler ops than Weaviate; richer filtering than raw pgvector. |
| **BGE‑M3** embeddings | Single model produces **dense + sparse (lexical) + multi‑vector** representations, multilingual, 8k context. Lets us do hybrid retrieval from *one* embedding pass. |
| **BGE‑reranker‑v2‑m3** | Cross‑encoder reranking is the highest quality‑per‑dollar lever in RAG; local, small, batchable. |
| **Feast** feature store | The diagram's "Feature Store / Custom Metadata Attributes" — Feast gives a clean offline/online split for document‑level features (freshness, authority, access tags) reused by the source identifier and reranker. |
| **LangGraph** for the agent loop | Explicit, inspectable state machine (not an opaque agent). Deterministic control flow = debuggable + traceable. |
| **Langfuse** (self‑hosted) | Purpose‑built LLM tracing: nested spans, token/cost accounting, prompt management, dataset‑based eval — all in one local container. |
| **Ragas + promptfoo** | Ragas gives RAG‑specific metrics (faithfulness, context precision/recall, answer relevancy); promptfoo gives assertion‑based regression gating in CI. |
| **Slack Bolt Socket Mode** | No inbound public endpoint required → works behind a firewall / fully local, which matches the "private" premise. |
| **vLLM in prod** | Continuous batching + paged attention → 5–20× throughput over Ollama for concurrent users; still local, still OpenAI‑compatible. |

---

## 6. Data & artifact model (what flows between stages)

Canonical objects, all versioned and hashed. Full schemas in [docs/05](05-storage-vector-and-feature-store.md).

- **`SourceDocument`** — `{doc_id, uri, source_type, raw_text, acl_tags, fetched_at, content_hash}`
- **`EnrichedDocument`** — adds `{summary, section_summaries[], table_descriptions[], custom_metadata{...}, enrichment_model, enrichment_version}`
- **`Chunk`** — `{chunk_id, doc_id, text, contextual_prefix, chunk_type(text|table|code), position, token_count, metadata{...}, content_hash}`
- **`EmbeddedChunk`** — adds `{dense_vector, sparse_vector, embedding_model, embedding_version}`
- **`RetrievalResult`** — `{chunk_id, dense_score, sparse_score, fused_score, rerank_score, source}`
- **`Answer`** — `{text, citations[], grounding_score, guardrail_verdicts[], trace_id, tokens{...}, cost}`

**Versioning rule:** any change to a prompt, model, chunker, or embedding model bumps the relevant
`*_version` and changes downstream `content_hash`es, so stale artifacts are detected and re‑built
incrementally rather than blindly served.

---

## 7. How to read the rest of the docs

Each subsequent document follows the same shape:

- **What** — the component's responsibility and its box in the diagram.
- **How** — concrete implementation with runnable reference snippets and config.
- **Why** — the reasoning and the alternatives rejected.
- **Optimizations** — the specific wins for tokenization, latency, cost, and quality, with numbers.
- **Failure modes & guardrails** — what breaks and how we contain it.
- **What "done" looks like** — acceptance criteria / tests.

Start with [docs/01 — Environment & Infrastructure](01-environment-and-infrastructure.md).
