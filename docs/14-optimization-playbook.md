# 14 — Optimization Playbook

**What:** the consolidated, cross‑cutting view of every optimization RAGnarok applies, grouped by
the lever it pulls — **tokenization, latency, cost, throughput, and quality** — with where it lives
and the rough magnitude of the win. Use this as the tuning checklist.

The meta‑principle: **measure first ([docs/12](12-observability-and-monitoring.md)), then pull the biggest lever.** In RAG, the biggest
levers are almost always (1) how many tokens the *large* model reads, and (2) retrieval precision.

---

## 1. Tokenization optimizations

| Optimization | Where | Effect |
|---|---|---|
| **Model‑correct token counting** | [docs/04](04-chunking-and-embedding.md),[docs/08](08-answer-generation-and-postprocessing.md) | chunk sizes & context budgets exact for the serving model — no over/under‑fill |
| **Guided‑JSON constrained decoding** | [docs/02](02-configuration-and-model-providers.md) | agents emit only schema fields → 30–70% fewer output tokens, zero reparse |
| **Reranked, budgeted context (≤8 chunks)** | [docs/07](07-hybrid-retrieval.md),[docs/08](08-answer-generation-and-postprocessing.md) | the dominant input‑token cost; broad‑retrieve/narrow‑rerank keeps it small |
| **Contextual prefix (few tokens/chunk, offline)** | [docs/04](04-chunking-and-embedding.md) | recall gain paid once at ingest, not per query |
| **Small‑LLM post‑processing** | [docs/08](08-answer-generation-and-postprocessing.md) | citations/format/claims at ~1/10th large‑model token cost |
| **Table markdown only when a table chunk is used** | [docs/04](04-chunking-and-embedding.md),[docs/08](08-answer-generation-and-postprocessing.md) | no verbose tables in context unless relevant |
| **`min_rerank_score` floor** | [docs/07](07-hybrid-retrieval.md) | drops weak chunks → smaller context, less dilution |

**Rule:** the large model's *input* tokens are the #1 cost driver. Precision retrieval + tight
context beats any decoding trick.

---

## 2. Latency optimizations

| Optimization | Where | Effect |
|---|---|---|
| **Streaming + instant ack** | [docs/08](08-answer-generation-and-postprocessing.md),[docs/10](10-slack-interface-and-serving.md) | perceived latency = TTFT (<1 s), not total |
| **Prompt‑prefix / KV caching** | [docs/08](08-answer-generation-and-postprocessing.md) | byte‑stable system prefix cached by vLLM → faster TTFT, cheaper decode |
| **Parallel pre‑process + warm embed** | [docs/06](06-query-preprocessing.md) | optimizer/source‑id/embed overlap → near‑zero added critical path |
| **Dense+sparse in one Qdrant round trip** | [docs/05](05-storage-vector-and-feature-store.md),[docs/07](07-hybrid-retrieval.md) | one network hop for hybrid |
| **Quantized ANN (int8/binary)** | [docs/05](05-storage-vector-and-feature-store.md) | faster search, smaller working set |
| **Batched embedding & reranking** | [docs/04](04-chunking-and-embedding.md),[docs/07](07-hybrid-retrieval.md) | GPU saturation → tens of ms for top‑50 rerank |
| **`needs_retrieval` gate** | [docs/06](06-query-preprocessing.md) | chitchat skips the whole pipeline |
| **Cheapest‑guard‑first** | [docs/09](09-guardrails-and-safety.md) | guardrails add ms, not a fixed LLM tax |
| **Adaptive top‑k by intent** | [docs/07](07-hybrid-retrieval.md) | simple queries do less work |
| **Everything expensive offline** | [docs/03](03-document-ingestion-and-enrichment.md),[docs/04](04-chunking-and-embedding.md) | enrichment/embedding never on the query path |

**Latency budget target (local, warm):** TTFT < 1 s, full answer < 8 s ([docs/00](00-overview-and-architecture.md) trace).

---

## 3. Cost optimizations

| Optimization | Where | Effect |
|---|---|---|
| **Right‑sized models per role** | [docs/02](02-configuration-and-model-providers.md) | small LLM for narrow tasks; large only for enrich + generate |
| **Quantized large model (AWQ/GPTQ)** | [docs/01](01-environment-and-infrastructure.md) | ~4× VRAM cut, 2–3× faster decode, negligible RAG‑quality loss |
| **Multi‑layer caching** | [docs/05](05-storage-vector-and-feature-store.md),[docs/14 §5] | embedding / query‑rewrite / retrieval / **response** caches |
| **Enrichment cache (content‑hash)** | [docs/03](03-document-ingestion-and-enrichment.md) | re‑ingest is mostly cache hits |
| **Near‑dup dedup** | [docs/03](03-document-ingestion-and-enrichment.md) | smaller index, less embed + retrieval waste |
| **Local‑first, hosted only for burst/quality** | [docs/02](02-configuration-and-model-providers.md),[docs/13](13-deployment-scaling-operations.md) | marginal query cost ≈ GPU‑seconds |
| **Abstain on low grounding** | [docs/08](08-answer-generation-and-postprocessing.md) | avoids the costliest failure: a confident wrong answer |
| **Cost/query dashboards + budget alerts** | [docs/12](12-observability-and-monitoring.md) | catch regressions before the bill does |

---

## 4. Throughput / scaling optimizations

| Optimization | Where | Effect |
|---|---|---|
| **vLLM continuous batching + paged attention** | [docs/01](01-environment-and-infrastructure.md) | 5–20× concurrent throughput vs. naive serving |
| **Per‑role endpoints** | [docs/01](01-environment-and-infrastructure.md),[docs/02](02-configuration-and-model-providers.md) | large model scales independently of small/embed/rerank |
| **Stateless app + horizontal scale** | [docs/13](13-deployment-scaling-operations.md) | linear scale behind a load balancer |
| **Semaphore + bounded queue** | [docs/10](10-slack-interface-and-serving.md) | graceful degradation, no meltdown under load |
| **Decoupled ingestion workers** | [docs/03](03-document-ingestion-and-enrichment.md),[docs/13](13-deployment-scaling-operations.md) | re‑ingest never touches query latency |

---

## 5. The caching hierarchy (single biggest cost/latency lever)

Every layer is version‑namespaced so a model/prompt/index change invalidates correctly — a stale
cache hit is worse than a miss.

```
Response cache        key = ans:{index_ver}:{norm_query_hash}      → skip entire pipeline
Retrieval cache       key = ret:{index_ver}:{query+filter_hash}    → skip search+rerank
Query-rewrite cache   key = qopt:{prompt_ver}:{query+history_hash} → skip optimizer LLM
Embedding cache       key = emb:{model_ver}:{text_hash}            → skip embedding (ingest+query)
Enrichment cache      key = enr:{enrich_ver}:{block_hash}          → skip large-LLM enrichment
Prompt-prefix KV      (in model server)                            → cheaper/faster decode
```

**Normalization** (lowercasing, whitespace, semantic‑equivalent rewrite) widens response/rewrite
cache hit rates. Hit rates are dashboarded ([docs/12](12-observability-and-monitoring.md)) — an unexpectedly low hit rate is a bug.

---

## 6. Quality optimizations (they also save tokens — better retrieval → smaller context)

| Optimization | Where | Effect |
|---|---|---|
| **Agentic query optimization** | [docs/06](06-query-preprocessing.md) | context resolution, expansion, decomposition → better recall |
| **Metadata‑filtered retrieval** | [docs/06](06-query-preprocessing.md),[docs/07](07-hybrid-retrieval.md) | precision + fewer false positives to rerank |
| **Hybrid (dense+sparse) + cross‑encoder rerank** | [docs/07](07-hybrid-retrieval.md) | best recall *and* precision; highest quality‑per‑dollar |
| **Content enrichment + table descriptions** | [docs/03](03-document-ingestion-and-enrichment.md) | tabular & context‑poor content becomes retrievable |
| **"Lost‑in‑the‑middle" ordering** | [docs/08](08-answer-generation-and-postprocessing.md) | best evidence where the model attends |
| **Business signals (freshness/authority)** | [docs/05](05-storage-vector-and-feature-store.md),[docs/07](07-hybrid-retrieval.md) | current/official sources win ties; deprecated demoted |
| **Grounding gate + abstain** | [docs/08](08-answer-generation-and-postprocessing.md),[docs/09](09-guardrails-and-safety.md) | trades a wrong answer for an honest "I don't know" |
| **Bounded self‑correction** | [docs/08](08-answer-generation-and-postprocessing.md) | recovers recoverable misses without unbounded cost |
| **Everything gated by eval** | [docs/11](11-evaluation-and-golden-datasets.md) | only data‑proven changes ship |

---

## 7. A tuning workflow (how to actually use this)

1. **Instrument** ([docs/12](12-observability-and-monitoring.md)) — get tokens/stage, TTFT breakdown, cost/query, cache hit rates.
2. **Find the biggest lever** — usually large‑model input tokens or retrieval precision.
3. **Change one thing** — e.g. lower `rerank_top_n`, raise `min_rerank_score`, tighten context budget.
4. **Gate on the golden set** ([docs/11](11-evaluation-and-golden-datasets.md)) — confirm quality holds while cost/latency drops.
5. **Ship behind config** ([docs/02](02-configuration-and-model-providers.md),[docs/13](13-deployment-scaling-operations.md)) — canary, watch post‑prod metrics, roll back if needed.
6. **Repeat** — RAG tuning is iterative; the eval + observability loop is what makes it safe.

---

## 8. Quick‑reference: default knobs

```yaml
retrieval:  { top_k_dense: 40, top_k_sparse: 40, fusion: rrf, rerank_top_n: 8, min_rerank_score: 0.15 }
generation: { max_context_chunks: 8, context_budget_tokens: 3500 }
chunking:   { target_tokens: 384, overlap: 0.15 }
models:     { large: quantized 32B, small: 7B, temp_small: 0.0, temp_large: 0.2 }
caching:    { response_ttl: 1h, rewrite_ttl: 1h, embedding_ttl: ∞(version-keyed) }
serving:    { stream: true, concurrency_semaphore: sized-to-GPU }
```

All are tuned per corpus by the eval harness — these are starting points, not gospel.

---

_End of the lifecycle documentation. Back to [README](../README.md) · [Overview](00-overview-and-architecture.md)._
