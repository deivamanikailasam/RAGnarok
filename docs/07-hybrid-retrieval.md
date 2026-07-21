# 07 — Hybrid Retrieval

**What:** the diagram's **Retrieval Process** — **Embedding‑model vector search** + **BM25
retriever / BM25 search**, fused and reranked into the **Retrieved chunks** the generator uses.
Hybrid retrieval is the quality core of the whole system.

Diagram mapping: `Processed queries + metadata filters → Embd. Model vector search + BM25 Search →
(fuse + rerank) → Retrieved chunks → Answer Generation`.

---

## 1. Why hybrid (and not just vectors)

| Method | Strong at | Weak at |
|---|---|---|
| **Dense (embeddings)** | semantics, paraphrase, "meaning" | exact IDs, codes, rare tokens, names, negation |
| **Sparse (BM25)** | exact terms, acronyms, part numbers, quoting | synonyms, paraphrase, semantic gaps |
| **Cross‑encoder rerank** | precise query↔passage relevance | too slow to run over the whole corpus |

Real queries need all three: dense for "what does the refund policy say", sparse for "SKU‑4471",
rerank to put the *single best* passage first. Hybrid consistently beats any single retriever on
recall **and** precision — this is well‑established and is why the diagram shows both search legs.

---

## 2. The retrieval pipeline

```
                 ┌─ dense search  (top 40) ─┐
Processed query ─┤                          ├─ RRF fusion → top 50 ─ rerank (cross-encoder) → top 8
   + filters     └─ sparse/BM25   (top 40) ─┘         │                       │
                                                      └─ + feature-store signals (freshness/authority)
```

```python
# src/ragnarok/retrieval/hybrid.py
async def retrieve(plan: QueryPlan, source: SourcePlan, user: User) -> list[RetrievalResult]:
    qfilter = build_qdrant_filter(source, user)          # metadata + access_tags (docs/06, docs/09)
    # one Qdrant call does dense + sparse in parallel server-side
    dense, sparse = await asyncio.gather(
        vs.search(vector=("dense",  embed(plan)),  filter=qfilter, limit=40),
        vs.search(vector=("sparse", sparse_embed(plan)), filter=qfilter, limit=40))
    fused = rrf_fuse(dense, sparse, k=60)                 # §3
    reranked = await rerank(plan.rewritten_query, fused[:50])   # §4
    final = apply_business_signals(reranked, feast_features(fused))   # §5
    return [r for r in final if r.rerank_score >= settings.retrieval.min_rerank_score][:8]
```

---

## 3. Fusion — Reciprocal Rank Fusion (RRF)

Dense cosine scores and BM25 scores aren't on the same scale, so we fuse by **rank**, not raw
score. RRF is robust, parameter‑light, and needs no score normalization:

```python
def rrf_fuse(dense, sparse, k=60):
    scores = defaultdict(float)
    for ranked in (dense, sparse):
        for rank, hit in enumerate(ranked):
            scores[hit.id] += 1.0 / (k + rank)
    return sorted(dedupe(dense, sparse), key=lambda h: scores[h.id], reverse=True)
```

Sub‑queries from decomposition ([docs/06](06-query-preprocessing.md)) are each retrieved and their results folded into the
same fusion pool, so a comparison query represents both sides.

**Why RRF over weighted‑sum:** no fragile per‑corpus weight tuning, resistant to score outliers,
and it just works across heterogeneous retrievers. (A tunable weight is available behind eval for
corpora that need lexical‑ or semantic‑leaning bias.)

---

## 4. Reranking — cross‑encoder

The fused top‑50 is reranked by **BGE‑reranker‑v2‑m3**, a cross‑encoder that reads the query and
each passage *together* (unlike bi‑encoder embeddings) for far sharper relevance:

```python
async def rerank(query: str, candidates) -> list[RetrievalResult]:
    pairs = [(query, c.text) for c in candidates]
    scores = await reranker_client.score(pairs, batch_size=32)   # local GPU service
    for c, s in zip(candidates, scores): c.rerank_score = s
    return sorted(candidates, key=lambda c: c.rerank_score, reverse=True)
```

**Why rerank is the best quality‑per‑dollar lever:** we retrieve *broad* (top‑40 each leg, high
recall) then rerank *narrow* (top‑8, high precision). The generator then sees a small, dense,
highly‑relevant context — which improves answer quality **and** cuts input tokens/cost/latency in
[docs/08](08-answer-generation-and-postprocessing.md). One cheap model call replaces stuffing 20 mediocre chunks into an expensive one.

---

## 5. Business signals from the feature store

After semantic reranking, we apply document‑level signals from Feast ([docs/05](05-storage-vector-and-feature-store.md)) as a **bounded**
adjustment — never overriding relevance, just breaking ties and demoting stale/deprecated content:

```python
def apply_business_signals(results, feats):
    for r in results:
        f = feats[r.doc_id]
        r.final_score = r.rerank_score \
            + 0.05 * recency_boost(f.freshness_days) \
            + 0.05 * authority_boost(f.authority) \
            - 0.20 * (f.authority == "deprecated")        # hard demote deprecated
    return sorted(results, key=lambda r: r.final_score, reverse=True)
```

**Why:** two passages can be equally relevant but one is the *current* policy and one is a
deprecated draft. Freshness/authority/popularity encode "which source should win" — exactly what the
diagram's feature store is for.

---

## 6. Optimizations introduced here

| Optimization | Wins |
|---|---|
| **Dense+sparse in one Qdrant collection** | single round trip, no separate BM25 cluster |
| **Broad retrieve → narrow rerank** | high recall *and* high precision; small final context |
| **RRF fusion** | robust, tuning‑free score combination |
| **int8‑quantized ANN + rerank recovery** | fast, small index; rerank restores any recall lost |
| **Batched reranking (GPU)** | top‑50 rerank in tens of ms |
| **Metadata + access filter pushed into search** | fewer candidates, faster, secure |
| **Adaptive `top_k`** by intent | factoids retrieve fewer; comparisons more — cost scales with need |
| **Retrieval result cache** (query+filter hash) | repeated queries skip search entirely |
| **`min_rerank_score` floor** | drops weak chunks → prevents "context dilution" in the LLM |

**Tokenization/latency link:** every chunk that survives to the generator costs input tokens on the
*large* model. Reranking to 8 tight chunks instead of 20 loose ones is one of the biggest
end‑to‑end cost and latency wins in the system ([docs/14](14-optimization-playbook.md)).

## 7. Failure modes & guards

- **Empty/low‑recall result** (over‑filter or novel query) → fallback to unfiltered access‑scoped
  search ([docs/06](06-query-preprocessing.md)); if still empty, generator is told "no context" and must abstain
  ([docs/08](08-answer-generation-and-postprocessing.md), [docs/09](09-guardrails-and-safety.md)) rather than hallucinate.
- **Reranker down** → degrade gracefully to RRF‑only ordering, labelled in the trace.
- **Duplicate near‑identical chunks** → deduped in fusion (content‑hash + cosine) so context isn't
  wasted on repeats.

## 8. What "done" looks like

- Hybrid beats dense‑only and sparse‑only on the golden set (recall@k and nDCG).
- Exact‑term queries (IDs, codes) and semantic queries both succeed.
- Deprecated/stale sources are demoted below current ones.

Next: [docs/08 — Answer Generation & Post‑Processing](08-answer-generation-and-postprocessing.md).
