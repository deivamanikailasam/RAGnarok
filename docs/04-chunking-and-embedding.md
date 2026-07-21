# 04 — Chunking & Embedding

**What:** the diagram's **Table‑aware Chunking** and **Embedding Model embedding generation** boxes.
We turn enriched documents into retrieval units and their vector representations. Chunking quality
sets the ceiling on retrieval quality — get this wrong and no reranker saves you.

---

## 1. Chunking strategy: structure‑aware, not fixed‑size

Naive fixed‑size chunking (every 512 tokens) splits sentences, orphans context, and shreds tables.
RAGnarok chunks by **document structure** with the following rules:

1. **Respect section boundaries.** Never merge across H1/H2 boundaries; a chunk belongs to exactly
   one section so its metadata (topic, authority, freshness from [docs/03](03-document-ingestion-and-enrichment.md)) is coherent.
2. **Target 256–512 tokens with ~15% overlap.** Small enough for precise retrieval, overlapped so a
   fact split across a boundary is still recoverable.
3. **Semantic/recursive splitting within sections.** Split on paragraph → sentence, using a
   recursive splitter that prefers natural breakpoints; optionally a semantic splitter that breaks
   where adjacent‑sentence embedding similarity drops (topic shift).
4. **Table‑aware handling (the highlighted box).** Tables are *not* flattened into prose. Each table
   becomes its own chunk(s) carrying **three** representations (see §2).
5. **Contextual prefix.** Every chunk is prepended with a short, cheap context header —
   `Document: <title> › Section: <heading>. <one‑line doc summary>.` — before embedding. This is
   "contextual retrieval": it disambiguates chunks that are meaningless in isolation and measurably
   improves recall.

```python
# src/ragnarok/ingestion/chunking.py
def chunk_document(doc: EnrichedDocument) -> list[Chunk]:
    chunks = []
    for section in doc.sections:
        if section.is_table:
            chunks += chunk_table(section, doc)          # §2
        else:
            for piece in recursive_split(section.text, target=384, overlap=0.15):
                chunks.append(Chunk(
                    text=piece,
                    contextual_prefix=context_header(doc, section),   # §1.5
                    chunk_type="text",
                    metadata=inherit_metadata(doc, section),          # topics, acl, freshness…
                    token_count=count_tokens(piece)))
    return dedupe_and_hash(chunks)
```

### Why overlap + contextual prefix

- **Overlap** protects against boundary‑straddling facts at ~15% token cost.
- **Contextual prefix** (borrowed from Anthropic's "contextual retrieval") turns "It supports SSO."
  into "Document: Security Whitepaper › Section: Auth. It supports SSO." — now retrievable by a query
  about *the product's* SSO. Typical recall lift is large for FAQ/policy corpora.

---

## 2. Table‑aware chunking in detail

A single table produces a chunk with **three aligned representations**, so it's retrievable by
semantics *and* exact lookup *and* usable by the generator:

1. **NL description** (from enrichment, [docs/03](03-document-ingestion-and-enrichment.md)) → **embedded** for semantic recall.
2. **Structured Markdown/HTML table** → stored in payload, handed to the generator verbatim so it
   can read exact cell values.
3. **Row‑level keywords** (headers + entities) → feed the sparse/BM25 vector for exact‑term matches
   ("SKU‑4471", "us‑east‑1").

```python
def chunk_table(section, doc) -> list[Chunk]:
    desc = section.table_description          # NL, from enrichment
    md   = section.to_markdown()              # exact structure
    return [Chunk(
        text=desc,                            # embedded
        payload_table=md,                     # given to generator as-is
        contextual_prefix=context_header(doc, section),
        chunk_type="table",
        metadata=inherit_metadata(doc, section) | {"table_headers": section.headers})]
```

Large tables are split by row groups with the header row repeated in each part, so no chunk loses
its column meaning.

---

## 3. Embedding generation

We embed with **BGE‑M3**, which emits **dense + sparse (lexical)** vectors in one pass — so hybrid
retrieval ([docs/07](07-hybrid-retrieval.md)) needs only a single embedding model.

```python
# src/ragnarok/ingestion/embedding.py
async def embed_chunks(chunks: list[Chunk]) -> list[EmbeddedChunk]:
    texts = [c.contextual_prefix + "\n" + c.text for c in chunks]   # prefix embedded too
    out = []
    for batch in batched(texts, size=64):                            # §Optimizations
        cached, misses = cache_lookup(batch)                         # content-hash keyed
        if misses:
            resp = await embed_client.encode(misses, return_dense=True, return_sparse=True)
            cache_store(misses, resp)
        out += assemble(batch, cached, resp)
    return [EmbeddedChunk(**c.dict(), dense_vector=o.dense, sparse_vector=o.sparse,
                          embedding_model="bge-m3", embedding_version=EMB_VERSION)
            for c, o in zip(chunks, out)]
```

### Why BGE‑M3

- One model → dense **and** sparse → true hybrid without a second system.
- Multilingual, 8k context (handles long table descriptions), strong MTEB scores.
- Runs locally on modest GPU; swappable to `text-embedding-3-large` by config for scale‑out.

---

## 4. Optimizations introduced here

| Optimization | Wins |
|---|---|
| **Batched embedding (64–256/req)** | 5–20× throughput vs. one‑at‑a‑time; saturates GPU |
| **Embedding cache (content‑hash keyed)** | re‑embed only changed chunks; near‑zero cost on re‑ingest |
| **Contextual prefix** | large recall gain; cost is a few tokens/chunk, paid once offline |
| **Table three‑representation** | tables become first‑class retrievable + answerable |
| **15% overlap (not 50%)** | recovers boundary facts without doubling index size |
| **Structure‑bounded chunks** | coherent per‑chunk metadata → precise filtered retrieval |
| **Quantized vectors (int8/binary) option** | 4–32× smaller index, faster search, minor recall loss (rerank recovers it) |
| **Token‑count stored per chunk** | lets the generator pack context to a budget (docs/08, docs/14) |

**Tokenization note:** we count tokens with the *serving model's* tokenizer (not a generic one) so
chunk sizes and context budgets are exact for the model that will actually read them. Chunk size is
tuned per corpus by the eval harness ([docs/11](11-evaluation-and-golden-datasets.md)), not guessed.

---

## 5. Idempotency & versioning

- `chunk_id = sha256(doc_id + position + text)`; `embedding_version` bumps on model/prefix change.
- Changing the chunker or embedding model re‑processes only affected docs (diff by hash), and the
  store keeps both versions during a migration so retrieval never sees a half‑rebuilt index
  (blue/green collection swap, [docs/05](05-storage-vector-and-feature-store.md)).

## 6. What "done" looks like

- Tables are retrievable semantically and answerable exactly.
- Re‑embedding an unchanged corpus is ~100% cache hits.
- Chunk size + overlap are set from an eval sweep, with per‑model token counts recorded.

Next: [docs/05 — Storage: Vector & Feature Store](05-storage-vector-and-feature-store.md).
