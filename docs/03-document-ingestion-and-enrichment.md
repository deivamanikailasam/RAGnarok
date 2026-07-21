# 03 — Document Ingestion & Content Enrichment

**What:** the first two boxes of the offline plane — **Content Extraction (GDoc API + file
loaders)** and **Content Enrichment** (a large LLM that adds summaries, table descriptions and
**custom metadata attributes**). This is where we make what we index *richer than the raw source*,
which is the single biggest quality lever before retrieval even begins.

Diagram mapping: `Knowledge Documents (GDocs) → Content Extraction (GDoc API) → LLM (large) Table
enrichment + custom metadata (Content Enrichment)`.

---

## 1. Ingestion pipeline shape

Ingestion is an **idempotent, incremental, queue‑driven** batch job — never blocking the online
path.

```
Connectors → Extract → Normalize → Enrich (LLM) → (hand off to docs/04 chunk+embed)
   │            │          │            │
 GDoc API   text+tables  clean md   summaries, table descriptions,
 files,     structure    + ACLs     custom metadata attributes
 web, DB
```

Orchestrated with Redis + `arq` workers so ingestion is horizontally scalable and restartable:

```python
# src/ragnarok/ingestion/pipeline.py
async def ingest_document(ctx, uri: str):
    raw   = await extract(uri)                 # §2
    if unchanged(raw.content_hash): return "skipped"   # §5 idempotency
    norm  = normalize(raw)                     # §3
    enr   = await enrich(norm)                 # §4  (large LLM)
    await enqueue_chunk_and_embed(enr)         # docs/04
    upsert_feature_store(enr.custom_metadata)  # docs/05
```

---

## 2. Content Extraction

### 2.1 Google Docs (the diagram's GDoc API path)

Use the Docs API to preserve **structure** (headings, lists, and crucially **tables**), not just a
flat text dump — structure is what lets us do table‑aware chunking and metadata later.

```python
# src/ragnarok/ingestion/connectors/gdoc.py
def extract_gdoc(doc_id: str) -> SourceDocument:
    doc = docs_service.documents().get(documentId=doc_id).execute()
    blocks = []
    for el in doc["body"]["content"]:
        if "paragraph" in el:
            blocks.append(TextBlock(text=para_text(el), style=heading_level(el)))
        elif "table" in el:
            blocks.append(TableBlock(rows=parse_table(el["table"])))   # preserved as structured rows
    return SourceDocument(
        doc_id=doc_id, uri=f"gdoc://{doc_id}", source_type="gdoc",
        title=doc.get("title"), blocks=blocks,
        acl_tags=fetch_acls(doc_id),                    # §6 security
        fetched_at=now(), content_hash=hash_blocks(blocks))
```

For hosted Google, the Docs + Drive API run over HTTPS. For a truly air‑gapped install, the same
`SourceDocument` shape is produced by local loaders below — the downstream pipeline is
connector‑agnostic.

### 2.2 Other connectors (all normalize to `SourceDocument`)

| Source | Loader | Notes |
|---|---|---|
| PDF | `unstructured` / `PyMuPDF` + table detection (`camelot`/`pdfplumber`) | keep tables as `TableBlock` |
| Office (docx/pptx/xlsx) | `python-docx`, `python-pptx`, `openpyxl` | native table structure |
| Markdown/HTML | `markdown-it` / `selectolax` | heading tree preserved |
| Confluence/Notion/SharePoint | REST APIs | map spaces → `acl_tags` |
| Databases | SQL → row‑templated docs | schema‑aware |

**Why preserve structure:** headings become chunk boundaries and metadata; tables get special
handling (a table flattened to text is nearly useless for retrieval). See [docs/04](04-chunking-and-embedding.md).

---

## 3. Normalization

Convert every connector's output to clean, canonical **Markdown‑ish blocks** with stable IDs:
strip boilerplate/nav, normalize whitespace/unicode, de‑hyphenate PDF line breaks, keep heading
hierarchy, and attach source coordinates (page/section) for citations later.

**Optimization — dedup at the source:** near‑duplicate documents (copies, re‑exports) are detected
by MinHash/SimHash on normalized text and collapsed to one canonical doc with alias IDs. This cuts
index size, embedding cost, and the "5 identical chunks" retrieval failure mode.

---

## 4. Content Enrichment (the large‑LLM box)

The enrichment agent runs the **large LLM once per document** (offline, so cost/latency are
amortised over every future query) to produce structured, index‑time value:

```python
class Enrichment(BaseModel):
    doc_summary: str                       # 2–4 sentences, for coarse routing + display
    section_summaries: list[SectionSummary]
    table_descriptions: list[TableDesc]    # natural-language description per table
    custom_metadata: CustomMetadata        # → feature store (docs/05)
    keywords: list[str]                    # boosts BM25 recall (docs/07)

class CustomMetadata(BaseModel):
    topics: list[str]
    entities: list[str]                    # products, teams, regions
    doc_type: Literal["policy","runbook","spec","faq","contract","report","other"]
    audience: list[str]                    # e.g. ["enterprise","internal"]
    freshness_date: date | None            # effective/updated date for recency ranking
    authority: Literal["official","draft","deprecated"]
    access_tags: list[str]                 # mirrors ACLs for filterable security
```

### 4.1 Why enrich, and what each piece buys us

- **Table descriptions** — the diagram's "Table enrichment". A table like a pricing matrix is
  turned into sentences ("Enterprise tier includes a 30‑day refund window; Pro tier 14 days…").
  This description is embedded *alongside* the structured table so a semantic query actually
  retrieves it. **Biggest single quality win for tabular corpora.**
- **Document & section summaries** — used for cheap source routing (source identifier, [docs/06](06-query-preprocessing.md)),
  for the "contextual prefix" prepended to chunks ([docs/04](04-chunking-and-embedding.md)), and for display.
- **Custom metadata (topics, entities, doc_type, freshness, authority, access_tags)** — the
  diagram's **"Custom Metadata Attributes"**. These power **metadata‑filtered retrieval** (only
  search `doc_type=policy AND audience=enterprise`), **recency/authority reranking**, and
  **security filtering**. Cheap to store, huge for precision.

### 4.2 How — one structured pass, guarded

```python
async def enrich(doc: NormalizedDoc) -> EnrichedDocument:
    msgs = prompts.render("content_enricher", "latest",
                          title=doc.title, blocks=doc.render_for_llm())
    result = await resilience.call("llm_large", msgs,
                                   response_schema=Enrichment.model_json_schema())
    enr = Enrichment.model_validate_json(result.choices[0].message.content)
    return EnrichedDocument(**doc.dict(), **enr.dict(),
                            enrichment_model=settings.models.llm_large.model,
                            enrichment_version=prompts.version("content_enricher"))
```

---

## 5. Idempotency, incrementality & versioning

- **Content hashing** — each document and block is content‑hashed. Re‑ingesting an unchanged doc is
  a no‑op (`skipped`), so nightly runs are cheap and safe to re‑run.
- **Incremental enrichment** — only changed sections are re‑enriched; unchanged sections keep their
  cached enrichment keyed by `(block_hash, enrichment_version)`. Prompt/model bumps trigger targeted
  re‑enrichment, not a full corpus rebuild.
- **Provenance** — every enriched doc records which model + prompt version produced it, so an
  enrichment regression is traceable and reversible.

**Optimization — enrichment cache in Redis/Postgres:** `key = sha256(block_text + enrichment_version)`.
On a large corpus this turns most re‑ingests into cache hits and slashes the large‑LLM bill.

---

## 6. Security & governance at ingest (shift‑left)

- **ACL capture** — `acl_tags`/`access_tags` are extracted at ingest and stored on every chunk's
  payload, so retrieval can filter by the caller's entitlements ([docs/07](07-hybrid-retrieval.md)). Never
  index content you can't also access‑control.
- **PII classification** — documents are scanned (Presidio) at ingest; PII spans are tagged so
  output guardrails ([docs/09](09-guardrails-and-safety.md)) can redact, and highly sensitive docs can be
  excluded from the index by policy.
- **Source allow‑list** — only configured connectors/spaces are ingested; drive‑by URLs are not.

---

## 7. Optimizations introduced here

| Optimization | Wins |
|---|---|
| All enrichment **offline, once per doc** | per‑query latency/cost unaffected by enrichment depth |
| **Table → natural‑language description** | large recall/precision gain on tabular data |
| **Contextual summaries prepended to chunks** | fewer "orphaned chunk" retrieval misses (docs/04) |
| **Content‑hash idempotency + enrichment cache** | cheap incremental re‑ingest; big LLM‑cost cut |
| **Near‑dup dedup (MinHash)** | smaller index, lower embed cost, less redundant retrieval |
| **Structured (guided‑JSON) enrichment** | reliable metadata, no parse failures, fewer tokens |
| **Metadata + ACL at ingest** | filterable precision + security with zero online cost |

## 8. What "done" looks like

- Re‑ingesting an unchanged corpus is ~100% cache hits.
- Every enriched doc has table descriptions, summaries, and validated custom metadata.
- Each chunk will carry ACL + PII + freshness + authority metadata into the store.
- A prompt/model version bump re‑enriches only affected docs.

Next: [docs/04 — Chunking & Embedding](04-chunking-and-embedding.md).
