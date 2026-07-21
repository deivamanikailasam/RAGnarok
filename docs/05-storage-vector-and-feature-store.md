# 05 — Storage: Vector & Feature Store

**What:** the diagram's **Offline Store** — the **Vector store** (embedded, indexed document chunks)
and the **Feature store** (Custom Metadata Attributes). This is the durable, queryable substrate the
online plane reads on every request.

---

## 1. Vector store — Qdrant

Qdrant holds each chunk as a point with **dense + sparse vectors** and a rich **payload** (the
metadata from [docs/03](03-document-ingestion-and-enrichment.md)/[docs/04](04-chunking-and-embedding.md)) used for filtering, security, and reranking signals.

### 1.1 Collection schema

```python
# src/ragnarok/stores/vector.py
client.create_collection(
    "chunks_v1",
    vectors_config={"dense": VectorParams(size=1024, distance=Distance.COSINE,
                                          quantization=ScalarQuantization(type="int8"))},
    sparse_vectors_config={"sparse": SparseVectorParams()},   # BM25/BGE-M3 lexical
)
# payload indexes for fast filtered search
for field in ["doc_type","audience","authority","topics","access_tags","entities"]:
    client.create_payload_index("chunks_v1", field, field_schema="keyword")
client.create_payload_index("chunks_v1", "freshness_date", field_schema="datetime")
```

### 1.2 Point (payload) layout

```json
{
  "id": "chunk_9f...",
  "vector": { "dense": [...1024...], "sparse": {"indices":[...], "values":[...]} },
  "payload": {
    "doc_id": "gdoc://1AbC",
    "text": "Enterprise tier includes a 30-day refund window ...",
    "table_markdown": "| Tier | Refund |\n|---|---|\n| Enterprise | 30 days |",
    "chunk_type": "table",
    "title": "Refund Policy", "section": "Enterprise",
    "doc_type": "policy", "audience": ["enterprise"], "authority": "official",
    "topics": ["refunds","billing"], "entities": ["Enterprise tier"],
    "access_tags": ["team:sales","tier:internal"],
    "freshness_date": "2026-03-01",
    "embedding_version": "bge-m3@1", "content_hash": "..."
  }
}
```

**Why Qdrant:** native sparse vectors mean dense + BM25 hybrid is **one collection, one query
round‑trip** (no separate Elasticsearch to operate); payload indexes make the source‑identifier's
metadata filters ([docs/06](06-query-preprocessing.md)) fast; int8 quantization shrinks the index ~4× with rerank
recovering any recall loss; single container, easy local ops.

### 1.3 Upsert (idempotent) and blue/green rebuilds

```python
def upsert(chunks: list[EmbeddedChunk], collection="chunks_v1"):
    client.upsert(collection, points=[to_point(c) for c in chunks])  # id = content hash → idempotent
```

Reindexing (new embedding model, new chunker) is a **blue/green** swap: build `chunks_v2`
alongside `chunks_v1`, validate with the eval suite ([docs/11](11-evaluation-and-golden-datasets.md)), then atomically repoint
the `chunks` alias. Online traffic never sees a partial index; rollback is repointing the alias.

---

## 2. Feature store — Feast (Custom Metadata Attributes)

The diagram draws the Feature store beside the Vector store for a reason: some attributes are
**document‑level features reused across many chunks and across stages** (source routing, reranking,
recency, authority, popularity). Duplicating these into every chunk payload is wasteful and hard to
update; a feature store gives a clean **offline (batch) / online (low‑latency lookup)** split.

### 2.1 What lives in the feature store vs. the payload

| Feature | Store | Why |
|---|---|---|
| chunk text, dense/sparse vectors | Qdrant payload | needed *inside* the search |
| filterable tags (doc_type, access_tags) | **both** (payload for filter; Feast for routing) | filtered search needs them in Qdrant; routing reads them cheaply from Feast |
| doc authority, freshness_date | Feast (online) | document‑level, updated independently of chunks |
| doc popularity / click‑through, feedback score | Feast (online) | updated continuously from usage (docs/11, docs/12) |
| entity → canonical source map | Feast | powers the source identifier (docs/06) |

### 2.2 Feature definitions

```python
# src/ragnarok/stores/features.py  (Feast)
document_stats = FeatureView(
    name="document_stats",
    entities=[Entity(name="doc_id")],
    schema=[Field("authority", String), Field("freshness_days", Int32),
            Field("popularity", Float32), Field("feedback_score", Float32),
            Field("doc_type", String)],
    online=True, source=postgres_source)   # offline: Postgres; online: Redis-backed
```

### 2.3 How it's used online (fast)

```python
feats = feast_store.get_online_features(
    features=["document_stats:authority","document_stats:freshness_days",
              "document_stats:popularity","document_stats:feedback_score"],
    entity_rows=[{"doc_id": d} for d in candidate_doc_ids]).to_dict()
# → mixed into the rerank score (docs/07) and source routing (docs/06)
```

**Why a feature store, not just payload duplication:** popularity/feedback/freshness change *daily*
from live usage; updating one Feast row per document is O(docs), while updating every chunk payload
is O(chunks) and races with search. Feast also gives point‑in‑time correctness for training/eval
data and one definition reused by retrieval, reranking, and evaluation.

---

## 3. Metadata store & registry (Postgres)

Postgres is the system of record for:
- **Document registry** — `doc_id`, source URI, content hash, enrichment/embedding versions, ACLs,
  ingest status (drives incremental re‑ingest, [docs/03](03-document-ingestion-and-enrichment.md)).
- **Feast offline store & registry.**
- **Prompt/version audit**, eval run history, feedback events.
- **Langfuse** backend ([docs/12](12-observability-and-monitoring.md)).

---

## 4. Caching layer — Redis

One Redis, several logical caches (all with TTLs and version‑namespaced keys):
- **Embedding cache** (ingest + query) — `emb:{model_ver}:{hash}`.
- **Query‑rewrite cache** ([docs/06](06-query-preprocessing.md)) — `qopt:{prompt_ver}:{hash}`.
- **Response cache** ([docs/08](08-answer-generation-and-postprocessing.md), [docs/14](14-optimization-playbook.md)) — `ans:{index_ver}:{norm_query_hash}`.
- **Feast online store** backing.

---

## 5. Optimizations introduced here

| Optimization | Wins |
|---|---|
| **int8 / binary quantized vectors** | 4–32× smaller index, faster ANN; rerank recovers recall |
| **Payload indexes on filter fields** | filtered retrieval stays O(log n), not full scan |
| **Sparse + dense in one collection** | hybrid in one round trip; no second search system to run |
| **Blue/green collection + alias** | zero‑downtime reindex, instant rollback |
| **Feature store for volatile doc features** | O(docs) updates, reused across stages, point‑in‑time correct |
| **Content‑hash IDs** | idempotent upserts; safe re‑runs |
| **HNSW params tuned per corpus** (`m`, `ef`) | recall/latency trade set by eval, not defaults |

---

## 6. Security & governance

- **`access_tags` on every payload** → retrieval filters to the caller's entitlements *inside* the
  search, so an unauthorized chunk is never even scored ([docs/07](07-hybrid-retrieval.md)).
- **Encryption at rest** on the volumes; Postgres row‑level security for the registry.
- **Deletion / right‑to‑be‑forgotten** — deleting a `doc_id` cascades: remove Qdrant points by
  `doc_id` filter, delete Feast rows, purge caches by namespace. Tested as an operational runbook
  ([docs/13](13-deployment-scaling-operations.md)).

## 7. What "done" looks like

- Filtered + hybrid search on 1–10M chunks returns in tens of ms locally.
- Reindex is blue/green with alias swap; rollback is one command.
- Deleting a document removes it from vectors, features, and caches verifiably.

Next: [docs/06 — Query Pre‑Processing](06-query-preprocessing.md).
