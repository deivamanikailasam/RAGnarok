# RAGnarok — Step‑by‑Step Implementation Guide

This is the **build guide**: the exact order you implement RAGnarok in, why each step exists, and
**why we chose each approach over the alternatives**, all framed around a concrete business use
case. Where a step needs deeper mechanics, it links to the matching deep‑dive in
[`docs/00`–`docs/14`](00-overview-and-architecture.md); this document is the *sequence and the
reasoning*, not a re‑explanation of every internal.

> **How to read this:** each step has **Business driver → What you build → How → Why this, not the
> alternatives → Definition of done.** Build in order — every step depends only on the ones before it.

---

## The business use case we are building for

We implement RAGnarok for a single, concrete scenario so every decision has a reason you can
check against reality. (The system is general; the use case keeps us honest.)

> **"Helios Assist" — an internal Customer‑Success knowledge assistant for a B2B SaaS company.**
> Customer‑Success (CS) agents handle live customer chats and need instant, *correct*, *cited*
> answers to questions like *"What's the refund window for an enterprise annual contract?"* or
> *"How do I enable SSO for a customer on the Growth plan?"* The knowledge lives in **Google Docs**
> (policies, pricing tables, product runbooks, contract templates). Agents ask in **Slack**.

**What this use case forces on the design (the requirements that drive every choice):**

| Business requirement | Consequence for the build |
|---|---|
| Pricing, contracts, and roadmap docs are **confidential** — cannot leave company infrastructure | **Fully local** by default; hosted APIs only as an opt‑in per‑model choice (Step 3) |
| A **wrong** answer to a customer is worse than a slow one (refund/SLA mistakes cost money & trust) | **Grounding gate + abstain** over "always answer" (Step 21); citations mandatory (Step 20) |
| Answers must differ by **customer tier** (enterprise ≠ consumer refund window) | **Metadata‑filtered retrieval** on `audience`/`tier` (Steps 8, 15) |
| Much knowledge lives in **tables** (pricing/SLA matrices) | **Table‑aware ingestion** over naive text splitting (Steps 9–11) |
| CS agents ask **short follow‑ups** in a live chat ("and for annual?") | **Query rewriting w/ thread context** (Step 14) |
| Answers must be **fast enough for live chat** | **Streaming + first token < 1s**; retrieve broad, rerank narrow (Steps 16, 19) |
| **Deprecated** policies must never be quoted as current | **Authority/freshness signals** demote stale docs (Step 13) |
| Security: an agent must not see docs their role can't access | **ACL filtering inside retrieval**, not in the prompt (Step 22) |
| Leadership needs proof it's accurate before rollout, and that it *stays* accurate | **Golden‑set gate (pre‑prod) + live scoring (post‑prod)** (Steps 24–25) |

Keep this table in view — every "why not the alternative" below traces back to one of these rows.

---

## Build order at a glance

```
PHASE A  Foundations         Steps 1–3    run models & infra locally, address them by role
PHASE B  Offline / ingest    Steps 4–13   turn GDocs into a searchable, enriched index
PHASE C  Online / answer     Steps 14–21  turn a question into a grounded, cited answer
PHASE D  Safety & serving    Steps 22–23  guardrails + Slack/API surface
PHASE E  Prove & operate     Steps 24–27  evaluation, observability, deploy, optimize
```

You build the offline plane before the online plane because **you cannot retrieve from an index
that doesn't exist** — and you build a thin end‑to‑end slice early (Step 16 note) so you get a
working demo fast, then deepen each stage.

---

# PHASE A — Foundations

## Step 1 — Stand up the local infrastructure

**Business driver:** confidential data ⇒ everything runs on our own box; reproducible so any
engineer can rebuild it. Deep dive: [`docs/01`](01-environment-and-infrastructure.md).

**What you build:** a Docker Compose stack — Qdrant (vector store), Postgres (registry + Feast +
Langfuse), Redis (cache + queue), Prometheus/Grafana (metrics), Langfuse (LLM tracing).

**How:** `docker compose -f docker/compose.core.yaml up -d`, then `ragnarok doctor` to health‑check
every dependency.

**Why this, not the alternatives:**

| Decision | Chosen | Rejected alternative | Why (business) |
|---|---|---|---|
| Where it runs | **Local Docker Compose** | Managed cloud SaaS (Pinecone, hosted OpenAI‑only) | Confidential pricing/contracts can't leave our network; Compose reproduces the whole stack on any workstation |
| Orchestration for a single box | **Docker Compose** | Kubernetes from day one | K8s is operational overhead we don't need at 1–30 users; Compose now, K8s later without code change (Step 26) |
| Stateful services | **Containers with volumes** | Bare‑metal installs | Disposable, versioned, identical across dev machines |

**Done when:** `ragnarok doctor` reports every component `ok` on a clean machine in < 15 min.

---

## Step 2 — Serve the models locally

**Business driver:** the LLMs and embedder must run on our hardware for the same confidentiality
reason, and be fast enough for live chat.

**What you build:** local model serving for four roles — a **large** instruct model (generation +
ingestion enrichment), a **small** instruct model (query/pre/post tasks), an **embedding** model,
and a **reranker**.

**How:** dev uses **Ollama** (`ollama pull qwen2.5:32b-instruct`, `…7b-instruct`, `bge-m3`);
prod uses **vLLM** with one OpenAI‑compatible endpoint per role. Reranker runs as a small
`FlagEmbedding` service.

**Why this, not the alternatives:**

| Decision | Chosen | Rejected | Why |
|---|---|---|---|
| Serving engine (prod) | **vLLM** | Ollama in prod, TGI, raw transformers | vLLM's continuous batching gives 5–20× throughput under concurrent CS agents; still local, still OpenAI‑compatible |
| Serving engine (dev) | **Ollama** | vLLM on a laptop | one‑command GPU model pulls; frictionless local iteration |
| Model sizing | **two sizes (large + small)** | one big model for everything | 80% of pipeline calls (query rewrite, source id, post‑process) are narrow tasks a 7B nails at a fraction of the cost/latency — see Step 3 |
| Large model precision | **4‑bit quantized (AWQ)** | full FP16 | ~4× less VRAM, ~2–3× faster decode, negligible RAG‑quality loss → fits our single‑GPU budget and hits the latency SLO |

**Done when:** each role answers on its own endpoint; `ragnarok doctor` shows the served model per role.

---

## Step 3 — Put every model behind a role‑based provider abstraction

**Business driver:** we want the *option* to burst the generation model to a hosted API under load,
or upgrade a model, **without touching code** — while keeping the confidential‑data stages local.
Deep dive: [`docs/02`](02-configuration-and-model-providers.md).

**What you build:** code addresses models by **role** (`llm_large`, `llm_small`, `embedding`,
`reranker`), never by vendor; roles resolve to an **OpenAI‑compatible** base URL from
`settings.yaml`. Add typed config (Pydantic), centralized retry/timeout/fallback, and
**guided‑JSON** structured output.

**Why this, not the alternatives:**

| Decision | Chosen | Rejected | Why |
|---|---|---|---|
| Model interface | **OpenAI‑compatible protocol as lingua franca** | Direct vendor SDKs (openai + anthropic + …) | Ollama, vLLM, OpenAI, Azure, Together all speak it → local⇄hosted is a URL string, no vendor lock‑in, one place for caching/tracing |
| Framework | **Thin abstraction + LangGraph for control flow** | LangChain/LlamaIndex "do‑everything" chains | We need an *inspectable, deterministic* state machine we can trace and gate; heavy framework magic hides exactly the parts we must audit for a money‑touching assistant |
| Agent output | **Constrained (guided) JSON via schema** | "Please return JSON" prompting + regex parsing | Invalid JSON becomes *impossible*, cuts output tokens 30–70%, removes a whole class of prod failures |
| Config | **Typed YAML, fail‑fast at startup** | scattered env vars / hardcoded | misconfig is a startup error, not a 3 a.m. incident |

**Done when:** switching `llm_large` from local to `gpt-4o` (or back) is a one‑line config change with
zero code edits, and every agent call returns a validated object.

---

# PHASE B — Offline plane: build the index

## Step 4 — Connect the Google Docs source & normalize

**Business driver:** the knowledge is in GDocs and their **structure matters** (headings, and
especially pricing/SLA **tables**). Deep dive: [`docs/03`](03-document-ingestion-and-enrichment.md).

**What you build:** a GDoc connector (Docs API) that preserves heading hierarchy and **tables as
structured rows**, plus normalization to canonical blocks with stable IDs; capture **ACLs** and
**PII flags** at read time. All connectors emit the same `SourceDocument` shape.

**Why this, not the alternatives:**

| Decision | Chosen | Rejected | Why |
|---|---|---|---|
| Extraction | **Docs API preserving structure** | Export to flat PDF/text then parse | flattening destroys table structure — fatal for a pricing/SLA corpus; the API gives clean headings + tables directly |
| Security capture | **Grab ACLs at ingest** | apply access rules later, in the app | you can't retrofit "who may see this" reliably; capturing it with the content is the only trustworthy source (Step 22) |

**Done when:** a GDoc with a pricing table ingests into structured blocks with ACLs and a content hash.

---

## Step 5 — Make ingestion idempotent & incremental

**Business driver:** policies change weekly; nightly re‑ingest must be cheap and safe to re‑run.

**What you build:** content‑hash every doc/block; unchanged docs are a no‑op; a document registry
(Postgres) tracks status and versions; run it on a Redis queue with `arq` workers.

**Why this, not the alternatives:** full re‑embed of the whole corpus every night is wasteful and
slow; **hash‑diffing** re‑processes only what changed. A queue decouples ingestion from serving so a
big refresh never slows live CS chats.

**Done when:** re‑ingesting an unchanged corpus is ~100% skips.

---

## Step 6 — Enrich content with the large LLM (the quality multiplier)

**Business driver:** raw doc text retrieves poorly; a pricing table is invisible to semantic search.
We enrich **once, offline**, so every future query benefits at no online cost. Deep dive:
[`docs/03`](03-document-ingestion-and-enrichment.md).

**What you build:** a large‑LLM pass per document producing **table descriptions** (turn the SLA
matrix into sentences), **doc/section summaries**, and **custom metadata** (`doc_type`, `audience`/
tier, `topics`, `entities`, `freshness_date`, `authority`, `access_tags`).

**Why this, not the alternatives:**

| Decision | Chosen | Rejected | Why (business) |
|---|---|---|---|
| Table handling | **LLM natural‑language description + keep structure** | index the raw table text | "\| Enterprise \| 30 days \|" doesn't match "how long can enterprise customers get a refund"; the description does |
| When to enrich | **Offline, once per doc** | enrich per query at request time | enrichment is expensive; doing it per query would blow the live‑chat latency budget and repeat work |
| Metadata source | **LLM‑extracted, structured, human‑reviewable** | manual tagging by SMEs | manual tagging doesn't scale to a growing corpus and rots; LLM extraction is consistent and cheap offline |

**Done when:** every doc has table descriptions, summaries, and validated tier/authority/freshness metadata.

---

## Step 7 — Cache enrichment; dedupe near‑duplicates

**Business driver:** keep the offline LLM bill and index size down as the corpus grows.

**What you build:** an enrichment cache keyed by `(block_hash, enrichment_version)`; MinHash/SimHash
dedup of copy‑pasted docs.

**Why:** re‑ingest becomes mostly cache hits (only changed sections re‑enriched); dedup shrinks the
index and kills the "five identical chunks" retrieval failure. Alternative — reprocessing
everything each run — is pure waste on a mostly‑stable corpus.

**Done when:** a prompt/model version bump re‑enriches only affected docs.

---

## Step 8 — Table‑aware chunking

**Business driver:** chunking quality caps retrieval quality; our tables must stay answerable.
Deep dive: [`docs/04`](04-chunking-and-embedding.md).

**What you build:** structure‑bounded chunks (respect section boundaries), ~256–512 tokens with
~15% overlap, a **contextual prefix** (`Document › Section. <summary>`) on each chunk, and **tables
as their own chunks** carrying three representations (NL description embedded, exact markdown for the
generator, header keywords for BM25).

**Why this, not the alternatives:**

| Decision | Chosen | Rejected | Why |
|---|---|---|---|
| Chunker | **Structure/semantic‑aware** | fixed 512‑token windows | fixed windows split sentences and shred tables; structure‑aware keeps a chunk's metadata (tier/authority) coherent |
| Overlap | **~15%** | 0% / 50% | 0% loses boundary facts; 50% doubles index cost; 15% recovers straddling facts cheaply |
| Context | **Prefix each chunk with doc/section context** | embed the bare chunk | "It's 30 days." is meaningless alone; the prefix makes it retrievable by "enterprise refund window" |

**Done when:** a pricing table is retrievable semantically *and* answerable with exact cell values.

---

## Step 9 — Embed with a hybrid‑capable model

**Business driver:** CS questions mix semantics ("how long to get money back") with exact terms
(plan names, SKUs, region codes). We need both. Deep dive: [`docs/04`](04-chunking-and-embedding.md), [`docs/07`](07-hybrid-retrieval.md).

**What you build:** embed chunks (prefix + text) with **BGE‑M3**, which emits **dense + sparse**
vectors in one pass; batch the calls; cache by content hash.

**Why this, not the alternatives:**

| Decision | Chosen | Rejected | Why |
|---|---|---|---|
| Embedding model | **BGE‑M3 (dense+sparse, local)** | OpenAI `text-embedding-3` | confidentiality (local) + it gives dense *and* lexical vectors from one model, so hybrid needs no second system |
| Throughput | **Batched (64–256/req) + cache** | one‑at‑a‑time | 5–20× faster ingest; re‑embed only changed chunks |

**Done when:** re‑embedding an unchanged corpus is ~100% cache hits; each chunk has dense + sparse vectors.

---

## Step 10 — Vector store: Qdrant with payload + filtering

**Business driver:** we must filter by customer tier and by ACL *inside* the search, and do hybrid
in one round trip for latency. Deep dive: [`docs/05`](05-storage-vector-and-feature-store.md).

**What you build:** a Qdrant collection with dense **and** sparse vectors, int8 quantization, and
payload indexes on `doc_type`, `audience`, `authority`, `topics`, `access_tags`, `freshness_date`.

**Why this, not the alternatives:**

| Decision | Chosen | Rejected | Why |
|---|---|---|---|
| Vector DB | **Qdrant** | pgvector | pgvector lacks first‑class sparse vectors + fast payload filtering at scale; we'd bolt on Elasticsearch for BM25 → two systems |
| Vector DB | **Qdrant** | Pinecone/Weaviate Cloud | confidentiality (must be local/self‑hosted); Qdrant runs in one container and has native hybrid |
| Lexical search | **Sparse vectors in Qdrant** | separate Elasticsearch/OpenSearch cluster | one store, one query round trip for hybrid; no extra cluster to operate for a single‑box deployment |
| Index size | **int8 quantized vectors** | full‑precision | 4× smaller/faster; the reranker (Step 19) recovers any recall lost |

**Done when:** filtered hybrid search on the corpus returns in tens of ms locally.

---

## Step 11 — Feature store (Feast) for volatile document signals

**Business driver:** "which source should win" depends on **freshness, authority, and popularity**
that change daily from usage — these are document‑level, reused across stages, and shouldn't be
duplicated into every chunk. Deep dive: [`docs/05`](05-storage-vector-and-feature-store.md).

**What you build:** Feast feature views for `authority`, `freshness_days`, `popularity`,
`feedback_score`, `doc_type`, with an offline (Postgres) / online (Redis) split.

**Why this, not the alternatives:** duplicating these into chunk payloads means O(chunks) updates
that race with search; Feast gives O(docs) updates and one definition reused by retrieval, reranking,
routing, and evaluation. Alternative — recompute on the fly — can't capture live popularity/feedback.

**Done when:** online feature lookup for a batch of `doc_id`s returns in low ms.

---

## Step 12 — Blue/green index & versioning

**Business driver:** changing the embedding model or chunker must never serve CS agents a
half‑rebuilt index.

**What you build:** build `chunks_vN+1` beside the live one, validate with eval (Step 24), then swap
an **alias** atomically; keep the old version for instant rollback.

**Why:** the alternative — mutating the live collection in place — exposes partial state and has no
rollback. Blue/green is zero‑downtime and reversible.

**Done when:** a reindex is an alias swap; rollback is one command.

---

## Step 13 — Wire freshness/authority as retrieval signals

**Business driver:** deprecated policies must never be quoted as current. (Implemented in retrieval,
Step 19, but the *data* is established here.)

**What you build:** ensure `authority` (`official`/`draft`/`deprecated`) and `freshness_date` from
enrichment flow into both the Qdrant payload (for hard filters) and Feast (for soft ranking boosts).

**Why:** business correctness — a wrong *but confident* refund policy is a real cost. We encode
"current & official wins" as data, not as a hope in the prompt.

**Done when:** deprecated docs can be demoted/excluded by config.

---

# PHASE C — Online plane: answer a question

> **Thin‑slice tip:** before deepening this phase, wire the simplest end‑to‑end path — embed query →
> Qdrant top‑k → stuff into the large LLM → return — so you have a working demo. Then add each step
> below and measure the lift with the eval harness (Step 24). Build the skeleton, then the muscle.

## Step 14 — Query pre‑processing (optimizer + source identifier)

**Business driver:** CS agents ask terse follow‑ups ("and for annual?") and answers must be scoped
to the right tier. Deep dive: [`docs/06`](06-query-preprocessing.md).

**What you build:** two **small‑LLM** agents (parallel, `temp=0`, guided JSON, cached): a **query
optimizer** (resolve thread context, rewrite, expand acronyms, decompose comparisons, gate chitchat)
and a **source identifier** (emit metadata filters like `doc_type=policy, audience=enterprise`).

**Why this, not the alternatives:**

| Decision | Chosen | Rejected | Why (business) |
|---|---|---|---|
| Fix the query first | **Rewrite + expand before retrieving** | embed the raw question | "and for annual?" retrieves nothing useful; context resolution is what makes live follow‑ups work |
| Scope the search | **Metadata filters from a source identifier** | always search the whole corpus | enterprise vs consumer refund windows differ; filtering removes wrong‑tier false positives before ranking |
| Model size | **Small LLM** | large LLM | these are narrow schema‑bound tasks; a 7B does them in ~150–300 ms at a fraction of the cost |
| Safety valve | **Fallback to unfiltered (access‑scoped) search if low‑confidence/few hits** | trust the filter absolutely | an over‑eager filter must never starve retrieval and make us say "I don't know" wrongly |

**Done when:** thread follow‑ups resolve correctly and tier filters improve precision without hurting recall.

---

## Step 15 — Enforce tier & access scoping as data (not prompt)

**Business driver:** an agent must only see what their role permits; tier scoping must be reliable.

**What you build:** build the Qdrant filter from the source identifier's suggestion **intersected
with the caller's real entitlements** — the model can only *narrow*, never *widen*, access.

**Why this, not the alternatives:** telling the LLM "don't reveal restricted docs" is weak and
bypassable (prompt injection). Filtering in the query means an unauthorized chunk is **never scored,
never seen**. Security by construction beats security by instruction.

**Done when:** an ACL test suite proves no unauthorized chunk is retrievable for a caller.

---

## Step 16 — Hybrid retrieval: dense + sparse + fusion

**Business driver:** mixed semantic/exact questions need both retrievers. Deep dive: [`docs/07`](07-hybrid-retrieval.md).

**What you build:** run dense and sparse search (top‑40 each) with the filter, fuse with **Reciprocal
Rank Fusion (RRF)**.

**Why this, not the alternatives:**

| Decision | Chosen | Rejected | Why |
|---|---|---|---|
| Retrieval | **Hybrid (dense + BM25)** | dense‑only | embeddings blur exact plan names/SKUs/region codes; BM25 nails them; together they beat either alone on recall *and* precision |
| Score combination | **RRF (rank‑based)** | weighted sum of raw scores | cosine and BM25 aren't on the same scale; RRF needs no fragile per‑corpus weight tuning and resists outliers |

**Done when:** exact‑term and semantic queries both succeed; hybrid beats dense‑only on the golden set.

---

## Step 17 — Rerank broad→narrow with a cross‑encoder

**Business driver:** the generator must see a *small, dense, highly‑relevant* context — for quality,
latency, and token cost. Deep dive: [`docs/07`](07-hybrid-retrieval.md).

**What you build:** rerank the fused top‑50 with **BGE‑reranker‑v2‑m3**, keep top‑8 above a score
floor.

**Why this, not the alternatives:** a bi‑encoder (embedding) scores query and passage separately; a
**cross‑encoder** reads them together and is far more precise. Retrieving broad (recall) then
reranking narrow (precision) is the **highest quality‑per‑dollar lever in RAG** — one cheap model
call replaces stuffing 20 mediocre chunks into an expensive generator. Skipping rerank (top‑k
straight from fusion) is the most common quality mistake.

**Done when:** the generator receives ≤8 tight chunks; answer quality rises while input tokens fall.

---

## Step 18 — Apply business signals (freshness/authority)

**Business driver:** current & official sources must win ties; deprecated demoted (from Step 13).

**What you build:** a **bounded** post‑rerank adjustment using Feast signals — small recency/authority
boosts, a hard demote for `deprecated`.

**Why bounded, not dominant:** relevance must stay primary; signals only break ties and demote stale
content. Letting popularity override relevance would surface popular‑but‑wrong docs.

**Done when:** given two equally relevant passages, the current/official one ranks above the deprecated one.

---

## Step 19 — Assemble a token‑budgeted, cited context

**Business driver:** predictable latency/cost for live chat, and every answer must be auditable.
Deep dive: [`docs/08`](08-answer-generation-and-postprocessing.md).

**What you build:** pack reranked chunks into a fixed **token budget** with numbered `[i]` citation
markers; hand tables over as **exact markdown**; place the best evidence at the **start and end**
("lost‑in‑the‑middle" mitigation).

**Why this, not the alternatives:** "concatenate all chunks" causes context dilution (quality *drops*)
and unpredictable cost. A budget + numbering gives predictable latency, precise citations, and a
substrate for the grounding check.

**Done when:** context is bounded, numbered, and tables carry exact values.

---

## Step 20 — Generate, then post‑process

**Business driver:** grounded, cited, well‑formatted answers; do cheap tasks cheaply.
Deep dive: [`docs/08`](08-answer-generation-and-postprocessing.md).

**What you build:** the **large LLM** generates a **streamed** answer constrained to "answer only
from context, cite with `[i]`, don't recompute table numbers, abstain if insufficient." A **small
LLM** then post‑processes: resolve citations to source links, extract atomic claims (for grounding),
format for Slack, suggest follow‑ups.

**Why this, not the alternatives:**

| Decision | Chosen | Rejected | Why |
|---|---|---|---|
| Response delivery | **Stream tokens** | wait for the full answer | live chat needs first token < 1s; perceived latency = time‑to‑first‑token |
| Post‑processing | **Small LLM** | do it on the large model | citations/formatting/claim‑splitting are mechanical; the large model would cost ~10× the tokens for no gain |
| Grounding prompt | **"Only from context, else abstain"** | let the model use general knowledge | a plausible‑but‑ungrounded refund policy is a business risk; we forbid it |

**Done when:** answers stream, cite sources, and every claim maps to a chunk.

---

## Step 21 — Grounding gate & abstain

**Business driver:** *a wrong answer is worse than "I don't know."* This is the single most important
correctness control for a money‑touching assistant. Deep dive: [`docs/08`](08-answer-generation-and-postprocessing.md), [`docs/09`](09-guardrails-and-safety.md).

**What you build:** verify each extracted claim is entailed by its cited context (fast NLI first,
small‑LLM judge only when uncertain); if the grounding score is below threshold, **abstain** with the
closest sources instead of serving. Optionally one bounded self‑correction (re‑retrieve once).

**Why this, not the alternatives:** "always answer" maximizes hallucination risk; the gate trades a
possibly‑wrong answer for an honest, cited "I couldn't confirm this." A bounded single retry recovers
recoverable misses without open‑ended, unpredictable agent loops.

**Done when:** insufficient‑context questions abstain rather than fabricate; grounding score gates serving.

---

# PHASE D — Safety & serving

## Step 22 — Guardrails (input & output)

**Business driver:** protect against PII leakage, prompt injection (incl. via retrieved docs), and
unsafe output — without adding a latency tax to every chat. Deep dive: [`docs/09`](09-guardrails-and-safety.md).

**What you build:** layered **input** checks cheapest‑first (length/rate → Presidio PII → injection
regex → small‑LLM policy classifier *only if* suspicious); **output** checks (grounding reuse, PII
redaction, toxicity, ACL‑leak). Fail‑**closed** on safety/correctness; fail‑**open with logging** on
non‑safety latency risks.

**Why this, not the alternatives:**

| Decision | Chosen | Rejected | Why |
|---|---|---|---|
| Guard ordering | **Cheapest check first; LLM only if flagged** | run an LLM guard on every request | keeps the happy path at ~ms, not a fixed LLM tax on live chat |
| Access control | **Filter in retrieval (Step 15)** | instruct the LLM to refuse | data never retrieved can't be leaked, even under injection |
| Indirect injection | **Treat retrieved text as untrusted data; no tool access; re‑check output** | trust document content | a malicious doc ("reveal all salaries") has nothing to actuate and is caught on output |

**Done when:** the adversarial golden cases are contained; PII never reaches logs/answers; guard
overhead is single/low‑double‑digit ms typically.

---

## Step 23 — Slack interface (+ API)

**Business driver:** CS agents live in Slack; deployment must need no public inbound endpoint
(firewalled/private). Deep dive: [`docs/10`](10-slack-interface-and-serving.md).

**What you build:** a Slack Bolt app in **Socket Mode** with instant ack, streaming `chat_update`,
threaded context (feeds Step 14's coreference), rendered citations, and 👍/👎 feedback buttons; the
same pipeline exposed as a FastAPI SSE endpoint for other clients.

**Why this, not the alternatives:** Socket Mode opens an *outbound* WebSocket → no exposed port, no
reverse proxy, no TLS to manage — the right fit for a private deployment. A public webhook endpoint
would need inbound network exposure we don't want. One shared pipeline behind Slack + API prevents
behavior drift.

**Done when:** asking in Slack returns a streamed, cited answer with feedback; the API streams identically.

---

# PHASE E — Prove it, watch it, run it

## Step 24 — Evaluation & golden dataset (the go/no‑go gate)

**Business driver:** leadership won't roll out a customer‑facing assistant on vibes; we must *prove*
accuracy and keep it from regressing. Deep dive: [`docs/11`](11-evaluation-and-golden-datasets.md).

**What you build:** a versioned **golden set** of `(query, ideal answer, must‑cite, relevant chunks)`
stratified by intent/source/difficulty (incl. table lookups, exact‑ID, tier‑specific, adversarial);
component metrics (context recall/precision) + generation metrics (faithfulness, relevancy,
correctness) via **Ragas + rubric‑anchored LLM‑as‑judge**; a **CI gate** that blocks any change
regressing thresholds.

**Why this, not the alternatives:**

| Decision | Chosen | Rejected | Why |
|---|---|---|---|
| Quality signal | **Golden set + automated metrics gate** | manual spot‑checking | spot checks miss regressions that a prompt tweak causes on a query type you didn't try |
| Judging | **Structured, reference‑guided, calibrated LLM‑judge** | raw "score 1–10" prompt / human‑only | calibrated structured judging is low‑variance, cheap, and we validate it against human labels so we can trust it |
| Where metrics live | **Component + end‑to‑end** | end‑to‑end only | component metrics tell us whether *retrieval* or *generation* failed → fast fixes |

**Done when:** `ragnarok eval --suite golden` produces a per‑metric, per‑stratum report and CI blocks regressions.

---

## Step 25 — Post‑prod continuous evaluation

**Business driver:** the corpus and questions drift; we must catch quality drops before customers do.

**What you build:** sample live answers (100% of 👎/low‑grounding) into a nightly batch, score with
the same judges + real feedback, alert on drift, and feed failures back into the golden set and
per‑document feedback scores (Step 11).

**Why:** launch‑and‑forget is how RAG quietly rots. Continuous scoring makes production a graded,
self‑hardening loop. Alternative — rely only on user complaints — means customers find the bugs.

**Done when:** live traffic is scored nightly with drift alerts, and prod failures become new golden cases.

---

## Step 26 — Observability (do this alongside everything, not last)

**Business driver:** you can't optimize, debug, or trust what you can't see. Deep dive: [`docs/12`](12-observability-and-monitoring.md).

**What you build:** one **Langfuse** trace per request (nested spans per stage with tokens/cost/model+
prompt version), **Prometheus/Grafana** SLIs (TTFT, cost/query, grounding, abstain rate, dependency
health), structured logs correlated by `trace_id`, and cost accounting even for local models
(GPU‑seconds).

**Why this, not the alternatives:** generic APM doesn't understand tokens, prompt versions, or
grounding. LLM‑native tracing (Langfuse) links a trace to its eval score and user feedback on one
object — essential to root‑cause an agentic pipeline to a specific stage/model/prompt.

**Done when:** any incident is root‑caused to a stage/model/prompt in minutes; dashboards show the RAG SLIs live.

---

## Step 27 — Deploy, scale, and optimize

**Business driver:** grow from one workstation to org‑wide without rewrites; keep it fast and cheap.
Deep dives: [`docs/13`](13-deployment-scaling-operations.md), [`docs/14`](14-optimization-playbook.md).

**What you build:** CI (lint/test → build → **eval gate** → ship); blue/green app + alias‑swap index;
per‑role scaling (vLLM continuous batching, then replicas); graceful degradation (fallback ladder,
RRF‑only if reranker down, response cache if Qdrant slow); the caching hierarchy and the tuning loop.

**Why this, not the alternatives:**

| Decision | Chosen | Rejected | Why |
|---|---|---|---|
| Scale path | **Same code, config‑only from local→cluster; hybrid (local data, burst generation to hosted)** | rewrite for prod | the Step‑3 abstraction means scaling is endpoints + replicas, and we can keep confidential stages local while bursting only generation |
| Cost control | **Cache hierarchy + right‑sized models + tight context** | throw a bigger GPU at it | the biggest levers are large‑model *input tokens* and cache hit rate, not raw hardware |
| Reliability | **Graceful degradation, labelled "degraded"** | hard‑fail on any dependency loss | CS agents get a (labelled) answer during partial outages instead of nothing |

**Done when:** one command runs it locally; config alone deploys it to a cluster; releases/reindexes
are zero‑downtime; failures degrade visibly; cost/query and TTFT stay within budget.

---

## Appendix — Decision log (the "why not" at a glance)

| Area | We chose | Over | One‑line reason |
|---|---|---|---|
| Hosting | Fully local (Qdrant/Ollama/vLLM) | Hosted RAG SaaS | confidential pricing/contract data can't leave our network |
| Model interface | OpenAI‑compatible role abstraction | vendor SDKs | local⇄hosted is a config line; no lock‑in |
| Models | Two sizes (large + small) | one big model | narrow tasks are 80% of calls; small model = fraction of cost/latency |
| Large model | 4‑bit quantized | FP16 | fits single GPU, faster, negligible RAG‑quality loss |
| Framework | Thin + LangGraph | LangChain/LlamaIndex chains | inspectable, traceable, gate‑able control flow for a money‑touching bot |
| Agent I/O | Guided JSON | prompt‑and‑parse | reliability + 30–70% fewer output tokens |
| Extraction | Structure‑preserving GDoc API | flat export | tables survive |
| Enrichment | Offline LLM, once | per‑query / manual tags | quality without online cost; scales |
| Chunking | Table‑aware, structure‑bounded, contextual prefix | fixed windows | tables answerable; chunks self‑describing |
| Embeddings | BGE‑M3 (dense+sparse, local) | OpenAI embeddings | local + hybrid from one model |
| Vector DB | Qdrant | pgvector / Pinecone | native hybrid + filtering, self‑hosted, one container |
| Lexical | Sparse vectors in Qdrant | separate Elasticsearch | one store, one round trip |
| Feature store | Feast | duplicate into payloads | O(docs) updates for volatile signals |
| Query | Optimize + source‑id (small LLM) | embed raw query | follow‑ups + tier scoping work |
| Access control | Filter in retrieval | prompt instruction | security by construction |
| Retrieval | Hybrid + RRF | dense‑only / weighted sum | recall+precision; no fragile weights |
| Reranking | Cross‑encoder broad→narrow | top‑k from fusion | biggest quality‑per‑dollar lever |
| Context | Token‑budgeted, cited, edge‑placed | concatenate all | predictable cost, no dilution, auditable |
| Generation | Stream + "only from context" | wait / free knowledge | fast + grounded |
| Correctness | Grounding gate + abstain | always answer | wrong answer worse than "I don't know" |
| Guardrails | Layered, cheapest‑first | LLM guard on every call | safety without a latency tax |
| Chat surface | Slack Socket Mode | public webhook | no inbound exposure for a private deploy |
| Quality assurance | Golden gate + post‑prod scoring | manual QA | proven pre‑launch, protected after |
| Observability | LLM‑native tracing (Langfuse) | generic APM | tokens/prompt/grounding per stage |
| Scale | Config‑only, hybrid burst | prod rewrite | grow without re‑architecting |

---

_See also: [README](../README.md) · architecture & internals in [`docs/00`–`docs/14`](00-overview-and-architecture.md)._
