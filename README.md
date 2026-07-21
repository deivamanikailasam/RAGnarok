# RAGnarok — A Production‑Grade, Fully‑Local Agentic RAG System

> An end‑to‑end, production‑grade **Agentic Retrieval‑Augmented Generation** platform that runs
> **entirely on your own machine / private infrastructure**, with the option to transparently
> swap in **OpenAI or any other model API** when you need to scale out.

RAGnarok is the reference implementation of the architecture below: agentic query & content
enrichment for higher‑quality retrieval inputs, hybrid retrieval (vector + BM25 + LLM reasoning),
and built‑in evaluation, golden datasets and continuous performance metrics.

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                         RAGnarok — Agentic RAG Platform                            │
│                                                                                    │
│   User ⇄ Slack Interface ⇄ ┌────────── Pre‑Process ──────────┐                     │
│                            │ query optimizer · source id     │                     │
│                            └───────────────┬─────────────────┘                     │
│                                            ▼                                        │
│                            ┌────────── Retrieval ────────────┐                      │
│                            │ vector search + BM25 + rerank   │                      │
│                            └───────────────┬─────────────────┘                      │
│                                            ▼                                        │
│                            ┌──────── Answer Generation ──────┐                      │
│                            │ large LLM generator + post‑proc │                      │
│                            └─────────────────────────────────┘                      │
│                                                                                    │
│   Offline: Ingest → Enrich (LLM) → Table‑aware chunk → Embed → Vector+Feature store │
│   Cross‑cutting: Guardrails · Observability · Pre‑prod & Post‑prod Evaluation       │
└──────────────────────────────────────────────────────────────────────────────────┘
```

## Why "fully local"?

Many organisations cannot send proprietary documents to a third‑party API. RAGnarok is designed
so that **every component has a local, open‑source default** — LLMs (via Ollama/vLLM), embeddings
(BGE‑M3), vector store (Qdrant), feature store (Feast), evaluation (Ragas), tracing (Langfuse),
metrics (Prometheus/Grafana). A single environment variable per model role lets you point any
individual model at a hosted API (OpenAI, Anthropic, Azure, Bedrock, Together, vLLM) without
touching application code, because **everything talks the OpenAI‑compatible protocol**.

## The lifecycle documentation

The `docs/` folder is the heart of this deliverable. Read it in order — each step states
**what** is implemented, **how**, **why**, and **which optimizations** it adds for
tokenization, latency, cost, and quality.

> **New here? Start with the [Step‑by‑Step Implementation Guide](docs/IMPLEMENTATION.md)** — a
> sequenced build order (27 steps) anchored to a concrete business use case, where every step
> explains **why we chose each approach over the alternatives**. The numbered `00`–`14` docs below
> are the deep‑dive internals it references.

| # | Document | Architecture area |
|---|----------|-------------------|
| 00 | [Overview & Architecture](docs/00-overview-and-architecture.md) | Whole system, data flow, tech choices |
| 01 | [Environment & Infrastructure](docs/01-environment-and-infrastructure.md) | Local runtime, Docker Compose, model serving |
| 02 | [Configuration & Model Providers](docs/02-configuration-and-model-providers.md) | Provider abstraction, prompt configs |
| 03 | [Document Ingestion & Content Enrichment](docs/03-document-ingestion-and-enrichment.md) | Content Extraction (GDoc API), LLM enrichment, custom metadata |
| 04 | [Chunking & Embedding](docs/04-chunking-and-embedding.md) | Table‑aware chunking, embedding generation |
| 05 | [Storage — Vector & Feature Store](docs/05-storage-vector-and-feature-store.md) | Offline store, Qdrant, Feast |
| 06 | [Query Pre‑Processing](docs/06-query-preprocessing.md) | Query optimizer, source identifier, metadata filters |
| 07 | [Hybrid Retrieval](docs/07-hybrid-retrieval.md) | Vector search, BM25, fusion, reranking |
| 08 | [Answer Generation & Post‑Processing](docs/08-answer-generation-and-postprocessing.md) | Large LLM generator, small LLM post‑processor |
| 09 | [Guardrails & Safety](docs/09-guardrails-and-safety.md) | Input/output guardrails, PII, injection, grounding |
| 10 | [Slack Interface & Serving API](docs/10-slack-interface-and-serving.md) | Slack Bolt, FastAPI, streaming |
| 11 | [Evaluation & Golden Datasets](docs/11-evaluation-and-golden-datasets.md) | Pre‑prod & post‑prod metrics, LLM‑as‑judge |
| 12 | [Observability & Monitoring](docs/12-observability-and-monitoring.md) | Tracing, metrics, logs, cost accounting |
| 13 | [Deployment, Scaling & Operations](docs/13-deployment-scaling-operations.md) | Runbooks, scaling, security, DR |
| 14 | [Optimization Playbook](docs/14-optimization-playbook.md) | Consolidated latency / cost / token / quality wins |
| 15 | [RAG Architectures](docs/15-rag-architectures.md) | The 12 selectable strategies (verification + catalog) |

## Reference technology stack (all local‑first)

| Concern | Local default | Hosted / scale‑out option |
|---|---|---|
| LLM serving | **Ollama** (dev) / **vLLM** (prod) | OpenAI, Anthropic, Azure OpenAI, Bedrock, Together |
| Large LLM (generation/enrichment) | Qwen2.5‑32B‑Instruct / Llama‑3.1‑70B | GPT‑4o class |
| Small LLM (pre/post‑process) | Qwen2.5‑7B / Llama‑3.1‑8B | GPT‑4o‑mini class |
| Embeddings | **BGE‑M3** (dense + sparse) | text‑embedding‑3‑large |
| Reranker | **BGE‑reranker‑v2‑m3** (cross‑encoder) | Cohere Rerank |
| Vector store | **Qdrant** | Qdrant Cloud / pgvector |
| Lexical / BM25 | Qdrant sparse vectors / OpenSearch | OpenSearch cluster |
| Feature store | **Feast** (+ Postgres) | Feast on managed Postgres/Redis |
| Orchestration | FastAPI + LangGraph agent loop | same, horizontally scaled |
| Task queue | Redis + arq/Celery | managed Redis |
| Guardrails | Guardrails‑AI / NeMo + Presidio | same |
| Evaluation | Ragas + promptfoo + custom judges | same, on CI |
| LLM tracing | **Langfuse** (self‑hosted) | Langfuse Cloud |
| Metrics/dashboards | Prometheus + Grafana | managed Prometheus |
| Chat surface | Slack Bolt (Socket Mode) | Slack + public gateway |

## Implementation status

The system is implemented as a runnable Python package (`src/ragnarok/`) built in 39 steps that map
1:1 to [docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md). Every step is committed individually with
tests. The whole pipeline runs **fully offline** using deterministic local backends (no GPU/LLM
required) for development and CI, and switches to real models/stores by environment variables.

Runtime cost/latency/token use is **actively managed per query** (Step 28): adaptive model routing
(simple queries → small model), dynamic per‑query token budgets, and a semantic response cache for
paraphrases — on top of the caching, streaming, reranking, and quantization the earlier steps add.
See [docs/14 — Optimization Playbook](docs/14-optimization-playbook.md).

**12 RAG architectures are implemented as pluggable, eval-comparable strategies** (Steps 29–39) —
the 8 well-known patterns (Naive, Hybrid, HyDE, Corrective/CRAG, Graph, Hybrid vector+graph,
Multimodal, Adaptive, Agentic) plus three the usual diagrams omit (RAG-Fusion, Self-RAG, RAPTOR).
Select with `rag.strategy` or let the Adaptive router choose per query. Genuineness verification and
the full catalog are in [docs/15 — RAG Architectures](docs/15-rag-architectures.md).

```bash
ragnarok ask "…"                       # uses rag.strategy (default: hybrid)
# or per call via the API / answer(strategy="hyde"|"graph"|"agentic"|…)
```

```bash
pip install -e ".[dev]"
pytest -q                        # ~100 tests, all green, no external services
python -m ragnarok.eval.ci_gate  # deterministic retrieval + guardrail gate
ragnarok doctor                  # health-check dependencies
```

Backends are selected by env (local defaults; prod values in `docker/app.Dockerfile`):
`RAGNAROK_VECTOR_STORE` (memory|qdrant), `RAGNAROK_FEATURE_STORE` (memory|feast),
`RAGNAROK_EMBED_BACKEND`/`RAGNAROK_RERANK_BACKEND` (local|http), `RAGNAROK_CACHE` (memory|redis),
and each model role's `base_url` (Ollama/vLLM/OpenAI/…).

## Repository layout

```
RAGnarok/
├── README.md
├── docs/                      # ← the full lifecycle documentation (this deliverable)
├── config/
│   ├── settings.example.yaml  # provider + runtime configuration
│   └── prompts/               # versioned prompt configs (see docs/02)
├── docker/                    # compose files for the local stack
├── src/ragnarok/
│   ├── ingestion/             # docs/03, docs/04
│   ├── stores/                # docs/05
│   ├── retrieval/             # docs/06, docs/07
│   ├── generation/            # docs/08
│   ├── guardrails/            # docs/09
│   ├── serving/               # docs/10  (FastAPI + Slack)
│   ├── eval/                  # docs/11
│   ├── observability/         # docs/12
│   ├── optimization/          # Step 28 (adaptive routing, budgets, semantic cache)
│   ├── strategies/            # Steps 29-39 (12 RAG architectures, docs/15)
│   └── agent/                 # Step 39 (Agentic RAG: tools, memory, ReAct)
├── datasets/golden/           # golden test data (docs/11)
└── tests/
```

## Quickstart (local)

```bash
# 1. Bring up the local stack (Qdrant, Postgres, Redis, Langfuse, Prometheus, Grafana)
docker compose -f docker/compose.core.yaml up -d

# 2. Serve local models
ollama pull qwen2.5:32b-instruct        # large
ollama pull qwen2.5:7b-instruct         # small
ollama pull bge-m3                      # embeddings

# 3. Configure (copy and edit)
cp config/settings.example.yaml config/settings.yaml

# 4. Ingest a corpus, then ask a question
ragnarok ingest ./datasets/raw
ragnarok ask "What is our refund policy for enterprise customers?"

# 5. Run the golden‑set evaluation
ragnarok eval --suite golden
```

See [docs/01](docs/01-environment-and-infrastructure.md) for the full setup.

---

_This repository is documentation‑first: the `docs/` set specifies the complete, buildable system.
Code modules are described with runnable reference snippets in each document so the implementation
can be assembled step by step._
