# Setup & Run — Local Machine Guide

This is the practical, copy‑paste guide to install and run RAGnarok on your own machine. It has two
tiers:

- **Tier 1 — Zero‑dependency** (5 minutes): install the package and run the tests, the offline
  retrieval demo, and the evaluation gate. **No GPU, no Docker, no LLM, no network.** Great for
  verifying the install and exploring the pipeline.
- **Tier 2 — Full local stack**: bring up the data services (Docker) and local models (Ollama /
  vLLM) so you can **ingest a corpus and get real grounded, cited answers** — fully on your machine.

> New to the project? Read the [README](../README.md) and the
> [Implementation Guide](IMPLEMENTATION.md) first for what the system does and why.

---

## 0. Prerequisites

| Tool | Version | Needed for |
|---|---|---|
| **Python** | 3.10+ (3.11 recommended) | everything |
| **git** | any | cloning |
| **Docker + Docker Compose** | recent | Tier 2 data services |
| **Ollama** | latest | Tier 2 local models (dev). Optional: vLLM for prod. |
| **make** | any | convenience targets (optional) |

Check:

```bash
python --version      # 3.10+
docker --version      # Tier 2
ollama --version      # Tier 2
```

---

## Tier 1 — Zero‑dependency (no GPU / Docker / LLM)

Everything here runs offline using deterministic local backends. This is exactly what CI runs.

### 1. Clone & install

```bash
git clone https://github.com/deivamanikailasam/RAGnarok.git
cd RAGnarok

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -e ".[dev]"          # or:  make install
```

### 2. Run the tests

```bash
pytest -q                        # or:  make test
```

Expected: **all tests pass** (100+), in a couple of seconds, with no external services.

### 3. Run the offline retrieval demo

Ingests the sample corpus (deterministic embedder + model‑free heuristic enrichment) and runs hybrid
retrieval + rerank — no LLM required:

```bash
python scripts/demo.py           # or:  make demo
python scripts/demo.py "which plans support SSO?"
```

You'll see the top retrieved chunks with scores and sources, e.g.:

```
Q: What is the refund window for enterprise customers?
  [1] score=0.457  (refund policy > Enterprise Exceptions)
      Enterprise customers on a signed annual contract have a 30-day refund window ...
```

### 4. Run the evaluation gate

The deterministic, model‑free golden‑set gate (retrieval quality + guardrail refusal):

```bash
make gate
# or: RAGNAROK_EMBED_BACKEND=local RAGNAROK_RERANK_BACKEND=local python -m ragnarok.eval.ci_gate
```

Expected: `RESULT: PASS` with `context_recall=1.000`.

### 5. Health‑check (optional)

```bash
ragnarok doctor                  # or: make doctor
```

With nothing running you'll see every dependency as `FAIL (unreachable)` — that's expected in Tier 1.
It turns `ok` once Tier 2 services are up.

**That's Tier 1.** You've verified the install and the retrieval/eval pipeline with zero external
dependencies. For real question‑answering (a generated, grounded answer), continue to Tier 2.

---

## Tier 2 — Full local stack (real answers)

Adds the data services and local models so the whole online pipeline runs on your machine.

### 1. Environment variables

```bash
cp .env.example .env
# edit .env if needed (defaults point at localhost)
```

### 2. Start the core data services (Docker)

Qdrant (vectors), Postgres (registry/Feast/Langfuse), Redis (cache/queue), Langfuse (tracing),
Prometheus + Grafana (metrics):

```bash
docker compose -f docker/compose.core.yaml up -d      # or: make up
```

Verify they're healthy:

```bash
docker compose -f docker/compose.core.yaml ps
```

- Qdrant dashboard: <http://localhost:6333/dashboard>
- Langfuse: <http://localhost:3000>
- Grafana: <http://localhost:3001> (anonymous access enabled)
- Prometheus: <http://localhost:9090>

### 3. Serve local models (Ollama)

Ollama exposes an OpenAI‑compatible API at `http://localhost:11434/v1`. Pull models sized to your
machine (use a smaller "large" model on a laptop):

```bash
# laptop-friendly
ollama pull qwen2.5:7b-instruct          # use for BOTH llm_large and llm_small to start
ollama pull qwen2.5:3b-instruct          # optional, even lighter small model

# workstation with a 24GB+ GPU
ollama pull qwen2.5:32b-instruct-q4_K_M  # llm_large (quantized)
ollama pull qwen2.5:7b-instruct          # llm_small

ollama serve   # if not already running as a service
```

Embeddings + reranking: Tier 2 can use either the **deterministic local backend** (no model, crude
but works) or the **real BGE‑M3 service** (better quality). Start simple with the local backend:

```bash
export RAGNAROK_EMBED_BACKEND=local
export RAGNAROK_RERANK_BACKEND=local
```

For real embeddings/rerank, run the model service (needs `pip install -e ".[serving]"` +
`FlagEmbedding`, GPU recommended) and set the backends to `http`:

```bash
RAGNAROK_MODELS_BACKEND=flag uvicorn ragnarok.serving.model_services:app --port 7997 &
export RAGNAROK_EMBED_BACKEND=http RAGNAROK_RERANK_BACKEND=http
```

### 4. Configure

Copy the settings file and point the model roles at Ollama:

```bash
cp config/settings.example.yaml config/settings.yaml
```

In `.env` (interpolated into `settings.yaml`) set the endpoints, and in `settings.yaml` set the model
names to what you pulled:

```dotenv
# .env
LLM_LARGE_BASE_URL=http://localhost:11434/v1
LLM_SMALL_BASE_URL=http://localhost:11434/v1
QDRANT_URL=http://localhost:6333
REDIS_URL=redis://localhost:6379/0
```

```yaml
# config/settings.yaml  (models section)
models:
  llm_large: { base_url: ${LLM_LARGE_BASE_URL}, model: qwen2.5:7b-instruct }   # laptop: reuse 7b
  llm_small: { base_url: ${LLM_SMALL_BASE_URL}, model: qwen2.5:7b-instruct }
```

Use Qdrant + Redis instead of in‑memory (recommended once services are up):

```bash
export RAGNAROK_VECTOR_STORE=qdrant
export RAGNAROK_CACHE=redis
```

Confirm everything is reachable:

```bash
ragnarok doctor        # should now show ok for qdrant / redis / llm_* / langfuse ...
```

### 5. Ingest a corpus

```bash
ragnarok ingest datasets/sample           # or your own folder of .md/.txt/.html
# re-run: unchanged docs are skipped (idempotent); --full-rebuild forces re-embed
```

This runs the offline plane: extract → enrich (LLM) → table‑aware chunk → embed → store (+ feature +
graph). A first run downloads/uses the models; re‑runs are mostly cache hits.

### 6. Ask a question

```bash
ragnarok ask "What is the refund window for enterprise customers?"
```

You get a **grounded, cited** answer — or an honest "I couldn't confirm this" if the context is
insufficient (the grounding gate, by design).

Pick a RAG strategy (the 12 architectures) via `rag.strategy` in `settings.yaml`, or let the Adaptive
router choose per query:

```yaml
rag:
  strategy: hybrid        # naive|hybrid|hyde|fusion|corrective|self_rag|graph|
                          # hybrid_graph|multimodal|raptor|adaptive|agentic
```

### 7. Run the service (API + Slack)

```bash
ragnarok serve            # FastAPI on :8000  (or: make serve)
```

Endpoints:

```bash
curl localhost:8000/healthz
curl localhost:8000/readyz
curl -N -X POST localhost:8000/v1/ask \
  -H 'content-type: application/json' \
  -d '{"query":"enterprise refund window?"}'      # streams Server-Sent Events
```

Prometheus scrapes `/metrics`; traces appear in Langfuse; dashboards in Grafana.

**Slack (optional):** create a Slack app with Socket Mode, put `SLACK_BOT_TOKEN` and
`SLACK_APP_TOKEN` in `.env`, then the same `ragnarok serve` process drives the Slack bot (no public
endpoint needed). See [docs/10](10-slack-interface-and-serving.md).

### 8. Full golden‑set evaluation (with real models)

```bash
ragnarok eval --suite datasets/golden/v1
```

Runs the pipeline over the golden set and scores faithfulness / relevancy / recall; the gate blocks
regressions in CI ([docs/11](11-evaluation-and-golden-datasets.md)).

---

## Command cheat‑sheet

| Task | Command | Tier |
|---|---|---|
| Install | `make install` / `pip install -e ".[dev]"` | 1 |
| Tests | `make test` / `pytest -q` | 1 |
| Offline demo | `make demo` / `python scripts/demo.py` | 1 |
| Eval gate (model‑free) | `make gate` | 1 |
| Health check | `ragnarok doctor` | 1/2 |
| Start services | `make up` | 2 |
| Serve models | `make models` (Ollama) | 2 |
| Ingest | `ragnarok ingest PATH` | 2 |
| Ask | `ragnarok ask "…"` | 2 |
| Serve API/Slack | `ragnarok serve` | 2 |
| Stop services | `make down` | 2 |

---

## Configuration reference (env → behavior)

Backends are chosen by environment variable, never hardcoded — so the same code runs offline or
against real services:

| Variable | Values | Default | Effect |
|---|---|---|---|
| `RAGNAROK_VECTOR_STORE` | `memory` \| `qdrant` | `memory` | vector store backend |
| `RAGNAROK_FEATURE_STORE` | `memory` \| `feast` | `memory` | feature store backend |
| `RAGNAROK_EMBED_BACKEND` | `local` \| `http` | `local` | embeddings (local = no GPU) |
| `RAGNAROK_RERANK_BACKEND` | `local` \| `http` | `local` | reranker |
| `RAGNAROK_CACHE` | `memory` \| `redis` | `memory` | cache backend |
| `RAGNAROK_MODELS_BACKEND` | `local` \| `flag` | `local` | embed/rerank service backend |
| `RAGNAROK_SETTINGS` | path | `config/settings.yaml` | settings file |
| `LLM_LARGE_BASE_URL` / `LLM_SMALL_BASE_URL` | URL | localhost | model role endpoints |

To scale a model role to a hosted API (e.g. OpenAI), set that role's `base_url` to
`https://api.openai.com/v1`, its `model` to a hosted model, and put the key in `.env` — no code
change. See [docs/02](02-configuration-and-model-providers.md).

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ragnarok doctor` shows FAIL for everything | You're in Tier 1 (no services). Start them: `make up`. |
| `ragnarok ask` hangs or errors on connect | No LLM reachable. Start Ollama (`ollama serve`) and set `LLM_*_BASE_URL`. |
| `ingest` fails calling the LLM | Enrichment needs an LLM. Use Tier 2, or use the offline demo (`scripts/demo.py`) which skips LLM enrichment. |
| Out of GPU memory pulling a large model | Use a smaller model (e.g. `qwen2.5:7b-instruct`) for `llm_large`, or route it to a hosted API. |
| Qdrant/Redis connection refused | `docker compose -f docker/compose.core.yaml ps`; ensure containers are healthy; check ports 6333/6379. |
| Crude/odd retrieval quality | The `local` embed/rerank backends are deterministic stand‑ins. Switch to `http` (real BGE‑M3) for quality. |
| Port already in use | Change the host port mapping in `docker/compose.core.yaml`, or stop the conflicting process. |
| "no space left on device" | Remove build artifacts/caches (`make clean`) and unused Docker images (`docker system prune`). |

---

## What runs where (mental model)

```
Tier 1 (offline):  pytest · scripts/demo.py · ci_gate        — deterministic, no services
Tier 2 (local):    docker compose (data)  +  Ollama (models)
                   ragnarok ingest  →  offline plane (enrich/chunk/embed/store)
                   ragnarok ask     →  online plane (preprocess→retrieve→generate→ground→guard)
                   ragnarok serve   →  FastAPI + Slack, /metrics → Prometheus/Grafana, traces → Langfuse
```

Next: [docs/01 — Environment & Infrastructure](01-environment-and-infrastructure.md) for the
infrastructure details, or [docs/13 — Deployment, Scaling & Operations](13-deployment-scaling-operations.md)
to go from this local setup to a cluster.
