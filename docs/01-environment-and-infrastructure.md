# 01 — Environment & Infrastructure

**What:** the local runtime that hosts every stateful dependency and every model, so the whole
system runs on a single workstation / server with no external calls — and scales out by changing
endpoints, not code.

---

## 1. Hardware baseline & sizing

| Tier | Machine | What runs locally | Notes |
|---|---|---|---|
| **Dev / laptop** | 32 GB RAM, Apple M‑series or 8 GB GPU | Ollama with 7B small + 3B/7B for enrichment, Qdrant, Postgres, Redis | Use a smaller "large" model (e.g. Qwen2.5‑14B) or route `llm_large` to a hosted API. |
| **Single‑GPU workstation** | 64–128 GB RAM, 1× 24–48 GB GPU (RTX 4090 / A6000) | vLLM serving a 7B small + quantized 32B large, BGE‑M3, BGE‑reranker, full stack | Comfortable for a team; ~10–30 concurrent users. |
| **Server / multi‑GPU** | 2–8× A100/H100 | vLLM tensor‑parallel 70B large, dedicated embedding + rerank servers | Org‑wide; hundreds of concurrent users. |

**Rule of thumb (GPU VRAM):** a model needs roughly `params × bytes_per_param` GB. At 4‑bit
(AWQ/GPTQ) that's ~0.5 GB/B → a 32B model ≈ 18–20 GB weights + KV cache. Keep small + embedding +
reranker resident; time‑share or second‑GPU the large model.

---

## 2. The local stack as Docker Compose

Everything stateful ships as containers so the environment is reproducible and disposable.
`docker/compose.core.yaml`:

```yaml
services:
  qdrant:                       # vector + sparse store  (docs/05, docs/07)
    image: qdrant/qdrant:latest
    ports: ["6333:6333", "6334:6334"]
    volumes: ["qdrant_data:/qdrant/storage"]

  postgres:                     # Feast registry/offline store, app metadata, Langfuse (docs/05, docs/12)
    image: postgres:16
    environment:
      POSTGRES_PASSWORD: ${PG_PASSWORD}
      POSTGRES_DB: ragnarok
    ports: ["5432:5432"]
    volumes: ["pg_data:/var/lib/postgresql/data"]

  redis:                        # task queue + response/embedding cache (docs/03, docs/06, docs/14)
    image: redis:7-alpine
    ports: ["6379:6379"]
    volumes: ["redis_data:/data"]

  langfuse:                     # LLM tracing / cost (docs/12)
    image: langfuse/langfuse:latest
    depends_on: [postgres]
    environment:
      DATABASE_URL: postgresql://postgres:${PG_PASSWORD}@postgres:5432/langfuse
      NEXTAUTH_SECRET: ${LANGFUSE_SECRET}
      SALT: ${LANGFUSE_SALT}
    ports: ["3000:3000"]

  prometheus:                   # metrics scrape (docs/12)
    image: prom/prometheus:latest
    volumes: ["./docker/prometheus.yml:/etc/prometheus/prometheus.yml"]
    ports: ["9090:9090"]

  grafana:                      # dashboards (docs/12)
    image: grafana/grafana:latest
    depends_on: [prometheus]
    ports: ["3001:3000"]
    volumes: ["grafana_data:/var/lib/grafana"]

volumes: { qdrant_data: {}, pg_data: {}, redis_data: {}, grafana_data: {} }
```

Model serving is kept **out** of Compose in dev (Ollama runs natively for GPU access) and put in a
separate `compose.serving.yaml` for vLLM in prod, so the data plane and the model plane scale
independently.

---

## 3. Model serving

### 3.1 Dev: Ollama

```bash
ollama pull qwen2.5:32b-instruct-q4_K_M   # llm_large
ollama pull qwen2.5:7b-instruct           # llm_small
ollama pull bge-m3                         # embedding (dense+sparse)
# reranker runs as a small python service (sentence-transformers) — see docs/07
ollama serve                               # exposes OpenAI-compatible API at :11434/v1
```

### 3.2 Prod: vLLM (OpenAI‑compatible, continuous batching)

```bash
# large model, one endpoint per role so they scale/upgrade independently
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-32B-Instruct-AWQ \
  --quantization awq --max-model-len 16384 \
  --gpu-memory-utilization 0.90 --port 8001

python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --max-model-len 16384 --port 8002
```

**Why vLLM in prod:** continuous batching + paged‑attention KV cache give 5–20× the throughput of
Ollama under concurrency, and it supports **guided/structured decoding** (JSON‑schema constrained
output) which our agents rely on ([docs/02](02-configuration-and-model-providers.md)).

Embeddings and rerank get their own lightweight servers (Infinity or a FastAPI wrapper around
`sentence-transformers`/`FlagEmbedding`) so a slow generation never blocks a fast embed.

---

## 4. The `.env` and secrets

Local‑first does not mean secret‑free. Keep credentials (Slack tokens, Langfuse keys, any hosted
API keys used for scale‑out) in a `.env` never committed, and load via Pydantic settings.

```dotenv
# --- model endpoints (roles → OpenAI-compatible base URLs) ---
LLM_LARGE_BASE_URL=http://localhost:8001/v1
LLM_SMALL_BASE_URL=http://localhost:8002/v1
EMBEDDING_BASE_URL=http://localhost:7997        # Infinity
RERANKER_URL=http://localhost:7998/rerank
# --- infra ---
QDRANT_URL=http://localhost:6333
DATABASE_URL=postgresql://postgres:***@localhost:5432/ragnarok
REDIS_URL=redis://localhost:6379/0
# --- observability ---
LANGFUSE_HOST=http://localhost:3000
LANGFUSE_PUBLIC_KEY=pk-...
LANGFUSE_SECRET_KEY=sk-...
# --- slack (docs/10) ---
SLACK_BOT_TOKEN=xoxb-...
SLACK_APP_TOKEN=xapp-...
# --- optional scale-out (leave blank to stay fully local) ---
OPENAI_API_KEY=
```

---

## 5. Project bootstrap

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .            # installs the ragnarok package + CLI
pre-commit install          # ruff + black + mypy + secret scan
docker compose -f docker/compose.core.yaml up -d
ragnarok doctor             # health-checks every dependency (below)
```

`ragnarok doctor` pings each endpoint and prints a readiness table — the first thing to run when
something is off:

```
component      endpoint                         status   detail
qdrant         http://localhost:6333            ok       12 collections
llm_large      http://localhost:8001/v1         ok       Qwen2.5-32B-Instruct-AWQ
llm_small      http://localhost:8002/v1         ok       Qwen2.5-7B-Instruct
embedding      http://localhost:7997            ok       BAAI/bge-m3 (dim=1024)
reranker       http://localhost:7998            ok       bge-reranker-v2-m3
langfuse       http://localhost:3000            ok       traces enabled
```

---

## 6. Environments & configuration profiles

Three profiles, selected by `RAGNAROK_ENV`, differing only in `config/settings.<env>.yaml`:

- **`local`** — Ollama, single replica, verbose tracing, sampling=100%.
- **`staging`** — vLLM, golden‑set eval gate on, tracing sampling=100%, synthetic load tests.
- **`prod`** — vLLM (possibly hosted fallback), tracing sampling≈10–20% + 100% on errors, alerting on.

No code differs between environments — only endpoints, replica counts, and sampling. This is what
makes "runs fully local" and "scales to a hosted cluster" the *same* system.

---

## 7. Optimizations introduced here

- **Separate endpoint per model role** → independent scaling, upgrading, and failure isolation
  (a stuck 32B never blocks a 7B query rewrite).
- **Quantized large model (AWQ/GPTQ 4‑bit)** → ~4× smaller VRAM, ~2–3× faster decode, negligible
  quality loss for RAG generation. Frees VRAM to keep small + embed + rerank co‑resident.
- **Native Ollama in dev, containerized vLLM in prod** → GPU passthrough simplicity locally,
  reproducible throughput remotely.
- **Redis as a shared cache** from day one → embedding cache, query‑rewrite cache, and full
  response cache all live here (see [docs/14](14-optimization-playbook.md)).

## 8. What "done" looks like

- `ragnarok doctor` reports every component `ok`.
- `docker compose up -d` + model pulls reproduce the stack on a clean machine in < 15 min.
- Switching a single role to a hosted API is a one‑line `.env`/config change with no code edits.

Next: [docs/02 — Configuration & Model Providers](02-configuration-and-model-providers.md).
