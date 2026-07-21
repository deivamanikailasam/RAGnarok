# 13 — Deployment, Scaling & Operations

**What:** how RAGnarok is packaged, released, scaled, secured, and operated — turning the local
stack into something a team can run reliably, and scale from one workstation to a cluster without
rewriting anything.

---

## 1. Packaging & release

- **Containers** for the app (API + Slack worker + ingestion workers), plus the Compose stack from
  [docs/01](01-environment-and-infrastructure.md). One image, different entrypoints (`serve`, `slack`, `ingest-worker`, `eval`).
- **CI pipeline**: lint/type/test → build image → **run golden‑set eval gate** ([docs/11](11-evaluation-and-golden-datasets.md)) →
  push image. A regression fails the pipeline; nothing ships unproven.
- **CD**: staging → smoke + eval → prod. **Blue/green** for the app and **alias‑swap** for the
  vector index ([docs/05](05-storage-vector-and-feature-store.md)) so releases and reindexes are zero‑downtime with instant rollback.
- **Versioned everything**: image tag, prompt versions, embedding/enrichment versions, index alias,
  golden‑set version — all recorded so any prod answer is reproducible.

---

## 2. Deployment topologies (same code, different scale)

| Topology | Model serving | Data plane | Users |
|---|---|---|---|
| **Single box (local)** | Ollama, all roles | Compose (Qdrant/PG/Redis) on same host | 1–10 |
| **Workstation** | vLLM (quantized large + small) + embed/rerank servers | Compose | 10–30 |
| **Cluster** | vLLM replicas per role behind a load balancer | Qdrant cluster, managed PG/Redis | 100s |
| **Hybrid** | small/embed local; `llm_large` → hosted API for burst | local data plane | elastic |

The **hybrid** row is the payoff of the [docs/02](02-configuration-and-model-providers.md) abstraction: keep sensitive‑data stages local,
burst only the generation role to a hosted API under load — a per‑role config change, no code.

---

## 3. Scaling each component

- **Stateless app** (API/Slack) — scale horizontally behind a load balancer; sessions are stateless
  (thread history read from Slack/store, not memory).
- **Model servers** — the throughput bottleneck. Scale with vLLM **continuous batching** first
  (huge win under concurrency), then replicas per role; autoscale on queue depth/utilization
  ([docs/12](12-observability-and-monitoring.md)). Separate endpoints per role means the large model scales independently of the small.
- **Qdrant** — vertical first (it's fast); shard/replicate for very large corpora; read replicas for
  query load. Quantized vectors keep it small and cache‑friendly.
- **Ingestion workers** — scale out on the Redis queue; fully decoupled from serving so a big
  re‑ingest never affects query latency.
- **Feast online store (Redis)** — replicate for read throughput.

**Capacity planning** comes from [docs/12](12-observability-and-monitoring.md) metrics: tokens/query × QPS → required decode
throughput → GPU count; p95 TTFT target → batching/replica settings.

---

## 4. Reliability & graceful degradation

The system is designed to bend, not break (defenses defined across [docs/02](02-configuration-and-model-providers.md), [docs/07](07-hybrid-retrieval.md), [docs/08](08-answer-generation-and-postprocessing.md)):

| Failure | Degradation |
|---|---|
| Large LLM saturated/down | fallback ladder → smaller/hosted model, labelled "degraded" |
| Reranker down | RRF‑only ordering |
| Feature store down | skip business signals, serve on rerank score |
| Qdrant slow | serve from response cache; shed load with "busy" status |
| Embedding server down | serve cached embeddings; queue new ingests |
| Model server overloaded | bounded queue + backpressure, never unbounded fan‑in |

All degradations are traced and alertable so "degraded but up" is visible, not silent.

---

## 5. Operational runbooks

Concrete, tested procedures shipped with the repo:

- **Reindex / re‑embed** — build `chunks_vN+1`, eval, alias‑swap, keep old N for rollback.
- **Prompt rollout** — promote version in staging, gate on golden set, canary in prod, roll back by
  config.
- **Corpus refresh** — nightly incremental ingest (content‑hash diff, [docs/03](03-document-ingestion-and-enrichment.md)); full rebuild only on
  chunker/embedding change.
- **Document deletion / RTBF** — cascade delete across Qdrant + Feast + caches ([docs/05](05-storage-vector-and-feature-store.md)); verify.
- **Incident triage** — alert → Grafana panel → example traces → stage/model/prompt root cause
  ([docs/12](12-observability-and-monitoring.md)) → mitigate (rollback prompt/model, scale, or shed).
- **Model upgrade** — stand up new role endpoint, A/B via config + post‑prod metrics, promote.
- **DR** — restore Qdrant snapshot + Postgres backup + re‑warm caches; RTO/RPO documented.

---

## 6. Security & compliance operations

(Controls specified in [docs/09](09-guardrails-and-safety.md); here's how they're operated.)
- **Secrets** in a vault/`.env` never committed; rotated on a schedule; least‑privilege connector
  credentials.
- **Network** — fully local means no egress by default; any hosted‑role egress is explicit,
  allow‑listed, and logged.
- **AuthN/Z** — SSO → entitlements → retrieval `access_tags` filter; audited.
- **Encryption** at rest (volumes/DB) and in transit (TLS between services).
- **Audit & retention** — who‑asked‑what, sources used, guardrail verdicts; configurable retention;
  PII‑scrubbed logs.
- **Backups** — Qdrant snapshots, Postgres dumps, config/prompt/golden‑set in git.

---

## 7. Cost management (operational)

- **Dashboards** track cost/query and tokens/stage ([docs/12](12-observability-and-monitoring.md)); budget alerts fire before overruns.
- **Levers** (detailed in [docs/14](14-optimization-playbook.md)): caching, right‑sized models, context trimming,
  quantization, batching, local vs hosted per role.
- **Local‑first economics** — after fixed hardware cost, marginal query cost is ~GPU‑seconds; hosted
  is used surgically for burst/quality where it pays.

---

## 8. What "done" looks like

- One command brings up local; the same artifacts deploy to a cluster via config only.
- CI blocks regressions; releases and reindexes are zero‑downtime with rollback.
- Every failure mode degrades gracefully and visibly.
- Runbooks exist and are tested for reindex, rollout, deletion, incident, upgrade, and DR.

Next: [docs/14 — Optimization Playbook](14-optimization-playbook.md).
