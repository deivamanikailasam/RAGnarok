# 12 — Observability & Monitoring

**What:** the tracing, metrics, logging, and cost‑accounting that make the system operable — the
prerequisite for every optimization in [docs/14](14-optimization-playbook.md) and every regression caught in
[docs/11](11-evaluation-and-golden-datasets.md). "You cannot optimize, debug, or trust what you cannot see."

---

## 1. The three pillars, plus the LLM‑specific fourth

| Pillar | Tool (local) | What it answers |
|---|---|---|
| **Traces** | Langfuse (self‑hosted) | "What happened inside *this one* request, step by step?" |
| **Metrics** | Prometheus + Grafana | "How is the system behaving in aggregate, over time?" |
| **Logs** | structured JSON → Loki/Postgres | "What exactly did component X log at time T?" |
| **LLM analytics** | Langfuse | "Tokens, cost, quality, and prompt versions per stage" |

Everything is **local‑first**; all of it can point at hosted backends by config, same as models.

---

## 2. Tracing — one span tree per request

Every request opens a root span; each stage ([docs/06](06-query-preprocessing.md)–[docs/09](09-guardrails-and-safety.md)) is a child span with inputs,
outputs, model+prompt versions, token counts, and latency. This is the single most valuable
artifact for debugging an agentic pipeline.

```python
# src/ragnarok/observability/trace.py  (OpenTelemetry + Langfuse)
@traced("retrieve")
async def retrieve(...):
    span.set(dense_k=40, sparse_k=40, filter=qfilter, rerank_top_n=8)
    ...
    span.set(returned=len(final), top_score=final[0].rerank_score)
```

A trace for one Slack question shows, nested:

```
ask (trace_id=abc, user=…, 3.9s, $0.0021, 5,240 tok)
├─ guard.input            18ms   verdict=allow
├─ preprocess             240ms
│  ├─ query_optimizer     150ms  llm_small  qopt@v3   in=180 out=64
│  └─ source_identifier   190ms  llm_small  srcid@v2  in=210 out=48
├─ retrieve               52ms   dense+sparse=80→fuse=50→rerank=8
├─ generate               3.1s   llm_large  ansgen@v5 in=3,900 out=520  (TTFT 780ms)
├─ postprocess            220ms  llm_small  post@v4   in=560 out=140
└─ guard.output           95ms   grounding=0.91 verdict=serve
```

**Why this matters:** every latency spike, token blowup, quality dip, or guardrail block is
attributable to a *specific stage, model, and prompt version* — no guessing. Langfuse also links the
trace to its evaluation scores ([docs/11](11-evaluation-and-golden-datasets.md)) and captures user feedback on the same object.

---

## 3. Metrics — the RAG SLIs

Exposed at `/metrics`, scraped by Prometheus, dashboarded in Grafana. The signals that actually
matter for a RAG service:

**Latency**
- `ttft_seconds` (time‑to‑first‑token) — the number users feel. p50/p95/p99.
- `request_duration_seconds` per stage and end‑to‑end.

**Throughput / load**
- requests/sec, concurrent generations, queue depth, model‑server utilization.

**Tokens & cost**
- `tokens_total{stage,role,direction}`; `cost_per_query` (even "local cost" via GPU‑seconds).
- prompt‑cache and response‑cache hit rates.

**Quality (online proxies)**
- grounding score distribution, abstain/"no answer" rate, thumbs‑down rate, guardrail trigger rates.

**Retrieval health**
- empty‑result rate, mean rerank top score, filter‑fallback rate.

**Dependency health**
- Qdrant/model‑server/Feast/Redis up, error rates, p95 latency per dependency.

```python
TTFT = Histogram("ttft_seconds", buckets=[.25,.5,1,2,4])
TOKENS = Counter("tokens_total", labelnames=["stage","role","direction"])
COST = Counter("query_cost_usd_total")
GROUNDING = Histogram("grounding_score", buckets=[.5,.6,.7,.8,.9,1.0])
```

---

## 4. Cost & token accounting (even when "free"/local)

Local models aren't billed per token, but they cost **GPU‑seconds, latency, and capacity** — so we
account for them as if they were:
- Token counts per stage/role are recorded on every call (from the serving response usage).
- A cost model maps `(model, tokens)` → USD for hosted roles and → GPU‑second cost for local roles,
  so "cost per query" is comparable whether a role is local or hosted. This makes the local↔hosted
  trade‑off ([docs/02](02-configuration-and-model-providers.md)) a data‑driven decision, and surfaces the token‑heavy stages that
  [docs/14](14-optimization-playbook.md) targets.

---

## 5. Logging & correlation

- **Structured JSON logs** with the `trace_id` on every line → jump from a metric spike to the trace
  to the raw logs in seconds.
- **PII‑scrubbed** before persistence ([docs/09](09-guardrails-and-safety.md)); sensitive fields hashed/redacted.
- **Log levels** tuned so prod isn't drowned; errors always carry the trace id surfaced to the user
  ([docs/10](10-slack-interface-and-serving.md)) for support.

---

## 6. Alerting & SLOs

Define SLOs and alert on burn, not on every blip:

| SLO | Target | Alert |
|---|---|---|
| Availability | 99.5% of `/v1/ask` succeed | error‑rate burn |
| TTFT | p95 < 1.5 s | sustained p95 breach |
| Faithfulness (post‑prod) | ≥ 0.85 rolling | daily drop > threshold |
| Abstain rate | < 15% | spike (coverage gap) |
| Cost/query | < budget | rolling budget breach |
| Dependency | all up | any down / p95 latency spike |

Alerts route to Slack/PagerDuty and link straight to the relevant Grafana panel + example traces.

---

## 7. Optimizations enabled here

Observability doesn't optimize by itself — it *makes optimization possible*:
- Token/cost per stage → tells you exactly where [docs/14](14-optimization-playbook.md)'s wins are (usually generator input tokens).
- TTFT breakdown → tells you whether to cache, prefetch, or shrink context.
- Cache hit rates → validate the caching layers are actually paying off.
- Grounding/abstain trends → early warning of corpus or model regressions.
- Per‑model latency → drives the fallback ladder and capacity planning.

## 8. What "done" looks like

- Every request has one nested trace with tokens, cost, model+prompt versions per stage.
- Grafana shows TTFT, cost/query, grounding, abstain, and dependency health live.
- Alerts fire on SLO burn and link to traces.
- A production incident can be root‑caused to a stage/model/prompt in minutes.

Next: [docs/13 — Deployment, Scaling & Operations](13-deployment-scaling-operations.md).
