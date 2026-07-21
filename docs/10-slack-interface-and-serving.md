# 10 — Slack Interface & Serving

**What:** the diagram's **Slack Interface** and the serving layer behind it — the FastAPI service
that runs the online agentic graph, streams answers, collects feedback, and exposes the same
pipeline over an HTTP/API surface for other clients.

Diagram mapping: `User ⇄ Query/Answer ⇄ Slack Interface ⇄ (Pre‑Process / Retrieval / Answer
Generation)`.

---

## 1. Why Slack, and why Socket Mode

Slack is where the users already are, and **Socket Mode** needs **no public inbound endpoint** — the
bot opens an outbound WebSocket to Slack. That's exactly right for a *fully local / private*
deployment behind a firewall: no reverse proxy, no exposed port, no TLS termination to manage.

```python
# src/ragnarok/serving/slack_app.py
from slack_bolt.async_app import AsyncApp
app = AsyncApp(token=env.SLACK_BOT_TOKEN)

@app.event("app_mention")
@app.command("/ask")
async def handle(ack, body, say, client):
    await ack()
    user = resolve_user(body)                       # → entitlements via SSO group map (docs/09)
    q, thread = extract_query(body)
    placeholder = await say(":mag: searching…", thread_ts=thread)   # instant ack
    async for update in run_pipeline_streaming(q, user, thread):    # §3
        await client.chat_update(channel=..., ts=placeholder["ts"], text=render(update))
    await add_feedback_buttons(client, placeholder)                 # §4
```

### UX affordances

- **Instant acknowledgement** (":mag: searching…") so the user knows it's working before the first
  token — perceived latency matters as much as real latency.
- **Streaming via `chat_update`** — the message fills in as tokens arrive ([docs/08](08-answer-generation-and-postprocessing.md)).
- **Threaded context** — replies stay in‑thread; thread history feeds the query optimizer's
  coreference resolution ([docs/06](06-query-preprocessing.md)).
- **Citations rendered** as source links; **"sources" expander** shows which chunks were used.
- **Feedback buttons** (👍/👎 + "wrong"/"incomplete"/"not grounded") — the post‑prod eval signal
  ([docs/11](11-evaluation-and-golden-datasets.md)).

---

## 2. The serving API (FastAPI)

Slack is one client; the same pipeline is exposed as a clean API so a web UI, CLI, or another
service can use it identically.

```python
# src/ragnarok/serving/api.py
@router.post("/v1/ask")
async def ask(req: AskRequest, user: User = Depends(auth)) -> StreamingResponse:
    return StreamingResponse(run_pipeline_streaming(req.query, user, req.thread),
                             media_type="text/event-stream")   # SSE streaming

@router.post("/v1/feedback")   # 👍/👎, corrections → Postgres → post-prod eval
@router.get("/healthz")        # liveness
@router.get("/readyz")         # dependency readiness (ragnarok doctor, docs/01)
@router.get("/metrics")        # Prometheus (docs/12)
```

Endpoints are auth'd (SSO/JWT → entitlements), rate‑limited ([docs/09](09-guardrails-and-safety.md)), and every request carries
a `trace_id` propagated to Langfuse ([docs/12](12-observability-and-monitoring.md)).

---

## 3. The online pipeline as a streaming graph

The handler runs the LangGraph state machine from [docs/00](00-overview-and-architecture.md), yielding UI updates at each phase:

```python
async def run_pipeline_streaming(q, user, thread):
    with trace("ask", user=user) as t:
        gv = await check_input(q, user)                       # docs/09
        if gv.blocked: yield final(gv.message); return
        pre = await preprocess(gv.sanitized, history(thread), user)   # docs/06
        if pre.skip_retrieval:                                 # chitchat gate
            yield final(await direct_answer(pre)); return
        chunks = await retrieve(pre.plan, pre.source, user)    # docs/07
        yield status(f"reading {len(chunks)} sources…")
        buf = ""
        async for delta in generate(pre.plan, build_context(chunks)):  # docs/08 (streamed)
            buf += delta; yield partial(buf)
        post = await postprocess(buf, chunks)                  # docs/08
        ov = await check_output(post.answer, post.claims, chunks, user)  # docs/09
        yield final(ov.answer, citations=post.citations, followups=post.followups)
```

**Backpressure & concurrency:** requests are handled async; heavy generation is bounded by a
semaphore sized to the model server's capacity, with a bounded queue and a "busy, queued…" status so
the box degrades gracefully instead of thrashing under load.

---

## 4. Feedback loop (closing the diagram's evaluation arrow)

Every answer is logged with its `trace_id`, retrieved chunks, and served text. Feedback (buttons or
`/feedback`) attaches to that trace and flows to:
- **Post‑prod metrics** ([docs/11](11-evaluation-and-golden-datasets.md)) — thumbs‑down rate, "not grounded" rate by topic/source.
- **Golden‑set growth** — corrected/hard questions become new golden cases.
- **Feature store** — per‑document feedback score feeds reranking ([docs/05](05-storage-vector-and-feature-store.md)/[docs/07](07-hybrid-retrieval.md)).

---

## 5. Optimizations introduced here

| Optimization | Wins |
|---|---|
| **Socket Mode** | no public endpoint → fits fully‑local/firewalled deployment |
| **Instant ack + streaming `chat_update`** | perceived latency collapses to time‑to‑first‑token |
| **Thread history → coreference** | better retrieval on follow‑ups without user effort |
| **Shared pipeline behind Slack + API** | one implementation, many surfaces; no drift |
| **Semaphore + bounded queue** | graceful degradation under load, no meltdown |
| **Feedback captured inline** | continuous, cheap post‑prod signal |

## 6. Failure modes & guards

- **Slack rate limits** → coalesce `chat_update`s (e.g. every ~300 ms of tokens), not per token.
- **Long answers** → post over a threshold as a threaded snippet/file.
- **Pipeline error mid‑stream** → replace placeholder with a graceful error + trace id for support.
- **Duplicate events** (Slack retries) → idempotency key on `event_id`.

## 7. What "done" looks like

- Asking in Slack returns a streamed, cited answer with feedback buttons.
- The same query via `/v1/ask` streams identically.
- Follow‑ups in a thread resolve context correctly.
- Every answer is traceable end‑to‑end by `trace_id`.

Next: [docs/11 — Evaluation & Golden Datasets](11-evaluation-and-golden-datasets.md).
