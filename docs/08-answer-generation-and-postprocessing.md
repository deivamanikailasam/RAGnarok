# 08 — Answer Generation & Post‑Processing

**What:** the diagram's **Answer Generation** box — the **large‑LLM answer generator** and the
**small‑LLM post‑processor** — plus the retrieved‑chunk context assembly that feeds them. This is
where grounded, cited answers are produced and polished.

Diagram mapping: `Retrieved chunks → LLM (large) answer generator → LLM (small) post‑processor →
Answer → Slack`.

---

## 1. Context assembly (before the generator)

The reranked top‑k chunks ([docs/07](07-hybrid-retrieval.md)) are packed into a **token‑budgeted** context with stable
citation markers. Packing is deliberate, not "concatenate everything":

```python
# src/ragnarok/generation/context.py
def build_context(chunks, budget_tokens: int):
    ctx, used = [], 0
    for i, c in enumerate(chunks, 1):
        block = f"[{i}] (source: {c.title} › {c.section}, {c.freshness_date})\n" \
              + (c.table_markdown or c.text)                 # tables handed over verbatim
        t = count_tokens(block)
        if used + t > budget_tokens: break                  # hard budget → predictable cost
        ctx.append(block); used += t
    return "\n\n".join(ctx), len(ctx)
```

**Why budgeted + numbered:** a fixed input budget makes latency and cost predictable and prevents
"context dilution" (quality *drops* when you stuff too many marginal chunks). Numbered `[i]` markers
let the model cite precisely and let us verify grounding in post‑processing.

**Ordering:** highest‑reranked chunks are placed at the **start and end** of the context ("lost in
the middle" mitigation) so the most relevant evidence sits where models attend best.

---

## 2. The answer generator (large LLM)

```python
# src/ragnarok/generation/generate.py
async def generate(plan, context, n_sources) -> AsyncIterator[str]:
    msgs = prompts.render("answer_generator", "latest",
                          question=plan.rewritten_query, context=context)
    async for delta in stream("llm_large", msgs):    # streamed to user (docs/10)
        yield delta
```

The prompt (`config/prompts/answer_generator.yaml`) enforces the non‑negotiables:

```
- Answer ONLY from the numbered context. If the context is insufficient, say you don't know
  and state what's missing. Never use outside knowledge or guess.
- Cite every claim with [i] markers referencing the context blocks used.
- Prefer exact values from tables; do not recompute or infer numbers not present.
- Be concise; match the user's language.
```

### Why these constraints

- **Grounding / anti‑hallucination** — "only from context, else abstain" is the primary defense
  against confident wrong answers, verified downstream (§4, [docs/09](09-guardrails-and-safety.md)).
- **Inline citations** — trust + auditability; also the substrate for the grounding check.
- **Table fidelity** — the generator reads exact cells from the verbatim table markdown
  ([docs/04](04-chunking-and-embedding.md)), never re‑derives numbers.

**Streaming** is essential to perceived latency: first token in <1 s, tokens flow while the full
answer (1.5–6 s locally) completes ([docs/00](00-overview-and-architecture.md) trace).

---

## 3. The post‑processor (small LLM)

A cheap second pass that the diagram draws explicitly. It runs **after** generation (on the
completed text, or on the buffered stream) and does the narrow, structured jobs a big model
shouldn't waste tokens on:

```python
class PostProcessed(BaseModel):
    answer: str                     # cleaned, formatted for the channel (Slack markdown)
    citations: list[Citation]       # [i] → {doc_id, title, uri, span}
    claims: list[GroundedClaim]     # claim → supporting [i]  (for the grounding gate)
    followups: list[str] = Field(max_length=3)
    self_reported_confidence: float
```

Responsibilities:
- **Citation resolution** — map `[i]` markers to real source URIs/titles for display and click‑through.
- **Claim extraction** — split the answer into atomic claims each tied to its supporting chunk(s);
  this feeds the grounding/faithfulness gate ([docs/09](09-guardrails-and-safety.md), [docs/11](11-evaluation-and-golden-datasets.md)).
- **Formatting** — channel‑specific (Slack mrkdwn, web HTML), length trimming, consistent tone.
- **Follow‑up suggestions** — cheap UX lift, generated from context.

**Why a small model here:** these are schema‑bound, mechanical tasks. Doing them on the large model
would cost 4–10× the tokens for no quality gain; the small model does them deterministically and in
parallel with the grounding check.

---

## 4. Grounding / faithfulness gate (inline)

Before the answer reaches the user, we verify each extracted claim is actually supported by the
cited context. Cheap claims → NLI/embedding entailment; ambiguous ones → a small‑LLM judge.

```python
async def grounding_score(claims, context) -> float:
    supported = 0
    for cl in claims:
        if entails(context[cl.cite], cl.text):          # fast NLI first
            supported += 1
        elif await llm_judge_entails(cl, context):      # escalate only if uncertain
            supported += 1
    return supported / max(len(claims), 1)
```

- Score `≥ grounding_min` (config, e.g. 0.6) → serve.
- Below → **abstain / hedge**: return "I couldn't find enough in our documents to answer confidently"
  plus the closest sources, rather than a possibly‑hallucinated answer. This is fail‑closed on
  correctness. Every verdict is traced ([docs/12](12-observability-and-monitoring.md)) and sampled into post‑prod eval.

---

## 5. The agentic loop & optional self‑correction

The whole online path is a LangGraph state machine: `guard → preprocess → retrieve → generate →
postprocess → grounding‑gate → guard‑out`. For high‑stakes intents we allow **one** bounded
corrective retry:

- If grounding is low *and* the model signalled "context insufficient", the graph can **re‑retrieve
  once** with relaxed filters or an expanded query, then regenerate. Bounded to one retry so latency
  and cost stay predictable — no open‑ended agent loops.

---

## 6. Optimizations introduced here

| Optimization | Wins |
|---|---|
| **Streaming generation** | first token <1 s; perceived latency dominated by TTFT, not total |
| **Token‑budgeted, reranked context (≤8 chunks)** | fewer input tokens → lower cost & latency, higher quality |
| **"Lost‑in‑the‑middle" ordering** | best evidence where the model attends → better answers, no extra cost |
| **Small‑LLM post‑processing** | citations/format/claims at ~1/10th the token cost of doing it on the large model |
| **Prompt‑prefix caching (KV cache)** | stable system+instructions prefix cached by vLLM → cheaper, faster decode |
| **Response cache** (normalized query + index version) | identical questions served instantly, zero LLM cost |
| **Constrained decoding on post‑processor** | reliable citations/claims, no parse retries |
| **Abstain‑on‑low‑grounding** | avoids the most expensive failure of all: a confident wrong answer |
| **Bounded single self‑correction** | recovers recoverable misses without unbounded cost |

**Tokenization detail:** the generator's context budget is set from the *serving model's* tokenizer
so we pack to the exact window; the system/instruction prefix is kept **byte‑stable** across
requests so vLLM/Ollama prefix‑KV caching applies, cutting time‑to‑first‑token on warm paths.

## 7. Failure modes & guards

- **Hallucination** → grounding gate + "only from context" prompt + abstain.
- **Prompt injection via retrieved content** ("ignore instructions and…") → context is wrapped as
  untrusted data; system prompt states retrieved text is data, not instructions; output guardrails
  re‑check ([docs/09](09-guardrails-and-safety.md)).
- **Over‑long answer / runaway** → `max_tokens` + post‑processor trimming.
- **Large model saturated** → resilience fallback ladder to smaller/hosted model, labelled degraded.

## 8. What "done" looks like

- Answers cite sources; every claim maps to a chunk; grounding score gates serving.
- Insufficient‑context questions produce an honest abstention, not a fabrication.
- First token consistently under ~1 s locally on warm paths.

Next: [docs/09 — Guardrails & Safety](09-guardrails-and-safety.md).
