# 02 — Configuration & Model Providers

**What:** the abstraction layer that lets one codebase address models by **role**, run **fully
local**, and swap in **OpenAI / any API** per role without edits. Also the home of the diagram's
**"Prompt configs"** box — versioned, testable prompts decoupled from code.

---

## 1. The provider abstraction (the core idea)

Application code never imports a vendor SDK. It asks for a **role** — `llm_large`, `llm_small`,
`embedding`, `reranker` — and gets back a client that speaks the **OpenAI‑compatible** protocol.
Because Ollama, vLLM, LM Studio, OpenAI, Azure, Together and Anyscale all expose that protocol,
"local vs. hosted" becomes a base‑URL string.

```python
# src/ragnarok/providers.py
from openai import OpenAI, AsyncOpenAI
from functools import lru_cache
from .config import settings

class ModelRole:
    def __init__(self, cfg):
        self.model = cfg.model
        self.temperature = cfg.temperature
        self._client = AsyncOpenAI(base_url=cfg.base_url, api_key=cfg.api_key or "sk-local",
                                   timeout=cfg.timeout_s, max_retries=0)  # retries handled by us

    async def complete(self, messages, *, response_schema=None, **kw):
        extra = {}
        if response_schema:                        # structured output (see §4)
            extra["extra_body"] = {"guided_json": response_schema}  # vLLM/Ollama
        return await self._client.chat.completions.create(
            model=self.model, messages=messages,
            temperature=kw.pop("temperature", self.temperature), **extra, **kw)

@lru_cache
def role(name: str) -> ModelRole:
    return ModelRole(getattr(settings.models, name))
```

Usage anywhere in the codebase:

```python
from ragnarok.providers import role
resp = await role("llm_small").complete(messages, response_schema=QueryPlan.model_json_schema())
```

**Why one protocol:** it collapses N vendor integrations into one, makes the "scalable to OpenAI or
other model APIs" requirement a config change, and means our tracing/retry/caching wrappers
(`docs/12`, `docs/14`) are written once.

---

## 2. `settings.yaml` — the single source of truth

```yaml
# config/settings.example.yaml
env: local

models:
  llm_large:                # generation + ingestion enrichment (docs/03, docs/08)
    base_url: ${LLM_LARGE_BASE_URL}
    model: qwen2.5:32b-instruct
    temperature: 0.2
    max_tokens: 1024
    timeout_s: 120
  llm_small:                # query optimize, source id, post-process (docs/06, docs/08)
    base_url: ${LLM_SMALL_BASE_URL}
    model: qwen2.5:7b-instruct
    temperature: 0.0        # deterministic for schema-bound tasks
    max_tokens: 512
    timeout_s: 30
  embedding:
    base_url: ${EMBEDDING_BASE_URL}
    model: bge-m3
    dim: 1024
  reranker:
    url: ${RERANKER_URL}
    model: bge-reranker-v2-m3

retrieval:                  # docs/07
  top_k_dense: 40
  top_k_sparse: 40
  fusion: rrf               # reciprocal rank fusion
  rerank_top_n: 8
  min_rerank_score: 0.15

generation:                 # docs/08
  max_context_chunks: 8
  cite_sources: true

guardrails:                 # docs/09
  input: { pii: block, injection: sanitize, max_query_tokens: 512 }
  output: { grounding_min: 0.6, pii: redact, toxicity: block }

eval:                       # docs/11
  golden_suite: datasets/golden/v1
  gate_thresholds: { faithfulness: 0.85, answer_relevancy: 0.80, context_recall: 0.75 }

observability:              # docs/12
  langfuse: { host: ${LANGFUSE_HOST}, sample_rate: 1.0 }
```

Loaded with Pydantic Settings so every value is typed, env‑interpolated, and validated at startup
(fail fast on a bad config, never mid‑request):

```python
# src/ragnarok/config.py
from pydantic_settings import BaseSettings
class Settings(BaseSettings):
    env: str
    models: Models
    retrieval: RetrievalCfg
    generation: GenerationCfg
    guardrails: GuardrailCfg
    eval: EvalCfg
    observability: ObsCfg
    class Config: env_nested_delimiter = "__"
settings = Settings()   # raises on startup if anything is missing/mistyped
```

**To scale a role to OpenAI:** set that role's `base_url: https://api.openai.com/v1`,
`model: gpt-4o-mini`, and put the key in `.env`. Nothing else changes.

---

## 3. Prompt configs (the diagram's "Prompt configs" box)

Prompts are **config, not code**: versioned YAML files, hot‑reloadable, individually testable, and
tracked in evaluation so a prompt change is a first‑class, reversible release.

```yaml
# config/prompts/query_optimizer.v3.yaml
id: query_optimizer
version: 3
model_role: llm_small
description: Rewrite/expand/decompose a user query for retrieval.
input_schema: { query: str, chat_history: list }
output_schema_ref: schemas/query_plan.json     # enforced via guided decoding
system: |
  You rewrite user questions to maximize retrieval quality. Output ONLY JSON matching the schema.
  - Resolve pronouns/ellipsis using chat_history.
  - Produce 1–3 focused sub-queries; expand key acronyms; keep proper nouns verbatim.
  - Do NOT answer the question.
few_shot:
  - input:  { query: "and for enterprise?", chat_history: ["What's the refund window?"] }
    output: { intent: "policy_lookup", sub_queries: ["enterprise refund window policy"], expansions: ["SLA","contract"] }
```

Loader + registry:

```python
# src/ragnarok/prompts.py
class PromptRegistry:
    def get(self, id: str, version: str | int = "latest") -> Prompt: ...
    def render(self, id, version, **vars) -> list[dict]:   # returns chat messages
        ...
```

**Why prompts as versioned config:**
- **Reproducibility** — every trace records `prompt_id@version`; you can replay any past answer.
- **Safe iteration** — promote `v4` in staging, gate it on the golden set ([docs/11](11-evaluation-and-golden-datasets.md)),
  roll back by pointing config at `v3`. No redeploy.
- **A/B testing** — route x% of traffic to a prompt version and compare post‑prod metrics.
- **Separation of duties** — a prompt engineer edits YAML; no code review of Python required.

---

## 4. Structured output (constrained decoding) — quality *and* token savings

Every agent (query optimizer, source identifier, post‑processor, LLM judges) emits **validated
JSON** via the model server's guided‑decoding feature (`guided_json` in vLLM, `format` in Ollama),
backed by a Pydantic schema.

```python
class QueryPlan(BaseModel):
    intent: Literal["policy_lookup","howto","factoid","comparison","other"]
    sub_queries: list[str] = Field(max_length=3)
    expansions: list[str] = Field(default_factory=list, max_length=8)
    needs_retrieval: bool = True
```

**Why this matters (optimizations):**
- **Latency & cost** — constrained decoding + a tight schema means the model can't ramble; output
  tokens drop 30–70% vs. free‑form "return JSON" prompting, and there are **zero reparse retries**.
- **Reliability** — invalid JSON is *impossible*, not just unlikely; removes an entire class of
  production failures and defensive parsing code.
- **Traceability** — typed objects flow through the pipeline and into traces cleanly.

---

## 5. Retry, timeout & fallback policy

Centralised so every model call inherits it (never scattered `try/except`):

```python
# src/ragnarok/resilience.py
async def call(role_name, *args, fallback_role=None, **kw):
    for attempt in range(3):                       # exp backoff: 0.5s, 1s, 2s
        try:
            return await role(role_name).complete(*args, **kw)
        except (Timeout, APIError) as e:
            await asyncio.sleep(0.5 * 2**attempt); last = e
    if fallback_role:                              # e.g. llm_large → llm_small, labelled "degraded"
        return await role(fallback_role).complete(*args, **kw)
    raise last
```

**Fallback ladder:** `llm_large` → smaller local model → hosted API (if configured) → cached/"I
can't answer confidently right now" message. Each hop is labelled in the trace so degraded answers
are visible in post‑prod metrics.

---

## 6. Optimizations introduced here

- **Role‑based provider abstraction** → local⇄hosted is config; no vendor lock‑in; one place to add
  caching/tracing/retries.
- **Guided JSON decoding** → fewer output tokens, zero parse failures, lower latency and cost.
- **Deterministic small‑model tasks (`temperature=0`)** → cacheable, reproducible, gate‑able.
- **Prompt versioning** → reversible, testable prompt releases with no redeploy.
- **Fail‑fast typed config** → misconfiguration is a startup error, never a 3 a.m. incident.

## 7. What "done" looks like

- One line switches any role local↔hosted with no code change.
- Every agent call goes through `role()` + guided JSON; there is no free‑form JSON parsing anywhere.
- Prompt versions are selectable per environment and recorded in every trace.

Next: [docs/03 — Document Ingestion & Content Enrichment](03-document-ingestion-and-enrichment.md).
