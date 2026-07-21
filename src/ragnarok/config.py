"""Typed configuration (Step 3).

Single source of truth: ``config/settings.yaml`` (falling back to ``settings.example.yaml``),
with ``${VAR}`` env interpolation, validated into typed Pydantic models at load time so a
misconfiguration is a startup error, never a mid-request surprise.

Access via ``get_settings()`` (lazily loaded + cached). Tests can inject with
``set_settings(...)`` / ``reset_settings()``.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- models


class LLMRole(BaseModel):
    base_url: str = "http://localhost:11434/v1"
    model: str
    api_key: str = "sk-local"
    temperature: float = 0.2
    max_tokens: int = 1024
    timeout_s: float = 120.0


class EmbeddingRole(BaseModel):
    base_url: str = "http://localhost:7997"
    model: str = "bge-m3"
    dim: int = 1024


class RerankerRole(BaseModel):
    url: str = "http://localhost:7997"
    model: str = "bge-reranker-v2-m3"


class Models(BaseModel):
    llm_large: LLMRole
    llm_small: LLMRole
    embedding: EmbeddingRole = Field(default_factory=EmbeddingRole)
    reranker: RerankerRole = Field(default_factory=RerankerRole)


class RetrievalCfg(BaseModel):
    top_k_dense: int = 40
    top_k_sparse: int = 40
    fusion: Literal["rrf", "weighted"] = "rrf"
    rrf_k: int = 60
    rerank_top_n: int = 8
    min_rerank_score: float = 0.15
    filter_fallback_min_candidates: int = 5


class GenerationCfg(BaseModel):
    max_context_chunks: int = 8
    context_budget_tokens: int = 3500
    cite_sources: bool = True
    self_correction_max_retries: int = 1


class ChunkingCfg(BaseModel):
    target_tokens: int = 384
    overlap: float = 0.15


class InputGuardCfg(BaseModel):
    pii: Literal["block", "redact", "off"] = "block"
    injection: Literal["sanitize", "block", "off"] = "sanitize"
    max_query_tokens: int = 512
    rate_limit_per_min: int = 30


class OutputGuardCfg(BaseModel):
    grounding_min: float = 0.6
    pii: Literal["redact", "block", "off"] = "redact"
    toxicity: Literal["block", "flag", "off"] = "block"


class GuardrailCfg(BaseModel):
    input: InputGuardCfg = Field(default_factory=InputGuardCfg)
    output: OutputGuardCfg = Field(default_factory=OutputGuardCfg)


class StoresCfg(BaseModel):
    qdrant_url: str = "http://localhost:6333"
    collection_alias: str = "chunks"
    vector_quantization: Literal["int8", "binary", "none"] = "int8"
    database_url: str = "postgresql://postgres:ragnarok@localhost:5432/ragnarok"
    redis_url: str = "redis://localhost:6379/0"


class EvalCfg(BaseModel):
    golden_suite: str = "datasets/golden/v1"
    gate_thresholds: dict[str, float] = Field(
        default_factory=lambda: {
            "faithfulness": 0.85,
            "answer_relevancy": 0.80,
            "context_recall": 0.75,
        }
    )
    judge_role: str = "llm_small"


class LangfuseCfg(BaseModel):
    host: str = "http://localhost:3000"
    public_key: str = ""
    secret_key: str = ""
    sample_rate: float = 1.0


class ObsCfg(BaseModel):
    langfuse: LangfuseCfg = Field(default_factory=LangfuseCfg)


class ServingCfg(BaseModel):
    stream: bool = True
    slack_socket_mode: bool = True


class CachingCfg(BaseModel):
    response_ttl_s: int = 3600
    rewrite_ttl_s: int = 3600


class OptimizationCfg(BaseModel):
    """Runtime cost/latency/token management (Step 28)."""

    adaptive_routing: bool = True  # route simple queries to the small generation model
    semantic_cache: bool = True  # serve paraphrased repeats from an embedding-similarity cache
    semantic_cache_threshold: float = 0.90  # cosine >= this counts as a hit
    semantic_cache_max_entries: int = 2000
    simple_intents: list[str] = Field(
        default_factory=lambda: ["factoid", "policy_lookup", "faq"]
    )
    simple_generation_role: str = "llm_small"
    complex_generation_role: str = "llm_large"
    # dynamic per-query budgets (simple queries get a tighter budget -> fewer tokens, lower latency)
    simple_rerank_top_n: int = 4
    complex_rerank_top_n: int = 8
    simple_context_tokens: int = 1500
    complex_context_tokens: int = 3500
    simple_max_output_tokens: int = 512
    complex_max_output_tokens: int = 1024
    max_cost_per_query_usd: float = 0.0  # 0 = disabled; > 0 forces the cheap path when exceeded


class RagCfg(BaseModel):
    """Selectable RAG architecture / strategy (Steps 29-39; catalog in docs/15)."""

    strategy: str = "hybrid"  # naive|hybrid|hyde|fusion|corrective|self_rag|graph|
    #                           hybrid_graph|multimodal|raptor|adaptive|agentic
    fusion_num_queries: int = 4
    corrective_grade_min: float = 0.30
    self_rag_relevance_min: float = 0.30
    graph_expand_hops: int = 1
    raptor_levels: int = 2
    adaptive_multistep_intents: list[str] = Field(
        default_factory=lambda: ["comparison", "multi_hop", "howto"]
    )
    agentic_max_steps: int = 4


class Settings(BaseModel):
    env: str = "local"
    models: Models
    rag: RagCfg = Field(default_factory=RagCfg)
    retrieval: RetrievalCfg = Field(default_factory=RetrievalCfg)
    generation: GenerationCfg = Field(default_factory=GenerationCfg)
    chunking: ChunkingCfg = Field(default_factory=ChunkingCfg)
    guardrails: GuardrailCfg = Field(default_factory=GuardrailCfg)
    stores: StoresCfg = Field(default_factory=StoresCfg)
    eval: EvalCfg = Field(default_factory=EvalCfg)
    observability: ObsCfg = Field(default_factory=ObsCfg)
    serving: ServingCfg = Field(default_factory=ServingCfg)
    caching: CachingCfg = Field(default_factory=CachingCfg)
    optimization: OptimizationCfg = Field(default_factory=OptimizationCfg)


# --------------------------------------------------------------------------- loading


def _expand_env(obj: object) -> object:
    """Recursively expand ${VAR} using the process environment."""
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    if isinstance(obj, dict):
        return {k: _expand_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env(v) for v in obj]
    return obj


def _settings_path() -> Path:
    explicit = os.environ.get("RAGNAROK_SETTINGS")
    if explicit:
        return Path(explicit)
    root = Path(__file__).resolve().parents[2]
    real = root / "config" / "settings.yaml"
    return real if real.exists() else root / "config" / "settings.example.yaml"


def load_settings(path: str | Path | None = None) -> Settings:
    p = Path(path) if path else _settings_path()
    raw = yaml.safe_load(p.read_text()) or {}
    return Settings.model_validate(_expand_env(raw))


_override: Settings | None = None


@lru_cache
def _cached_settings() -> Settings:
    return load_settings()


def get_settings() -> Settings:
    return _override if _override is not None else _cached_settings()


def set_settings(settings: Settings) -> None:
    """Override settings (tests)."""
    global _override
    _override = settings


def reset_settings() -> None:
    global _override
    _override = None
    _cached_settings.cache_clear()
