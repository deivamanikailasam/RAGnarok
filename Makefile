# RAGnarok — common local tasks. See docs/SETUP.md for the full guide.
.DEFAULT_GOAL := help
PY ?= python

.PHONY: help install install-all test lint typecheck gate demo doctor \
        up down models serve ingest ask clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	 awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Install the package (core deps only) in editable mode
	$(PY) -m pip install -e ".[dev]"

install-all: ## Install with all optional integrations (llm, stores, serving, ...)
	$(PY) -m pip install -e ".[all]"

test: ## Run the test suite (no external services needed)
	$(PY) -m pytest -q

lint: ## Lint with ruff
	ruff check src tests

typecheck: ## Type-check with mypy (advisory)
	mypy src || true

gate: ## Run the deterministic, model-free golden-set gate
	RAGNAROK_EMBED_BACKEND=local RAGNAROK_RERANK_BACKEND=local $(PY) -m ragnarok.eval.ci_gate

demo: ## Run the zero-dependency offline retrieval demo (no LLM/GPU/services)
	$(PY) scripts/demo.py

doctor: ## Health-check every dependency endpoint
	ragnarok doctor

up: ## Start the core local stack (Qdrant, Postgres, Redis, Langfuse, Prometheus, Grafana)
	docker compose -f docker/compose.core.yaml up -d

down: ## Stop the core local stack
	docker compose -f docker/compose.core.yaml down

models: ## Pull + serve local models via Ollama (dev)
	bash scripts/serve_models.sh

serve: ## Run the FastAPI + Slack service
	ragnarok serve

ingest: ## Ingest the sample corpus (requires a running LLM for enrichment)
	ragnarok ingest datasets/sample

ask: ## Ask a question (requires a running LLM); use: make ask Q="your question"
	ragnarok ask "$(Q)"

clean: ## Remove caches and local artifacts
	rm -rf .ragnarok .ruff_cache .pytest_cache .mypy_cache **/__pycache__ src/*.egg-info
