#!/usr/bin/env bash
# Dev model serving via Ollama (Step 2). For prod GPU serving use docker/compose.serving.yaml.
#
# Ollama exposes an OpenAI-compatible API at http://localhost:11434/v1 — point the llm_* roles
# there in config/settings.yaml. The embedding+reranker service runs separately.
set -euo pipefail

echo ">> Pulling local models via Ollama..."
ollama pull qwen2.5:32b-instruct-q4_K_M   # llm_large  (quantized: ~4x less VRAM)
ollama pull qwen2.5:7b-instruct           # llm_small
ollama pull bge-m3                         # embedding (dense; sparse via model service)

echo ">> Starting Ollama (OpenAI-compatible at :11434/v1)"
ollama serve &

echo ">> Starting embedding + reranker service on :7997"
# Use the real backend if FlagEmbedding + weights are available; else deterministic local.
RAGNAROK_MODELS_BACKEND="${RAGNAROK_MODELS_BACKEND:-local}" \
  uvicorn ragnarok.serving.model_services:app --host 0.0.0.0 --port 7997 &

wait
echo ">> Models serving. Set in config/settings.yaml:"
echo "   llm_large.base_url / llm_small.base_url -> http://localhost:11434/v1"
echo "   embedding.base_url / reranker.url       -> http://localhost:7997"
