# Embedding + reranker microservice image (Step 2).
FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml ./
COPY src ./src

# Install the package plus serving + the real model backend (FlagEmbedding).
# For a CPU/dev image, set RAGNAROK_MODELS_BACKEND=local and skip FlagEmbedding.
RUN pip install --no-cache-dir -e ".[serving]" \
    && pip install --no-cache-dir "FlagEmbedding>=1.2" || true

ENV RAGNAROK_MODELS_BACKEND=flag
EXPOSE 7997
CMD ["uvicorn", "ragnarok.serving.model_services:app", "--host", "0.0.0.0", "--port", "7997"]
