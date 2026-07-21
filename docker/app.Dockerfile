# RAGnarok application image (Step 27): API + Slack + ingestion workers.
# One image, different entrypoints (serve / slack / ingest-worker / eval).
FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY config ./config

RUN pip install --no-cache-dir -e ".[llm,stores,serving,guardrails,obs,worker]"

ENV RAGNAROK_ENV=prod \
    RAGNAROK_VECTOR_STORE=qdrant \
    RAGNAROK_FEATURE_STORE=feast \
    RAGNAROK_EMBED_BACKEND=http \
    RAGNAROK_RERANK_BACKEND=http \
    RAGNAROK_CACHE=redis

EXPOSE 8000
# Default entrypoint serves the API; override CMD for slack / ingest workers.
CMD ["ragnarok", "serve", "--host", "0.0.0.0", "--port", "8000"]
