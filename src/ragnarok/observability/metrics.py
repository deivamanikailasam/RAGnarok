"""Prometheus metrics (Step 23 stub; expanded in Step 26).

Provides a minimal registry so the serving /metrics endpoint works. Step 26 adds the full RAG SLIs
(TTFT, tokens, cost, grounding, abstain rate). prometheus_client is lazy-imported.
"""

from __future__ import annotations

from typing import Any


def metrics_response() -> Any:  # pragma: no cover - requires prometheus_client + fastapi
    from fastapi import Response
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
