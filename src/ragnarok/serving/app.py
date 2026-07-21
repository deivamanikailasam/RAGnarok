"""FastAPI serving surface (Step 23).

Exposes the same online pipeline behind /v1/ask (SSE streaming), /v1/feedback (post-prod signal),
and health/metrics endpoints. Slack is one client of this pipeline; the API is another — one
implementation, no drift. FastAPI is lazy-imported so the package loads without it.
"""

from __future__ import annotations

import json
from typing import Any


def build_app() -> Any:  # pragma: no cover - exercised via integration
    from fastapi import Depends, FastAPI
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel

    from ragnarok.observability.metrics import metrics_response
    from ragnarok.pipeline import answer
    from ragnarok.stores.factory import get_feature_store, get_vector_store
    from ragnarok.user import User

    app = FastAPI(title="RAGnarok")

    class AskRequest(BaseModel):
        query: str
        history: list[str] = []
        thread: str | None = None

    class FeedbackRequest(BaseModel):
        trace_id: str
        doc_ids: list[str] = []
        vote: int  # +1 / -1

    def current_user() -> User:
        # real deployments resolve entitlements from the SSO/JWT (Steps 15/22)
        return User(id="api", entitlements=["public"])

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz() -> dict[str, Any]:
        import os

        from ragnarok.cli import DEFAULT_ENDPOINTS, check_endpoint

        checks = {n: check_endpoint(n, os.environ.get(e, d)).ok for n, (e, d) in DEFAULT_ENDPOINTS.items()}
        return {"ready": all(checks.values()), "checks": checks}

    @app.get("/metrics")
    def metrics() -> Any:
        return metrics_response()

    @app.post("/v1/ask")
    async def ask(req: AskRequest, user: User = Depends(current_user)) -> StreamingResponse:
        async def event_stream() -> Any:
            result = await answer(
                req.query, user, store=get_vector_store(), features=get_feature_store(),
                history=req.history,
            )
            payload = {
                "answer": result.text,
                "citations": [c.__dict__ for c in result.citations],
                "followups": result.followups,
                "grounded": result.grounded,
                "abstained": result.abstained,
                "trace_id": result.trace_id,
            }
            yield f"data: {json.dumps(payload)}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.post("/v1/feedback")
    def feedback(req: FeedbackRequest, user: User = Depends(current_user)) -> dict[str, str]:
        features = get_feature_store()
        for doc_id in req.doc_ids:
            features.record_feedback(doc_id, 1.0 if req.vote >= 0 else -1.0)
        return {"status": "recorded"}

    return app


def run(host: str = "0.0.0.0", port: int = 8000) -> None:  # pragma: no cover  # noqa: S104
    import uvicorn

    uvicorn.run(build_app(), host=host, port=port)
