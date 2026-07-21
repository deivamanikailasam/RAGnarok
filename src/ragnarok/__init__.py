"""RAGnarok — a production-grade, fully-local agentic RAG system.

The package is organized to mirror the architecture and the step-by-step build guide in
``docs/IMPLEMENTATION.md``:

- ``ragnarok.ingestion``   offline plane: connectors, enrichment, chunking, embedding (Steps 4-9)
- ``ragnarok.stores``      vector + feature stores, registry (Steps 5, 10-13)
- ``ragnarok.retrieval``   query pre-process + hybrid retrieval + rerank (Steps 14-18)
- ``ragnarok.generation``  context assembly, generation, post-process, grounding (Steps 19-21)
- ``ragnarok.guardrails``  input/output safety (Step 22)
- ``ragnarok.serving``     Slack + FastAPI surfaces (Step 23)
- ``ragnarok.eval``        golden-set + post-prod evaluation (Steps 24-25)
- ``ragnarok.observability`` tracing + metrics (Step 26)

Core (``config``, ``providers``, ``resilience``, ``prompts``) is stdlib/pydantic-only so the
package imports cleanly without the heavy optional integrations installed.
"""

__version__ = "0.1.0"
