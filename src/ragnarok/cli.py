"""RAGnarok command-line interface.

Dependency-light on purpose: ``ragnarok doctor`` uses only the standard library so it works on a
fresh clone before the heavy extras are installed. Subcommands added in later steps
(``ingest``, ``ask``, ``eval``, ``serve``) lazy-import their modules so importing the CLI never
pulls in optional dependencies.

    ragnarok doctor          # health-check every dependency (Step 1)
    ragnarok ingest PATH     # offline ingestion (Steps 4-12)
    ragnarok ask "..."       # one-shot query against the pipeline (Steps 14-21)
    ragnarok eval --suite X  # run an evaluation suite (Steps 24-25)
    ragnarok serve           # run the FastAPI + Slack service (Step 23)
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# `ragnarok doctor` — dependency readiness table (Step 1)
# ---------------------------------------------------------------------------

DEFAULT_ENDPOINTS: dict[str, tuple[str, str]] = {
    # name          -> (env var,               default url)
    "qdrant": ("QDRANT_URL", "http://localhost:6333"),
    "postgres": ("DATABASE_URL", "postgresql://localhost:5432/ragnarok"),
    "redis": ("REDIS_URL", "redis://localhost:6379/0"),
    "langfuse": ("LANGFUSE_HOST", "http://localhost:3000"),
    "prometheus": ("PROMETHEUS_URL", "http://localhost:9090"),
    "llm_large": ("LLM_LARGE_BASE_URL", "http://localhost:8001/v1"),
    "llm_small": ("LLM_SMALL_BASE_URL", "http://localhost:8002/v1"),
    "embedding": ("EMBEDDING_BASE_URL", "http://localhost:7997"),
    "reranker": ("RERANKER_URL", "http://localhost:7997"),
}


@dataclass
class Check:
    name: str
    endpoint: str
    ok: bool
    detail: str


def _tcp_ok(host: str, port: int, timeout: float = 1.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _http_detail(url: str) -> str:
    try:
        with urllib.request.urlopen(url, timeout=2.0) as r:  # noqa: S310 (local health check)
            return f"HTTP {r.status}"
    except urllib.error.HTTPError as e:
        return f"HTTP {e.code}"  # reachable, non-200 is fine for a liveness ping
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return str(e.reason if hasattr(e, "reason") else e)


def check_endpoint(name: str, url: str) -> Check:
    parsed = urlparse(url)
    host = parsed.hostname or "localhost"
    default_ports = {"http": 80, "https": 443, "postgresql": 5432, "redis": 6379}
    port = parsed.port or default_ports.get(parsed.scheme, 80)
    reachable = _tcp_ok(host, port)
    if not reachable:
        return Check(name, url, False, "unreachable (is the service up?)")
    detail = _http_detail(url) if parsed.scheme in ("http", "https") else "tcp ok"
    return Check(name, url, True, detail)


def cmd_doctor(_args: argparse.Namespace) -> int:
    checks = [
        check_endpoint(name, os.environ.get(env, default))
        for name, (env, default) in DEFAULT_ENDPOINTS.items()
    ]
    width = max(len(c.name) for c in checks)
    ep_width = max(len(c.endpoint) for c in checks)
    print(f"{'component'.ljust(width)}  {'endpoint'.ljust(ep_width)}  status  detail")
    for c in checks:
        status = "ok  " if c.ok else "FAIL"
        print(f"{c.name.ljust(width)}  {c.endpoint.ljust(ep_width)}  {status}    {c.detail}")
    failed = [c.name for c in checks if not c.ok]
    if failed:
        print(f"\n{len(failed)} component(s) not ready: {', '.join(failed)}", file=sys.stderr)
        return 1
    print("\nAll components ready.")
    return 0


# ---------------------------------------------------------------------------
# Subcommands wired in later steps (lazy imports keep CLI import cheap)
# ---------------------------------------------------------------------------


def cmd_ingest(args: argparse.Namespace) -> int:
    from ragnarok.ingestion.pipeline import ingest_path

    return ingest_path(args.path, full_rebuild=args.full_rebuild)


def cmd_ask(args: argparse.Namespace) -> int:
    import asyncio

    from ragnarok.pipeline import answer_once

    result = asyncio.run(answer_once(args.query))
    print(result.text)
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    import asyncio

    from ragnarok.eval.run import run_suite

    report = asyncio.run(run_suite(args.suite))
    return 0 if report.passed else 2


def cmd_serve(args: argparse.Namespace) -> int:
    from ragnarok.serving.app import run

    run(host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ragnarok", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="health-check every dependency").set_defaults(func=cmd_doctor)

    pi = sub.add_parser("ingest", help="ingest a corpus (offline plane)")
    pi.add_argument("path")
    pi.add_argument("--full-rebuild", action="store_true", help="ignore hash cache; re-embed all")
    pi.set_defaults(func=cmd_ingest)

    pa = sub.add_parser("ask", help="ask a question against the pipeline")
    pa.add_argument("query")
    pa.set_defaults(func=cmd_ask)

    pe = sub.add_parser("eval", help="run an evaluation suite")
    pe.add_argument("--suite", default="golden")
    pe.set_defaults(func=cmd_eval)

    ps = sub.add_parser("serve", help="run the FastAPI + Slack service")
    ps.add_argument("--host", default="0.0.0.0")  # noqa: S104
    ps.add_argument("--port", type=int, default=8000)
    ps.set_defaults(func=cmd_serve)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
