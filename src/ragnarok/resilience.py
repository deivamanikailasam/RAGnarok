"""Centralized retry / timeout / fallback for model calls (Step 3).

Every model call inherits this policy instead of scattering try/except everywhere. On failure:
retry with exponential backoff, then fall through a *fallback ladder* (e.g. llm_large ->
llm_small), labelling the response ``degraded`` so post-prod metrics can see it (Step 25/26).
"""

from __future__ import annotations

import asyncio
from typing import Any

from ragnarok.providers import LLMResponse, role

# Exceptions we consider transient/retryable. openai isn't imported here (lazy), so match by name.
_RETRYABLE_NAMES = {
    "APITimeoutError",
    "APIConnectionError",
    "InternalServerError",
    "RateLimitError",
    "TimeoutError",
    "ConnectionError",
}


def _is_retryable(exc: BaseException) -> bool:
    return type(exc).__name__ in _RETRYABLE_NAMES


async def call(
    role_name: str,
    messages: list[dict[str, str]],
    *,
    response_schema: dict | None = None,
    fallback_role: str | None = None,
    retries: int = 3,
    base_delay: float = 0.5,
    **kw: Any,
) -> LLMResponse:
    last: BaseException | None = None
    for attempt in range(retries):
        try:
            return await role(role_name).complete(
                messages, response_schema=response_schema, **kw
            )
        except BaseException as exc:  # noqa: BLE001 - we re-raise non-retryable below
            last = exc
            if not _is_retryable(exc) or attempt == retries - 1:
                break
            await asyncio.sleep(base_delay * (2**attempt))

    if fallback_role is not None:
        resp = await role(fallback_role).complete(
            messages, response_schema=response_schema, **kw
        )
        resp.degraded = True
        return resp

    assert last is not None
    raise last
