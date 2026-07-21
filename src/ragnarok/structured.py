"""Structured-output helper (Step 3).

Combines the provider abstraction, the resilience policy, and a Pydantic schema into one call that
returns a *validated object*. Guided decoding makes invalid JSON nearly impossible; a single
repair retry covers the rare edge (e.g. a hosted model without guided decoding).
"""

from __future__ import annotations

from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from ragnarok import resilience

T = TypeVar("T", bound=BaseModel)


async def generate_structured(
    role_name: str,
    messages: list[dict[str, str]],
    model_cls: type[T],
    *,
    fallback_role: str | None = None,
    **kw: Any,
) -> T:
    schema = model_cls.model_json_schema()
    resp = await resilience.call(
        role_name, messages, response_schema=schema, fallback_role=fallback_role, **kw
    )
    try:
        return model_cls.model_validate_json(resp.content)
    except (ValidationError, ValueError) as err:
        repair = messages + [
            {"role": "assistant", "content": resp.content},
            {
                "role": "user",
                "content": (
                    f"That did not match the required schema ({err}). "
                    "Return ONLY corrected JSON matching the schema."
                ),
            },
        ]
        resp2 = await resilience.call(role_name, repair, response_schema=schema, **kw)
        return model_cls.model_validate_json(resp2.content)
