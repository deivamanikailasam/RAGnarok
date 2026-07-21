"""Role-based, provider-agnostic model access (Step 3).

Application code asks for a *role* (``llm_large``, ``llm_small``) and gets back a client that
speaks the OpenAI-compatible protocol. Because Ollama, vLLM, OpenAI, Azure, Together all speak
that protocol, "local vs hosted" is just a base-URL string in config — no vendor lock-in, and one
place to add caching/tracing/retries.

Structured output: pass ``response_schema`` (a JSON schema) and the model server's guided-decoding
feature constrains the output to valid JSON — fewer tokens, zero parse failures (Step 3 rationale).

Testing: register a ``FakeLLM`` for a role with ``set_role_client`` so pipelines run without a
live model server.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Protocol

from ragnarok.config import LLMRole, get_settings


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class LLMResponse:
    content: str
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    degraded: bool = False

    def json(self) -> Any:
        return json.loads(self.content)


class LLMClient(Protocol):
    role_name: str
    model: str

    async def complete(
        self, messages: list[dict[str, str]], *, response_schema: dict | None = None, **kw: Any
    ) -> LLMResponse: ...

    async def stream(
        self, messages: list[dict[str, str]], **kw: Any
    ) -> Any: ...  # -> AsyncIterator[str]


class OpenAICompatClient:
    """Wraps an OpenAI-compatible endpoint (Ollama/vLLM/OpenAI/…)."""

    def __init__(self, role_name: str, cfg: LLMRole) -> None:
        self.role_name = role_name
        self.cfg = cfg
        self.model = cfg.model
        self._client: Any | None = None

    def _c(self) -> Any:
        if self._client is None:
            from openai import AsyncOpenAI  # lazy: import only when a real call happens

            self._client = AsyncOpenAI(
                base_url=self.cfg.base_url,
                api_key=self.cfg.api_key or "sk-local",
                timeout=self.cfg.timeout_s,
                max_retries=0,  # retries handled centrally in resilience.py
            )
        return self._client

    def _extra(self, response_schema: dict | None) -> dict:
        if not response_schema:
            return {}
        # vLLM/Ollama accept guided_json; OpenAI accepts response_format json_schema. Send both
        # under extra_body so whichever server is behind the endpoint applies the constraint.
        return {"extra_body": {"guided_json": response_schema}}

    async def complete(
        self, messages: list[dict[str, str]], *, response_schema: dict | None = None, **kw: Any
    ) -> LLMResponse:
        resp = await self._c().chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=kw.pop("temperature", self.cfg.temperature),
            max_tokens=kw.pop("max_tokens", self.cfg.max_tokens),
            **self._extra(response_schema),
            **kw,
        )
        usage = Usage(
            prompt_tokens=getattr(resp.usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(resp.usage, "completion_tokens", 0) or 0,
        )
        return LLMResponse(resp.choices[0].message.content or "", usage, self.model)

    async def stream(self, messages: list[dict[str, str]], **kw: Any) -> Any:
        stream = await self._c().chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=kw.pop("temperature", self.cfg.temperature),
            max_tokens=kw.pop("max_tokens", self.cfg.max_tokens),
            stream=True,
            **kw,
        )

        async def _gen() -> Any:
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta

        return _gen()


class FakeLLM:
    """Programmable client for tests / offline runs.

    Provide a ``handler(messages, response_schema) -> str`` or a static ``response`` string.
    """

    def __init__(
        self,
        role_name: str = "fake",
        *,
        response: str | None = None,
        handler: Callable[[list[dict], dict | None], str] | None = None,
    ) -> None:
        self.role_name = role_name
        self.model = "fake"
        self._response = response
        self._handler = handler
        self.calls: list[dict] = []

    def _resolve(self, messages: list[dict], schema: dict | None) -> str:
        if self._handler is not None:
            return self._handler(messages, schema)
        if self._response is not None:
            return self._response
        return "{}" if schema else ""

    async def complete(
        self, messages: list[dict[str, str]], *, response_schema: dict | None = None, **kw: Any
    ) -> LLMResponse:
        self.calls.append({"messages": messages, "schema": response_schema})
        content = self._resolve(messages, response_schema)
        return LLMResponse(content, Usage(len(str(messages)) // 4, len(content) // 4), "fake")

    async def stream(self, messages: list[dict[str, str]], **kw: Any) -> Any:
        content = self._resolve(messages, None)

        async def _gen() -> Any:
            for word in content.split(" "):
                yield word + " "

        return _gen()


# --------------------------------------------------------------------------- registry

_clients: dict[str, LLMClient] = {}


def role(name: str) -> LLMClient:
    """Return the client for a model role, building it from settings on first use."""
    if name in _clients:
        return _clients[name]
    cfg = getattr(get_settings().models, name)
    client: LLMClient = OpenAICompatClient(name, cfg)
    _clients[name] = client
    return client


def set_role_client(name: str, client: LLMClient) -> None:
    """Override a role's client (tests / offline)."""
    _clients[name] = client


def reset_clients() -> None:
    _clients.clear()
