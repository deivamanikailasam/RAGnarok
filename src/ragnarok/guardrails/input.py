"""Input guardrails (Step 22).

Layered cheapest-first so most traffic never touches an LLM: length -> rate limit -> PII -> injection
heuristics -> (optional) small-LLM policy classifier only when the input looks suspicious. This adds
~ms typically, not a fixed LLM tax per request.
"""

from __future__ import annotations

import re
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field

from ragnarok.config import InputGuardCfg, get_settings
from ragnarok.guardrails import pii as pii_mod
from ragnarok.tokenization import count_tokens
from ragnarok.user import User

# Known direct-injection patterns (indirect injection is handled in generation, Step 20).
_INJECTION_RE = re.compile(
    r"(?i)(ignore (all |the )?(previous|prior|above) instructions|"
    r"disregard (the |your )?(system|previous)|reveal (the |your )?(system prompt|instructions)|"
    r"you are now|act as (an? )?(dan|jailbreak)|print your (system )?prompt)"
)


@dataclass
class InputVerdict:
    allowed: bool
    sanitized: str
    reason: str = ""
    flags: list[str] = field(default_factory=list)


class RateLimiter:
    def __init__(self, max_per_min: int, clock: Callable[[], float] = time.monotonic) -> None:
        self.max = max_per_min
        self.clock = clock
        self._hits: dict[str, deque] = defaultdict(deque)

    def allow(self, user_id: str) -> bool:
        now = self.clock()
        q = self._hits[user_id]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= self.max:
            return False
        q.append(now)
        return True


_default_limiter: RateLimiter | None = None


def _limiter(cfg: InputGuardCfg) -> RateLimiter:
    global _default_limiter
    if _default_limiter is None or _default_limiter.max != cfg.rate_limit_per_min:
        _default_limiter = RateLimiter(cfg.rate_limit_per_min)
    return _default_limiter


def strip_injection(text: str) -> str:
    return _INJECTION_RE.sub("[removed]", text)


def check_input(
    query: str, user: User, *, cfg: InputGuardCfg | None = None, limiter: RateLimiter | None = None
) -> InputVerdict:
    cfg = cfg or get_settings().guardrails.input
    flags: list[str] = []

    if count_tokens(query) > cfg.max_query_tokens:
        return InputVerdict(False, query, "query_too_long", ["length"])

    if not (limiter or _limiter(cfg)).allow(user.id):
        return InputVerdict(False, query, "rate_limited", ["rate_limit"])

    spans = pii_mod.detect(query)
    if spans:
        flags.append("pii")
        if cfg.pii == "block":
            return InputVerdict(False, query, "pii_in_query", flags)
        if cfg.pii == "redact":
            query = pii_mod.redact(query, spans)

    if cfg.injection != "off" and _INJECTION_RE.search(query):
        flags.append("injection")
        if cfg.injection == "block":
            return InputVerdict(False, query, "prompt_injection", flags)
        query = strip_injection(query)

    return InputVerdict(True, query, "", flags)
