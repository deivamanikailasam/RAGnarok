"""Cost accounting (Step 26).

Even local models cost GPU-seconds/latency/capacity, so we account for them as if billed. Maps
(model, tokens) -> USD for hosted roles and -> a GPU-second-equivalent for local roles, so
"cost per query" is comparable whether a role is local or hosted — making the local↔hosted
trade-off (Step 3) data-driven and surfacing the token-heavy stages Step 14 targets.
"""

from __future__ import annotations

from ragnarok.providers import Usage

# USD per 1K tokens (in, out). Local models use a GPU-second-equivalent proxy.
PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o": (0.0025, 0.01),
    "gpt-4o-mini": (0.00015, 0.0006),
    # local models: tiny proxy cost so token-heavy stages still show up in dashboards
    "qwen2.5:32b-instruct": (0.00010, 0.00030),
    "qwen2.5:7b-instruct": (0.00002, 0.00006),
}
_DEFAULT_LOCAL = (0.00005, 0.00015)


def cost_of(model: str, usage: Usage) -> float:
    in_rate, out_rate = PRICING.get(model, _DEFAULT_LOCAL)
    return (usage.prompt_tokens / 1000) * in_rate + (usage.completion_tokens / 1000) * out_rate
