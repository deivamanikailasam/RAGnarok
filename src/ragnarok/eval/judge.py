"""LLM-as-Judge (Step 24).

Structured, rubric-bound, reference-guided scoring for what lexical metrics can't capture
(faithfulness, relevancy, correctness). Structured output makes scores aggregatable; the judge is
calibrated against human labels periodically (we evaluate the evaluator). Uses a strong/offline
model role so judging is cheap and infrequent.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ragnarok.config import get_settings
from ragnarok.prompts import prompts
from ragnarok.structured import generate_structured


class Judgement(BaseModel):
    faithfulness: int = Field(ge=1, le=5)
    relevancy: int = Field(ge=1, le=5)
    correctness: int = Field(ge=1, le=5)
    reasoning: str = ""
    unsupported_claims: list[str] = Field(default_factory=list)

    def normalized(self) -> dict[str, float]:
        return {
            "faithfulness": (self.faithfulness - 1) / 4,
            "relevancy": (self.relevancy - 1) / 4,
            "correctness": (self.correctness - 1) / 4,
        }


async def judge_answer(question: str, answer: str, context: str, ideal: str) -> Judgement:
    role_name = get_settings().eval.judge_role
    msgs = prompts().render(
        "llm_judge", "latest", question=question, answer=answer, context=context, ideal=ideal
    )
    return await generate_structured(role_name, msgs, Judgement)
