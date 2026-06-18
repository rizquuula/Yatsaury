"""Quality verification for Yatsaury samples."""
from __future__ import annotations

import re
from dataclasses import dataclass

from yatsaury.llm.client import LLMClient
from yatsaury.models import Sample


def normalize_ws(text: str) -> str:
    """Collapse any whitespace run to a single space and strip."""
    return re.sub(r'\s+', ' ', text).strip()


def quote_check(sample: Sample) -> bool:
    """Return True if sample.supporting_quote is a real (whitespace-fuzzy) substring
    of sample.source_text.

    Whitespace-fuzzy: normalize all whitespace runs to a single space before comparison.
    Returns False if supporting_quote is empty.
    """
    if not sample.supporting_quote:
        return False
    normalized_quote = normalize_ws(sample.supporting_quote)
    normalized_source = normalize_ws(sample.source_text)
    return normalized_quote in normalized_source


@dataclass
class JudgeResult:
    """Result from an LLM judge evaluation."""

    grounding_score: float  # 0.0–1.0
    is_supported: bool
    rationale: str


def judge_prompt(sample: Sample) -> list[dict]:
    """Return messages for the judge call."""
    question = sample.payload.get("question", "")
    answer = sample.payload.get("answer", "")
    system_msg = (
        "You are a grounding judge. Given a source text, a question, an answer, "
        "and a supporting quote, evaluate whether the answer is grounded in the source. "
        "Respond with valid JSON only, using this schema: "
        '{"grounding_score": <float 0.0-1.0>, "is_supported": <bool>, "rationale": <string>}'
    )
    user_msg = (
        f"Source text:\n{sample.source_text}\n\n"
        f"Question: {question}\n"
        f"Answer: {answer}\n"
        f"Supporting quote: {sample.supporting_quote}\n\n"
        "Return JSON with grounding_score, is_supported, and rationale."
    )
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def judge_sample(sample: Sample, llm: LLMClient, judge_model: str = "") -> JudgeResult:
    """Call the LLM as a judge to score sample grounding.

    Prompt: given source_text and (question, answer, supporting_quote), return JSON:
    {"grounding_score": 0.95, "is_supported": true, "rationale": "The answer..."}

    If judge_model is non-empty, create a NEW LLMClient with that model (same base_url and
    api_key as the passed llm). Access llm._model, llm._client.base_url, and llm._client.api_key.

    Raises ValueError if LLM returns invalid JSON or missing required keys.
    Clamps grounding_score to [0.0, 1.0].
    """
    active_llm: LLMClient
    if judge_model:
        active_llm = LLMClient(
            base_url=str(llm._client.base_url),
            api_key=llm._client.api_key,
            model=judge_model,
        )
    else:
        active_llm = llm

    messages = judge_prompt(sample)
    raw = active_llm.complete_json(messages)

    required_keys = {"grounding_score", "is_supported", "rationale"}
    missing = required_keys - raw.keys()
    if missing:
        raise ValueError(f"LLM response missing required keys: {missing}. Got: {raw!r}")

    score = float(raw["grounding_score"])
    score = max(0.0, min(1.0, score))

    return JudgeResult(
        grounding_score=score,
        is_supported=bool(raw["is_supported"]),
        rationale=str(raw["rationale"]),
    )


def verify_samples(
    samples: list[Sample],
    llm: LLMClient,
    min_score: float = 0.7,
    judge_model: str = "",
    use_judge: bool = True,
) -> list[Sample]:
    """Run all quality checks and return only passing samples.

    For each sample:
    1. quote_check() — if fails, drop immediately (grounding_score = 0, verified = False)
    2. If use_judge: judge_sample() -> set grounding_score, verified = is_supported
    3. Drop if grounding_score < min_score

    Returns list of passing samples with grounding_score and verified fields updated.
    """
    passing: list[Sample] = []

    for sample in samples:
        if not quote_check(sample):
            continue

        if use_judge:
            result = judge_sample(sample, llm, judge_model=judge_model)
            updated = sample.model_copy(
                update={
                    "grounding_score": result.grounding_score,
                    "verified": result.is_supported,
                }
            )
            if updated.grounding_score < min_score:
                continue
            passing.append(updated)
        else:
            passing.append(sample)

    return passing
