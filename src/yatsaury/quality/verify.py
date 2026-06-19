"""Quality verification for Yatsaury samples."""
from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from itertools import islice

from yatsaury.llm.client import LLMClient
from yatsaury.models import DatasetType, Sample

logger = logging.getLogger(__name__)

_SCORE_RUBRIC = """
Score (quality_score, integer 0–100):
  0–20  : Unusable — multiple critical flaws, should be discarded
  21–40 : Poor — major issues, not suitable for training
  41–60 : Mediocre — usable but noisy, needs curation
  61–80 : Good — minor issues, acceptable for most use cases
  81–100: Excellent — high quality, ready to use as-is

Evaluate across four dimensions and mention each briefly in the rationale:
  Grounding    — Is every claim supported by the source text? No hallucination?
  Clarity      — Is the text clearly written, specific, and unambiguous?
  Completeness — Does the content fully cover what is needed without gaps?
  Relevance    — Is this sample useful and on-topic for its dataset purpose?

Set is_accepted to true if quality_score >= 60.
"""

_SINGLE_SCHEMA = (
    '{"quality_score": <int 0-100>, "is_accepted": <bool>, '
    '"rationale": "<one sentence per dimension>"}'
)

_BATCH_SCHEMA = (
    '{"results": [{"quality_score": <int 0-100>, "is_accepted": <bool>, '
    '"rationale": "<string>"}, ...]}'
)


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

    quality_score: int  # 0–100
    is_accepted: bool
    rationale: str


# ── Per-type system prompts ──────────────────────────────────────────────────

def _qa_system() -> str:
    return (
        "You are a dataset quality judge for question-answering pairs.\n"
        "Given a source text, a question, an answer, and a supporting quote, "
        "rate the overall quality of this QA training sample.\n"
        + _SCORE_RUBRIC
        + "Respond with valid JSON only, using this schema:\n"
        + _SINGLE_SCHEMA
    )


def _instruction_system() -> str:
    return (
        "You are a dataset quality judge for instruction-tuning triples.\n"
        "Given a source text, an instruction, an optional input, an output, "
        "and a supporting quote, rate the overall quality of this training sample.\n"
        + _SCORE_RUBRIC
        + "Respond with valid JSON only, using this schema:\n"
        + _SINGLE_SCHEMA
    )


def _summary_system() -> str:
    return (
        "You are a dataset quality judge for summarization samples.\n"
        "Given a passage, a summary, key points, and a supporting quote, "
        "rate the overall quality of this summarization training sample.\n"
        + _SCORE_RUBRIC
        + "Respond with valid JSON only, using this schema:\n"
        + _SINGLE_SCHEMA
    )


# ── Per-type user message builders ──────────────────────────────────────────

def _qa_user(sample: Sample) -> str:
    q = sample.payload.get("question", "")
    a = sample.payload.get("answer", "")
    return (
        f"Source text:\n{sample.source_text}\n\n"
        f"Question: {q}\n"
        f"Answer: {a}\n"
        f"Supporting quote: {sample.supporting_quote}"
    )


def _instruction_user(sample: Sample) -> str:
    instr = sample.payload.get("instruction", "")
    inp = sample.payload.get("input", "")
    out = sample.payload.get("output", "")
    return (
        f"Source text:\n{sample.source_text}\n\n"
        f"Instruction: {instr}\n"
        f"Input: {inp}\n"
        f"Output: {out}\n"
        f"Supporting quote: {sample.supporting_quote}"
    )


def _summary_user(sample: Sample) -> str:
    summary = sample.payload.get("summary", "")
    key_points = sample.payload.get("key_points", [])
    kp_text = "; ".join(key_points) if key_points else ""
    return (
        f"Passage:\n{sample.source_text}\n\n"
        f"Summary: {summary}\n"
        f"Key points: {kp_text}\n"
        f"Supporting quote: {sample.supporting_quote}"
    )


def _judge_messages(sample: Sample) -> list[dict]:
    """Build single-item judge messages, dispatching on dataset_type."""
    if sample.dataset_type == DatasetType.instruction:
        system = _instruction_system()
        user = _instruction_user(sample)
    elif sample.dataset_type == DatasetType.summary:
        system = _summary_system()
        user = _summary_user(sample)
    else:
        system = _qa_system()
        user = _qa_user(sample)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# ── Batch prompt helpers ─────────────────────────────────────────────────────

def _batch_system(dataset_type: DatasetType) -> str:
    if dataset_type == DatasetType.instruction:
        kind = "instruction-tuning triples"
    elif dataset_type == DatasetType.summary:
        kind = "summarization samples"
    else:
        kind = "question-answering pairs"
    return (
        f"You are a dataset quality judge for {kind}.\n"
        "You will evaluate multiple samples that share the same source text.\n"
        + _SCORE_RUBRIC
        + f"Respond with valid JSON only, using this schema:\n{_BATCH_SCHEMA}\n"
        "Return exactly one result object per numbered item, in order."
    )


def _batch_item_text(sample: Sample, index: int) -> str:
    if sample.dataset_type == DatasetType.instruction:
        instr = sample.payload.get("instruction", "")
        inp = sample.payload.get("input", "")
        out = sample.payload.get("output", "")
        return (
            f"Item {index + 1}:\n"
            f"  Instruction: {instr}\n"
            f"  Input: {inp}\n"
            f"  Output: {out}\n"
            f"  Supporting quote: {sample.supporting_quote}"
        )
    if sample.dataset_type == DatasetType.summary:
        summary = sample.payload.get("summary", "")
        key_points = sample.payload.get("key_points", [])
        kp_text = "; ".join(key_points) if key_points else ""
        return (
            f"Item {index + 1}:\n"
            f"  Summary: {summary}\n"
            f"  Key points: {kp_text}\n"
            f"  Supporting quote: {sample.supporting_quote}"
        )
    q = sample.payload.get("question", "")
    a = sample.payload.get("answer", "")
    return (
        f"Item {index + 1}:\n"
        f"  Question: {q}\n"
        f"  Answer: {a}\n"
        f"  Supporting quote: {sample.supporting_quote}"
    )


# ── Parsing ──────────────────────────────────────────────────────────────────

def _parse_result(raw: dict) -> JudgeResult:
    required = {"quality_score", "is_accepted", "rationale"}
    missing = required - raw.keys()
    if missing:
        raise ValueError(f"LLM response missing required keys: {missing}. Got: {raw!r}")
    score = int(round(float(raw["quality_score"])))
    score = max(0, min(100, score))
    return JudgeResult(
        quality_score=score,
        is_accepted=bool(raw["is_accepted"]),
        rationale=str(raw["rationale"]),
    )


# ── Public judge API ─────────────────────────────────────────────────────────

def _make_active_llm(llm: LLMClient, judge_model: str) -> LLMClient:
    if judge_model:
        return LLMClient(
            base_url=str(llm._client.base_url),
            api_key=llm._client.api_key,
            model=judge_model,
        )
    return llm


def judge_sample(sample: Sample, llm: LLMClient, judge_model: str = "") -> JudgeResult:
    """Call the LLM to judge quality of a single sample (score 0–100).

    Prompt is customized per dataset_type (qa / instruction / summary).
    If judge_model is non-empty, a new LLMClient is created with that model.
    Raises ValueError if LLM returns invalid JSON or missing required keys.
    """
    results = judge_sample_batch([sample], llm, judge_model=judge_model)
    return results[0]


def judge_sample_batch(
    samples: list[Sample],
    llm: LLMClient,
    judge_model: str = "",
) -> list[JudgeResult]:
    """Call the LLM to judge multiple samples in a single request.

    All samples should share the same source_text and dataset_type (the first
    sample's values are used as the shared context). For a single sample, the
    per-type single-item prompt is used. For multiple samples, a numbered batch
    prompt is used with one shared source_text header.

    Returns a list of JudgeResult in the same order as input.
    On count mismatch in the batch response, logs a warning and returns [].
    Raises ValueError on invalid JSON or missing required keys.
    """
    if not samples:
        return []

    active_llm = _make_active_llm(llm, judge_model)

    if len(samples) == 1:
        messages = _judge_messages(samples[0])
        raw = active_llm.complete_json(messages)
        return [_parse_result(raw)]

    # Batch path
    system_msg = _batch_system(samples[0].dataset_type)
    shared_source = samples[0].source_text
    items_text = "\n\n".join(_batch_item_text(s, i) for i, s in enumerate(samples))
    user_msg = f"Source text (shared):\n{shared_source}\n\n{items_text}"
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
    raw = active_llm.complete_json(messages)

    results_raw = raw.get("results", [])
    if len(results_raw) != len(samples):
        logger.warning(
            "Batch judge returned %d results for %d samples — dropping sub-batch",
            len(results_raw),
            len(samples),
        )
        return []

    return [_parse_result(r) for r in results_raw]


def _batched(iterable, n: int):
    it = iter(iterable)
    while chunk := list(islice(it, n)):
        yield chunk


def verify_samples(
    samples: list[Sample],
    llm: LLMClient,
    min_score: float = 70.0,
    judge_model: str = "",
    use_judge: bool = True,
    batch_size: int = 1,
) -> list[Sample]:
    """Run all quality checks and return only passing samples.

    For each sample:
    1. quote_check() — drop immediately if the supporting_quote is not in source_text
    2. If use_judge: call the LLM judge (single or batch) — set quality_score and verified
    3. Drop if quality_score < min_score (scale 0–100)

    When batch_size > 1, samples sharing (source_text, dataset_type) are grouped
    and sent together in sub-batches of up to batch_size per LLM call.

    Returns list of passing samples with quality_score and verified fields updated.
    """
    quote_passed = [s for s in samples if quote_check(s)]

    if not use_judge:
        return quote_passed

    groups: dict[tuple[str, str], list[Sample]] = defaultdict(list)
    for sample in quote_passed:
        key = (sample.source_text, sample.dataset_type.value)
        groups[key].append(sample)

    passing: list[Sample] = []
    for group in groups.values():
        for sub_batch in _batched(group, batch_size):
            results = judge_sample_batch(sub_batch, llm, judge_model=judge_model)
            if not results:
                continue
            for sample, result in zip(sub_batch, results):
                updated = sample.model_copy(
                    update={
                        "quality_score": float(result.quality_score),
                        "verified": result.is_accepted,
                    }
                )
                if updated.quality_score >= min_score:
                    passing.append(updated)

    return passing
