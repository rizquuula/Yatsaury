"""Exact and near-duplicate removal for Sample lists."""

from __future__ import annotations

import hashlib

from rapidfuzz import fuzz

from yatsaury.models import Sample
from yatsaury.quality.verify import normalize_ws


def content_hash(sample: Sample) -> str:
    """SHA-256 of normalized (question + answer) for exact dedup."""
    question = sample.payload.get("question", "")
    answer = sample.payload.get("answer", "")
    text = normalize_ws(question + " " + answer)
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


def dedup_samples(
    samples: list[Sample],
    similarity_threshold: float = 0.92,
) -> list[Sample]:
    """Remove duplicates from samples list.

    1. Exact dedup: group by content_hash. Keep one per group (highest grounding_score).
    2. Near-dup: among remaining, if rapidfuzz.fuzz.ratio(a_text, b_text) >= threshold*100,
       keep the one with higher grounding_score.

    Returns deduplicated list. Sets dedup_hash on every output sample.
    similarity_threshold: float in [0, 1]; compared as threshold*100 against fuzz.ratio.
    """
    if not samples:
        return []

    # Step 1 — Exact dedup: group by content_hash, keep highest grounding_score per group.
    hash_to_best: dict[str, Sample] = {}
    for sample in samples:
        h = content_hash(sample)
        existing = hash_to_best.get(h)
        if existing is None or sample.grounding_score > existing.grounding_score:
            hash_to_best[h] = sample

    exact_deduped = list(hash_to_best.values())

    # Step 2 — Near-dup removal: greedy pairwise comparison.
    threshold_score = similarity_threshold * 100

    def _text(s: Sample) -> str:
        q = s.payload.get("question", "")
        a = s.payload.get("answer", "")
        return normalize_ws(q + " " + a)

    accepted: list[Sample] = []
    for candidate in exact_deduped:
        candidate_text = _text(candidate)
        dominated = False
        to_replace: int | None = None

        for i, kept in enumerate(accepted):
            kept_text = _text(kept)
            ratio = fuzz.ratio(candidate_text, kept_text)
            if ratio >= threshold_score:
                # Near-duplicate found: keep the one with higher grounding_score.
                if candidate.grounding_score > kept.grounding_score:
                    to_replace = i
                else:
                    dominated = True
                break

        if dominated:
            continue
        if to_replace is not None:
            accepted[to_replace] = candidate
        else:
            accepted.append(candidate)

    # Step 3 — Set dedup_hash on every returned sample.
    result = [
        sample.model_copy(update={"dedup_hash": content_hash(sample)})
        for sample in accepted
    ]

    return result


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def embed_texts(
    texts: list[str],
    client,
    model: str = "text-embedding-3-small",
) -> list[list[float]]:
    """Call client.embeddings.create(input=texts, model=model).

    Returns list of embedding vectors.
    """
    response = client.embeddings.create(input=texts, model=model)
    return [item.embedding for item in response.data]


def dedup_by_embeddings(
    samples: list[Sample],
    client,
    threshold: float = 0.95,
    model: str = "text-embedding-3-small",
) -> list[Sample]:
    """Embed each sample's (question + answer) text. Remove near-dups by cosine similarity.

    If similarity >= threshold: keep the one with higher grounding_score.
    Returns deduplicated list.
    This is an OPTIONAL alternative to rapidfuzz; not wired into the default pipeline.
    """
    if not samples:
        return []

    texts = [
        sample.payload.get("question", "") + " " + sample.payload.get("answer", "")
        for sample in samples
    ]
    vectors = embed_texts(texts, client, model=model)

    kept_indices: list[int] = []
    for i in range(len(samples)):
        dominated = False
        to_replace: int | None = None
        for j_pos, j in enumerate(kept_indices):
            sim = cosine_similarity(vectors[i], vectors[j])
            if sim >= threshold:
                if samples[i].grounding_score > samples[j].grounding_score:
                    to_replace = j_pos
                else:
                    dominated = True
                break
        if dominated:
            continue
        if to_replace is not None:
            kept_indices[to_replace] = i
        else:
            kept_indices.append(i)

    return [samples[i] for i in kept_indices]
