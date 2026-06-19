"""Tests for dedup_samples and content_hash in yatsaury.quality.dedup."""
from __future__ import annotations

from yatsaury.models import Citation, DatasetType, Sample
from yatsaury.quality.dedup import content_hash, dedup_samples


def make_sample(
    question: str,
    answer: str,
    quality_score: float = 50.0,
    sid: str = "s1",
) -> Sample:
    return Sample(
        id=sid,
        chunk_id="c1",
        dataset_type=DatasetType.qa,
        payload={"question": question, "answer": answer},
        source_text=f"{question} {answer}",
        supporting_quote=question,
        source_citation=Citation(title="T", source_uri="uri://x"),
        quality_score=quality_score,
    )


def test_exact_dedup_keeps_higher_score() -> None:
    """Two identical samples → only the one with higher quality_score is kept."""
    s_high = make_sample("What is Islam?", "A religion.", quality_score=80.0, sid="s1")
    s_low = make_sample("What is Islam?", "A religion.", quality_score=50.0, sid="s2")
    result = dedup_samples([s_high, s_low])
    assert len(result) == 1
    assert result[0].quality_score == 80.0


def test_near_dup_collapses() -> None:
    """Two near-identical samples → only the higher-score one is kept."""
    s_high = make_sample(
        "What is Islam?",
        "Islam is a monotheistic Abrahamic religion.",
        quality_score=90.0,
        sid="s1",
    )
    s_low = make_sample(
        "What is Islam?",
        "Islam is a monotheistic Abrahamic faith.",
        quality_score=60.0,
        sid="s2",
    )
    result = dedup_samples([s_high, s_low], similarity_threshold=0.85)
    assert len(result) == 1
    assert result[0].quality_score == 90.0


def test_different_samples_both_kept() -> None:
    """Two completely different samples → both are kept."""
    s1 = make_sample("What is Islam?", "A monotheistic religion.", quality_score=70.0, sid="s1")
    s2 = make_sample(
        "What is the capital of France?",
        "Paris is the capital of France.",
        quality_score=80.0,
        sid="s2",
    )
    result = dedup_samples([s1, s2])
    assert len(result) == 2


def test_empty_list() -> None:
    """Empty input → empty output."""
    result = dedup_samples([])
    assert result == []


def test_single_sample() -> None:
    """Single sample → returned as-is (with dedup_hash set)."""
    s = make_sample("What is Islam?", "A religion.", sid="s1")
    result = dedup_samples([s])
    assert len(result) == 1
    assert result[0].id == "s1"


def test_dedup_hash_set() -> None:
    """All returned samples have dedup_hash starting with 'sha256:'."""
    samples = [
        make_sample("What is Islam?", "A religion.", sid="s1"),
        make_sample("What is Python?", "A programming language.", sid="s2"),
    ]
    result = dedup_samples(samples)
    for sample in result:
        assert sample.dedup_hash.startswith("sha256:")


def test_no_duplicate_hashes() -> None:
    """After dedup, no two returned samples share the same dedup_hash."""
    samples = [
        make_sample("What is Islam?", "A monotheistic religion.", sid="s1"),
        make_sample("What is Python?", "A programming language.", sid="s2"),
        make_sample("What is gravity?", "A fundamental force.", sid="s3"),
    ]
    result = dedup_samples(samples)
    hashes = [s.dedup_hash for s in result]
    assert len(hashes) == len(set(hashes))


def test_content_hash_deterministic() -> None:
    """Same question+answer always produces the same hash."""
    s1 = make_sample("What is Islam?", "A religion.", sid="s1")
    s2 = make_sample("What is Islam?", "A religion.", sid="s2")
    assert content_hash(s1) == content_hash(s2)


def test_content_hash_different() -> None:
    """Different question+answer produces different hashes."""
    s1 = make_sample("What is Islam?", "A religion.", sid="s1")
    s2 = make_sample("What is Python?", "A language.", sid="s2")
    assert content_hash(s1) != content_hash(s2)


def test_exact_dedup_three_samples() -> None:
    """Three samples where two are identical and one is different → 2 kept."""
    s_dup_high = make_sample("What is Islam?", "A religion.", quality_score=90.0, sid="s1")
    s_dup_low = make_sample("What is Islam?", "A religion.", quality_score=40.0, sid="s2")
    s_different = make_sample("What is Python?", "A language.", quality_score=70.0, sid="s3")
    result = dedup_samples([s_dup_high, s_dup_low, s_different])
    assert len(result) == 2
    ids = {s.id for s in result}
    assert "s1" in ids
    assert "s3" in ids
    assert "s2" not in ids
