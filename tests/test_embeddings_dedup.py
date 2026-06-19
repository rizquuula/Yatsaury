"""Tests for embeddings-based deduplication."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tests.helpers import make_qa_sample


@pytest.fixture
def mock_embed_client():
    return MagicMock()


def test_cosine_similarity_identical():
    from yatsaury.quality.dedup import cosine_similarity
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal():
    from yatsaury.quality.dedup import cosine_similarity
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_dedup_by_embeddings_identical(mock_embed_client):
    """Two samples with identical embeddings → 1 kept (higher quality_score wins)."""
    from yatsaury.quality.dedup import dedup_by_embeddings

    mock_embed_client.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=[1.0, 0.0]), MagicMock(embedding=[1.0, 0.0])]
    )
    s1 = make_qa_sample(quality_score=80.0)
    s2 = make_qa_sample(quality_score=60.0)
    result = dedup_by_embeddings([s1, s2], mock_embed_client, threshold=0.95)
    assert len(result) == 1
    assert result[0] is s1  # higher score wins


def test_dedup_by_embeddings_different(mock_embed_client):
    """Two samples with orthogonal embeddings → both kept."""
    from yatsaury.quality.dedup import dedup_by_embeddings

    mock_embed_client.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=[1.0, 0.0]), MagicMock(embedding=[0.0, 1.0])]
    )
    s1 = make_qa_sample()
    s2 = make_qa_sample()
    result = dedup_by_embeddings([s1, s2], mock_embed_client, threshold=0.95)
    assert len(result) == 2


def test_embed_texts_called_with_combined_text(mock_embed_client):
    """dedup_by_embeddings calls embeddings with the sample text."""
    from yatsaury.quality.dedup import dedup_by_embeddings

    mock_embed_client.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=[1.0, 0.0])]
    )
    s = make_qa_sample(question="Q?", answer="A.")
    dedup_by_embeddings([s], mock_embed_client, threshold=0.95)
    call_input = mock_embed_client.embeddings.create.call_args[1]["input"]
    assert any("Q?" in t or "A." in t for t in call_input)
