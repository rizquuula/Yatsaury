"""Tests for chunk_document — token-aware splitter."""
from __future__ import annotations

from uuid import uuid4

import pytest

from yatsaury.models import Document, SourceType
from yatsaury.processing.chunk import chunk_document


def make_doc(text: str) -> Document:
    return Document(
        id=uuid4().hex,
        source_uri="<text>",
        source_type=SourceType.text,
        raw_text=text,
    )


class TestChunkDocument:
    def test_short_text_single_chunk(self):
        doc = make_doc("Short text that fits in one chunk.")
        chunks = chunk_document(doc, chunk_size=512, overlap=64)
        assert len(chunks) == 1
        c = chunks[0]
        assert c.token_count <= 512
        assert c.char_span[0] == 0
        assert c.char_span[1] == len(doc.raw_text)
        assert c.ordinal == 0
        assert c.doc_id == doc.id

    def test_empty_text_returns_empty(self):
        doc = make_doc("")
        chunks = chunk_document(doc, chunk_size=512, overlap=64)
        assert chunks == []

    def test_long_text_multiple_chunks(self):
        # ~1500 tokens of text
        text = "The quick brown fox jumps over the lazy dog. " * 100
        doc = make_doc(text)
        chunks = chunk_document(doc, chunk_size=100, overlap=20)
        assert len(chunks) > 1
        for c in chunks:
            assert c.token_count <= 100

    def test_ordinals_sequential(self):
        text = "word " * 300
        doc = make_doc(text)
        chunks = chunk_document(doc, chunk_size=50, overlap=10)
        for i, c in enumerate(chunks):
            assert c.ordinal == i

    def test_doc_id_on_chunks(self):
        doc = make_doc("some text " * 50)
        chunks = chunk_document(doc, chunk_size=50, overlap=10)
        for c in chunks:
            assert c.doc_id == doc.id

    def test_chunk_ids_unique(self):
        text = "word " * 300
        doc = make_doc(text)
        chunks = chunk_document(doc, chunk_size=50, overlap=10)
        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_overlap_tokens_shared(self):
        """Last `overlap` tokens of chunk N appear at start of chunk N+1."""
        import tiktoken
        text = "alpha beta gamma delta epsilon zeta eta theta iota kappa " * 20
        doc = make_doc(text)
        chunks = chunk_document(doc, chunk_size=30, overlap=5)
        if len(chunks) < 2:
            pytest.skip("Need at least 2 chunks for overlap test")
        enc = tiktoken.get_encoding("cl100k_base")
        toks_0 = enc.encode(chunks[0].text)
        toks_1 = enc.encode(chunks[1].text)
        # Last 5 tokens of chunk 0 should be first 5 tokens of chunk 1
        assert toks_0[-5:] == toks_1[:5]

    def test_char_spans_cover_text(self):
        """All characters in raw_text should be covered by some chunk's span."""
        text = "word " * 200
        doc = make_doc(text)
        chunks = chunk_document(doc, chunk_size=50, overlap=10)
        # First chunk starts at 0
        assert chunks[0].char_span[0] == 0
        # Last chunk ends at or before len(text)
        assert chunks[-1].char_span[1] <= len(text)

    def test_chunk_text_matches_span(self):
        """chunk.text must equal doc.raw_text[start:end]."""
        text = "The quick brown fox. " * 50
        doc = make_doc(text)
        chunks = chunk_document(doc, chunk_size=40, overlap=8)
        for c in chunks:
            start, end = c.char_span
            assert doc.raw_text[start:end] == c.text
