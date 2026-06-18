"""Token-aware document chunker using tiktoken."""
from __future__ import annotations

import tiktoken

from yatsaury.models import Chunk, Document


def chunk_document(
    doc: Document,
    chunk_size: int = 512,
    overlap: int = 64,
    encoding_name: str = "cl100k_base",
) -> list[Chunk]:
    """Split doc.raw_text into overlapping token-bounded chunks.

    Each chunk:
    - token_count <= chunk_size
    - overlaps with the next chunk by `overlap` tokens
    - char_span is (start_char, end_char) in doc.raw_text
    - ordinal is 0-based index
    """
    text = doc.raw_text
    if not text:
        return []

    enc = tiktoken.get_encoding(encoding_name)
    tokens = enc.encode(text)
    total = len(tokens)

    if total == 0:
        return []

    chunks: list[Chunk] = []
    start = 0  # token index

    while start < total:
        end = min(start + chunk_size, total)
        chunk_tokens = tokens[start:end]
        chunk_text = enc.decode(chunk_tokens)

        # Find char_start by searching for chunk_text in the original
        search_from = len(enc.decode(tokens[:start])) if start > 0 else 0
        char_start = text.find(chunk_text, max(0, search_from - 10))
        if char_start == -1:
            char_start = search_from  # fallback
        char_end = char_start + len(chunk_text)

        # Clamp to text bounds
        char_end = min(char_end, len(text))

        ordinal = len(chunks)
        chunk_id = f"chk_{doc.id[:8]}_{ordinal:04d}"

        chunks.append(
            Chunk(
                id=chunk_id,
                doc_id=doc.id,
                text=chunk_text,
                token_count=len(chunk_tokens),
                char_span=(char_start, char_end),
                ordinal=ordinal,
            )
        )

        if end >= total:
            break
        # Advance by (chunk_size - overlap) tokens
        step = chunk_size - overlap
        start += max(1, step)  # guard against zero/negative step

    return chunks
