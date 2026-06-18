"""Text cleaner — normalizes extracted text for chunking."""
from __future__ import annotations

import re


def clean_text(text: str) -> str:
    """Normalize extracted text for chunking.

    Steps applied in order:
    1. Normalize line endings: \\r\\n → \\n
    2. De-hyphenate line breaks: "some-\\nword" → "someword"
    3. Normalize whitespace within each line: collapse spaces/tabs to single space
    4. Strip leading/trailing whitespace from each line
    5. Strip repeated blank lines: >2 consecutive newlines → exactly 2
    6. Strip leading/trailing whitespace from the whole text

    Does NOT remove content, page numbers, or headers.
    """
    if not text:
        return text

    # 1. Normalize CRLF to LF
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 2. De-hyphenate: "word-\n" → "word" (join hyphenated line breaks)
    text = re.sub(r"-\n", "", text)

    # 3 & 4. Per-line: collapse internal whitespace, strip edges
    lines = text.split("\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in lines]
    text = "\n".join(lines)

    # 5. Collapse runs of more than 2 consecutive newlines down to 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 6. Strip whole-text whitespace
    return text.strip()
