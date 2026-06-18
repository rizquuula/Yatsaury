"""QA pair generator."""
from __future__ import annotations

from uuid import uuid4

from yatsaury.generators.base import register_generator
from yatsaury.llm.client import LLMClient
from yatsaury.llm.prompts import qa_generation_prompt
from yatsaury.models import Chunk, Citation, DatasetType, Sample


class QaGenerator:
    """Generate Q&A pairs from a chunk using an LLM."""

    dataset_type = "qa"

    def generate(self, chunk: Chunk, n: int, llm: LLMClient) -> list[Sample]:
        """Call the LLM and convert response pairs into Sample objects.

        - Returns [] if LLM responds with {"insufficient": true}
        - Drops pairs with empty supporting_quote
        """
        messages = qa_generation_prompt(chunk.text, n=n)
        response = llm.complete_json(messages)

        if response.get("insufficient"):
            return []

        pairs = response.get("pairs", [])
        samples: list[Sample] = []

        for pair in pairs:
            question = pair.get("question", "")
            answer = pair.get("answer", "")
            supporting_quote = pair.get("supporting_quote", "")

            if not supporting_quote:
                continue

            samples.append(
                Sample(
                    id=uuid4().hex,
                    chunk_id=chunk.id,
                    dataset_type=DatasetType.qa,
                    payload={"question": question, "answer": answer},
                    source_text=chunk.text,
                    supporting_quote=supporting_quote,
                    source_citation=Citation(title="", source_uri=""),
                )
            )

        return samples


# Register at import time
register_generator(QaGenerator())
