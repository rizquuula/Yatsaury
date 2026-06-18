# tests/helpers.py
from uuid import uuid4

from yatsaury.models import Chunk, Citation, DatasetType, Sample


def make_chunk(text: str = "Test source text.") -> Chunk:
    return Chunk(
        id=uuid4().hex, doc_id="doc1", text=text,
        token_count=len(text.split()), char_span=(0, len(text)), ordinal=0,
    )


def make_qa_sample(
    question="Q?", answer="A.", quote="A.", fact_id=None, grounding_score=None
) -> Sample:
    kwargs = dict(
        id=uuid4().hex, chunk_id="chk1", dataset_type=DatasetType.qa,
        payload={"question": question, "answer": answer},
        source_text=quote, supporting_quote=quote,
        source_citation=Citation(title="", source_uri=""),
        fact_id=fact_id,
    )
    if grounding_score is not None:
        kwargs["grounding_score"] = grounding_score
    return Sample(**kwargs)


def make_sample(fact_id=None) -> Sample:
    return make_qa_sample(fact_id=fact_id)
