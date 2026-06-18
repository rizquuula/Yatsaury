"""Tests for schema adapters: ChatmlSchema and QaSchema."""
from __future__ import annotations

from uuid import uuid4

import pytest

from yatsaury.models import Citation, DatasetType, Sample
from yatsaury.schemas.alpaca import AlpacaSchema
from yatsaury.schemas.base import get_schema, list_schemas
from yatsaury.schemas.chatml import ChatmlSchema
from yatsaury.schemas.completion import CompletionSchema
from yatsaury.schemas.qa import QaSchema
from yatsaury.schemas.rag import RagSchema
from yatsaury.schemas.raw import RawSchema
from yatsaury.schemas.sharegpt import ShareGptSchema


def make_qa_sample(
    question: str = "What is Islam?",
    answer: str = "A monotheistic religion.",
) -> Sample:
    return Sample(
        id=uuid4().hex,
        chunk_id="chk_test_0000",
        dataset_type=DatasetType.qa,
        payload={"question": question, "answer": answer},
        source_text="Islam is a monotheistic religion.",
        supporting_quote="monotheistic religion",
        source_citation=Citation(title="Test Doc", source_uri="test://doc"),
    )


class TestChatmlSchema:
    def test_renders_messages_key(self):
        sample = make_qa_sample()
        result = ChatmlSchema().render(sample)
        assert "messages" in result

    def test_renders_three_messages(self):
        sample = make_qa_sample()
        result = ChatmlSchema().render(sample)
        assert len(result["messages"]) == 3

    def test_message_roles(self):
        sample = make_qa_sample()
        result = ChatmlSchema().render(sample)
        roles = [m["role"] for m in result["messages"]]
        assert roles == ["system", "user", "assistant"]

    def test_user_message_is_question(self):
        sample = make_qa_sample(question="What is Iman?")
        result = ChatmlSchema().render(sample)
        user_msg = result["messages"][1]
        assert user_msg["content"] == "What is Iman?"

    def test_assistant_message_is_answer(self):
        sample = make_qa_sample(answer="Faith in Allah.")
        result = ChatmlSchema().render(sample)
        assistant_msg = result["messages"][2]
        assert assistant_msg["content"] == "Faith in Allah."

    def test_default_system_prompt(self):
        sample = make_qa_sample()
        result = ChatmlSchema().render(sample)
        system_msg = result["messages"][0]
        assert system_msg["content"] == "You are a helpful assistant."

    def test_custom_system_prompt(self):
        sample = make_qa_sample()
        result = ChatmlSchema(system_prompt="You are an Islamic scholar.").render(sample)
        assert result["messages"][0]["content"] == "You are an Islamic scholar."

    def test_supports_qa(self):
        assert ChatmlSchema().supports("qa") is True

    def test_supports_instruction(self):
        assert ChatmlSchema().supports("instruction") is True

    def test_supports_summary(self):
        assert ChatmlSchema().supports("summary") is True

    def test_does_not_support_rag(self):
        assert ChatmlSchema().supports("rag") is False

    def test_name_attribute(self):
        assert ChatmlSchema.name == "chatml"


class TestQaSchema:
    def test_renders_question_and_answer(self):
        sample = make_qa_sample(question="Who was the Prophet?", answer="Muhammad (pbuh).")
        result = QaSchema().render(sample)
        assert result == {"question": "Who was the Prophet?", "answer": "Muhammad (pbuh)."}

    def test_supports_qa(self):
        assert QaSchema().supports("qa") is True

    def test_supports_instruction(self):
        assert QaSchema().supports("instruction") is True

    def test_does_not_support_rag(self):
        assert QaSchema().supports("rag") is False

    def test_name_attribute(self):
        assert QaSchema.name == "qa"


class TestSchemaRegistry:
    def test_get_chatml_from_registry(self):
        adapter = get_schema("chatml")
        assert adapter.name == "chatml"

    def test_get_qa_from_registry(self):
        adapter = get_schema("qa")
        assert adapter.name == "qa"

    def test_list_schemas_includes_both(self):
        names = list_schemas()
        assert "chatml" in names
        assert "qa" in names

    def test_get_missing_raises(self):
        with pytest.raises(KeyError):
            get_schema("nonexistent_schema_xyz")


# ---------------------------------------------------------------------------
# New fixtures for Slice B schemas
# ---------------------------------------------------------------------------


def make_instruction_sample() -> Sample:
    return Sample(
        id=uuid4().hex,
        chunk_id="chk_test_0001",
        dataset_type=DatasetType.instruction,
        payload={
            "instruction": "Explain the concept.",
            "input": "",
            "output": "The concept means X.",
        },
        source_text="The concept means X.",
        supporting_quote="concept means X",
        source_citation=Citation(title="Test Doc", source_uri="test://doc"),
    )


def make_summary_sample() -> Sample:
    return Sample(
        id=uuid4().hex,
        chunk_id="chk_test_0002",
        dataset_type=DatasetType.summary,
        payload={
            "passage": "Long passage text.",
            "summary": "Short summary.",
            "key_points": ["p1", "p2"],
        },
        source_text="Long passage text.",
        supporting_quote="Long passage",
        source_citation=Citation(title="Test Doc", source_uri="test://doc"),
    )


def make_rag_sample() -> Sample:
    return Sample(
        id=uuid4().hex,
        chunk_id="chk_test_0003",
        dataset_type=DatasetType.rag,
        payload={"text": "Chunk text.", "title": "", "page": 1, "char_span": [0, 11]},
        source_text="Chunk text.",
        supporting_quote="Chunk text",
        source_citation=Citation(title="Test Doc", source_uri="test://doc"),
    )


# ---------------------------------------------------------------------------
# TestAlpacaSchema
# ---------------------------------------------------------------------------


class TestAlpacaSchema:
    def test_renders_qa_sample(self):
        sample = make_qa_sample()
        result = AlpacaSchema().render(sample)
        assert "instruction" in result
        assert "input" in result
        assert "output" in result

    def test_renders_instruction_sample(self):
        sample = make_instruction_sample()
        result = AlpacaSchema().render(sample)
        assert result["instruction"] == "Explain the concept."
        assert result["output"] == "The concept means X."

    def test_renders_summary_sample(self):
        sample = make_summary_sample()
        result = AlpacaSchema().render(sample)
        assert result["instruction"] == "Summarize the following passage."
        assert result["input"] == sample.payload["passage"]

    def test_supports_qa(self):
        assert AlpacaSchema().supports("qa") is True

    def test_supports_instruction(self):
        assert AlpacaSchema().supports("instruction") is True

    def test_supports_summary(self):
        assert AlpacaSchema().supports("summary") is True

    def test_does_not_support_rag(self):
        assert AlpacaSchema().supports("rag") is False

    def test_name_attribute(self):
        assert AlpacaSchema.name == "alpaca"

    def test_registered(self):
        assert get_schema("alpaca").name == "alpaca"


# ---------------------------------------------------------------------------
# TestShareGptSchema
# ---------------------------------------------------------------------------


class TestShareGptSchema:
    def test_renders_qa_sample(self):
        sample = make_qa_sample()
        result = ShareGptSchema().render(sample)
        assert "conversations" in result
        assert len(result["conversations"]) == 2
        assert result["conversations"][0]["from"] == "human"
        assert result["conversations"][1]["from"] == "gpt"

    def test_qa_human_is_question(self):
        sample = make_qa_sample(question="What is Iman?")
        result = ShareGptSchema().render(sample)
        assert result["conversations"][0]["value"] == "What is Iman?"

    def test_qa_gpt_is_answer(self):
        sample = make_qa_sample(answer="Faith in Allah.")
        result = ShareGptSchema().render(sample)
        assert result["conversations"][1]["value"] == "Faith in Allah."

    def test_renders_instruction_sample(self):
        sample = make_instruction_sample()
        result = ShareGptSchema().render(sample)
        # input is empty, so human value is just instruction
        assert result["conversations"][0]["value"] == "Explain the concept."

    def test_renders_instruction_with_input(self):
        sample = Sample(
            id=uuid4().hex,
            chunk_id="chk_test_inst_in",
            dataset_type=DatasetType.instruction,
            payload={"instruction": "Do this.", "input": "with context", "output": "Done."},
            source_text="Done.",
            supporting_quote="Done",
            source_citation=Citation(title="Test Doc", source_uri="test://doc"),
        )
        result = ShareGptSchema().render(sample)
        assert result["conversations"][0]["value"] == "Do this.\nwith context"

    def test_renders_summary_sample(self):
        sample = make_summary_sample()
        result = ShareGptSchema().render(sample)
        assert result["conversations"][0]["value"] == f"Summarize:\n{sample.payload['passage']}"
        assert result["conversations"][1]["value"] == sample.payload["summary"]

    def test_supports_qa(self):
        assert ShareGptSchema().supports("qa") is True

    def test_supports_instruction(self):
        assert ShareGptSchema().supports("instruction") is True

    def test_supports_summary(self):
        assert ShareGptSchema().supports("summary") is True

    def test_does_not_support_rag(self):
        assert ShareGptSchema().supports("rag") is False

    def test_name_attribute(self):
        assert ShareGptSchema.name == "sharegpt"

    def test_registered(self):
        assert get_schema("sharegpt").name == "sharegpt"


# ---------------------------------------------------------------------------
# TestCompletionSchema
# ---------------------------------------------------------------------------


class TestCompletionSchema:
    def test_renders_qa_sample(self):
        sample = make_qa_sample(question="What is Islam?", answer="A monotheistic religion.")
        result = CompletionSchema().render(sample)
        assert result == {
            "prompt": "What is Islam?\n\n",
            "completion": " A monotheistic religion.",
        }

    def test_renders_instruction_sample(self):
        sample = make_instruction_sample()
        result = CompletionSchema().render(sample)
        assert result == {
            "prompt": "Explain the concept.\n\n",
            "completion": " The concept means X.",
        }

    def test_renders_summary_sample(self):
        sample = make_summary_sample()
        result = CompletionSchema().render(sample)
        assert result == {
            "prompt": "Summarize:\nLong passage text.\n\n",
            "completion": " Short summary.",
        }

    def test_supports_qa(self):
        assert CompletionSchema().supports("qa") is True

    def test_supports_instruction(self):
        assert CompletionSchema().supports("instruction") is True

    def test_supports_summary(self):
        assert CompletionSchema().supports("summary") is True

    def test_does_not_support_rag(self):
        assert CompletionSchema().supports("rag") is False

    def test_name_attribute(self):
        assert CompletionSchema.name == "completion"

    def test_registered(self):
        assert get_schema("completion").name == "completion"


# ---------------------------------------------------------------------------
# TestRagSchema
# ---------------------------------------------------------------------------


class TestRagSchema:
    def test_renders_rag_sample(self):
        sample = make_rag_sample()
        result = RagSchema().render(sample)
        for key in ("id", "text", "title", "page", "source", "char_span"):
            assert key in result

    def test_id_is_chunk_id(self):
        sample = make_rag_sample()
        result = RagSchema().render(sample)
        assert result["id"] == sample.chunk_id

    def test_text_matches(self):
        sample = make_rag_sample()
        result = RagSchema().render(sample)
        assert result["text"] == sample.payload["text"]

    def test_source_is_uri(self):
        sample = make_rag_sample()
        result = RagSchema().render(sample)
        assert result["source"] == sample.source_citation.source_uri

    def test_supports_rag(self):
        assert RagSchema().supports("rag") is True

    def test_does_not_support_qa(self):
        assert RagSchema().supports("qa") is False

    def test_does_not_support_instruction(self):
        assert RagSchema().supports("instruction") is False

    def test_does_not_support_summary(self):
        assert RagSchema().supports("summary") is False

    def test_name_attribute(self):
        assert RagSchema.name == "rag"

    def test_registered(self):
        assert get_schema("rag").name == "rag"


# ---------------------------------------------------------------------------
# TestRawSchema
# ---------------------------------------------------------------------------


class TestRawSchema:
    def test_renders_qa_sample(self):
        sample = make_qa_sample()
        result = RawSchema().render(sample)
        for key in ("id", "chunk_id", "dataset_type", "payload"):
            assert key in result

    def test_dataset_type_is_string(self):
        sample = make_qa_sample()
        result = RawSchema().render(sample)
        assert isinstance(result["dataset_type"], str)

    def test_supports_all_types(self):
        schema = RawSchema()
        assert schema.supports("qa") is True
        assert schema.supports("instruction") is True
        assert schema.supports("summary") is True
        assert schema.supports("rag") is True

    def test_name_attribute(self):
        assert RawSchema.name == "raw"

    def test_registered(self):
        assert get_schema("raw").name == "raw"


# ---------------------------------------------------------------------------
# TestCompatibilityMatrix
# ---------------------------------------------------------------------------


class TestCompatibilityMatrix:
    def test_alpaca_does_not_support_rag(self):
        assert AlpacaSchema().supports("rag") is False

    def test_rag_schema_does_not_support_qa(self):
        assert get_schema("rag").supports("qa") is False

    def test_rag_schema_does_not_support_instruction(self):
        assert get_schema("rag").supports("instruction") is False

    def test_rag_schema_does_not_support_summary(self):
        assert get_schema("rag").supports("summary") is False

    def test_raw_supports_qa(self):
        assert get_schema("raw").supports("qa") is True

    def test_raw_supports_instruction(self):
        assert get_schema("raw").supports("instruction") is True

    def test_raw_supports_summary(self):
        assert get_schema("raw").supports("summary") is True

    def test_raw_supports_rag(self):
        assert get_schema("raw").supports("rag") is True
