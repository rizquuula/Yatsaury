"""Tests for schema adapters: ChatmlSchema and QaSchema."""
from __future__ import annotations

from uuid import uuid4

import pytest

from yatsaury.models import Citation, DatasetType, Sample
from yatsaury.schemas.base import get_schema, list_schemas
from yatsaury.schemas.chatml import ChatmlSchema
from yatsaury.schemas.qa import QaSchema


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
