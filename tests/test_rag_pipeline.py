
from unittest.mock import patch

from rag.rag_pipeline import generate_rag_answer


def test_empty_question_is_rejected():
    result = generate_rag_answer("")

    assert result == "Invalid input: Question cannot be empty."


def test_no_documents_returns_helpful_message():
    with patch(
        "rag.rag_pipeline.retrieve_agriculture_knowledge",
        return_value=[],
    ):
        result = generate_rag_answer("How can I improve soil fertility?")

    assert "No relevant agricultural documents" in result


def test_retrieved_context_is_sent_to_gemini():
    mock_results = [
        {
            "source": "save_and_grow.pdf",
            "content": "Soil fertility depends on available nutrients.",
            "distance": 0.5,
        }
    ]

    with (
        patch(
            "rag.rag_pipeline.retrieve_agriculture_knowledge",
            return_value=mock_results,
        ),
        patch(
            "rag.rag_pipeline.generate_gemini_response",
            return_value="Improve soil fertility using suitable nutrient management.",
        ) as mock_gemini,
    ):
        result = generate_rag_answer("How can I improve soil fertility?")

    assert "Improve soil fertility" in result

    prompt = mock_gemini.call_args[0][0]

    assert "save_and_grow.pdf" in prompt
    assert "available nutrients" in prompt
    assert "How can I improve soil fertility?" in prompt
