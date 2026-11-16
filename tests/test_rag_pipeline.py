
import sqlite3
from unittest.mock import patch

from rag.rag_pipeline import (
    generate_rag_answer,
    get_cached_answer,
    save_cached_answer,
)


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


def test_get_cached_answer_returns_a_valid_answer():
    connection = sqlite3.connect(":memory:")
    try:
        with patch("rag.rag_pipeline.sqlite3.connect", return_value=connection):
            save_cached_answer(" Soil health? ", "Use cover crops.")

            assert get_cached_answer("SOIL HEALTH?") == "Use cover crops."
    finally:
        connection.close()


def test_get_cached_answer_deletes_invalid_answers():
    invalid_answers = [
        "",
        "The AI could not complete your request.",
        "Quota is exhausted.",
        "Temporarily unavailable.",
        "Analysis failed.",
    ]

    for answer in invalid_answers:
        connection = sqlite3.connect(":memory:")
        try:
            with patch(
                "rag.rag_pipeline.sqlite3.connect",
                return_value=connection,
            ):
                save_cached_answer("soil health", answer)

                assert get_cached_answer("soil health") is None
                cached_rows = connection.execute(
                    "SELECT COUNT(*) FROM answer_cache"
                ).fetchone()[0]

                assert cached_rows == 0
        finally:
            connection.close()
