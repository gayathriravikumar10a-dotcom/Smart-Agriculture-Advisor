
import sqlite3
from pathlib import Path

from rag.retriever import retrieve_agriculture_knowledge
from agents.crop_recommendation_agent import generate_gemini_response


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DB = PROJECT_ROOT / "rag" / "answer_cache.sqlite3"


def get_cached_answer(question: str):
    """Return a previously generated valid answer, if available."""

    with sqlite3.connect(CACHE_DB) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS answer_cache (
                question TEXT PRIMARY KEY,
                answer TEXT NOT NULL
            )
            """
        )

        row = connection.execute(
            "SELECT answer FROM answer_cache WHERE question = ?",
            (question.strip().casefold(),),
        ).fetchone()

        if row:
            answer = row[0]
            if (
                not answer.strip()
                or "could not complete your request" in answer.lower()
                or "quota is exhausted" in answer.lower()
                or "temporarily unavailable" in answer.lower()
                or "analysis failed" in answer.lower()
            ):
                connection.execute(
                    "DELETE FROM answer_cache WHERE question = ?",
                    (question.strip().casefold(),),
                )
                return None
            return answer

    return None


def save_cached_answer(question: str, answer: str):
    """Save a successful answer for future use."""

    with sqlite3.connect(CACHE_DB) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS answer_cache (
                question TEXT PRIMARY KEY,
                answer TEXT NOT NULL
            )
            """
        )

        connection.execute(
            """
            INSERT OR REPLACE INTO answer_cache (question, answer)
            VALUES (?, ?)
            """,
            (question.strip().casefold(), answer),
        )


def generate_rag_answer(question: str) -> str:
    """Generate an answer grounded in retrieved agricultural documents."""

    if not isinstance(question, str) or not question.strip():
        return "Invalid input: Question cannot be empty."

    cached_answer = get_cached_answer(question)

    if cached_answer:
        print("Returning cached answer...")
        return cached_answer

    results = retrieve_agriculture_knowledge(
        query=question,
        top_k=3,
    )

    if not results:
        return (
            "No relevant agricultural documents were found. "
            "Please add reference PDFs to the knowledge base."
        )

    context_parts = []

    for index, result in enumerate(results, start=1):
        context_parts.append(
            f"Source {index}: {result['source']}\n"
            f"Content: {result['content']}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are an agricultural knowledge assistant.

Answer the farmer's question using the reference material below.

Rules:
1. Give clear, practical, farmer-friendly guidance.
2. Use only information supported by the provided context.
3. Do not invent fertilizer doses, pesticide doses, or statistics.
4. If the context does not contain enough information, say so.
5. Mention the source document names used.

REFERENCE MATERIAL:
{context}

FARMER'S QUESTION:
{question}

Provide the answer:
"""

    answer = generate_gemini_response(prompt)

    # Do not cache quota errors or failed responses.
    failure_terms = (
        "analysis failed",
        "quota is exhausted",
        "temporarily unavailable",
        "rate-limited",
    )

    if answer and not any(
        term in answer.casefold() for term in failure_terms
    ):
        save_cached_answer(question, answer)

    return answer


if __name__ == "__main__":
    question = "How can farmers improve soil fertility?"

    answer = generate_rag_answer(question)

    print("\nRAG AGRICULTURAL ANSWER")
    print("=" * 50)
    print(answer)
