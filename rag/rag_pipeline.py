
from rag.retriever import retrieve_agriculture_knowledge
from agents.crop_recommendation_agent import generate_gemini_response


def generate_rag_answer(question: str) -> str:
    """Generate an answer grounded in retrieved agricultural documents."""

    if not isinstance(question, str) or not question.strip():
        return "Invalid input: Question cannot be empty."

    results = retrieve_agriculture_knowledge(
        question=question,
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

    return generate_gemini_response(prompt)


if __name__ == "__main__":
    question = "How can farmers improve soil fertility?"

    answer = generate_rag_answer(question)

    print("\nRAG AGRICULTURAL ANSWER")
    print("=" * 50)
    print(answer)
