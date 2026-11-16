
import json
import math
import re
import sqlite3
from pathlib import Path

from rag.embedding_utils import generate_embeddings

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATABASE_PATH = (
    PROJECT_ROOT / "rag" / "vector_store" / "agriculture_knowledge.sqlite3"
)

MIN_SCORE = 0.35

STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were",
    "what", "how", "why", "when", "where", "which",
    "can", "could", "should", "would", "do", "does",
    "did", "of", "for", "to", "in", "on", "with",
    "and", "or", "by", "from", "about", "be", "it",
    "their", "they", "farmers", "farmer"
}


def tokenize(text):
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return {
        word for word in words
        if word not in STOP_WORDS and len(word) > 2
    }


def cosine_similarity(vector_a, vector_b):
    if len(vector_a) != len(vector_b):
        raise ValueError("Embedding dimensions do not match.")

    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
    norm_a = math.sqrt(sum(a * a for a in vector_a))
    norm_b = math.sqrt(sum(b * b for b in vector_b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def load_document_chunks():
    if not DATABASE_PATH.exists():
        raise FileNotFoundError(
            "Knowledge database not found. Run python -m rag.ingest first."
        )

    with sqlite3.connect(DATABASE_PATH) as connection:
        rows = connection.execute("""
            SELECT content, source, page, embedding
            FROM chunks
        """).fetchall()

    return [
        {
            "content": content,
            "source": source,
            "page": page,
            "embedding": json.loads(embedding),
        }
        for content, source, page, embedding in rows
    ]


def rank_chunks(query, query_embedding, chunks, top_k):
    query_words = tokenize(query)
    query_lower = query.lower()

    results = []

    for chunk in chunks:
        content = chunk["content"]
        content_lower = content.lower()
        content_words = tokenize(content)

        semantic_score = cosine_similarity(
            query_embedding,
            chunk["embedding"],
        )

        matched_words = query_words & content_words
        keyword_score = (
            len(matched_words) / len(query_words)
            if query_words else 0.0
        )

        topic_boost = 0.0
        if "crop rotation" in query_lower:
            if "crop rotation" in content_lower:
                topic_boost += 0.10

            benefit_terms = [
                "soil fertility",
                "disease",
                "pest",
                "nematode",
                "weed",
                "nitrogen",
                "sustainability",
                "soil health",
            ]
            matched_benefits = [
                term for term in benefit_terms
                if term in content_lower
            ]
            topic_boost += min(len(matched_benefits) * 0.05, 0.20)

        combined_score = (
            0.55 * semantic_score
            + 0.35 * keyword_score
            + topic_boost
        )

        results.append({
            "content": content,
            "source": chunk["source"],
            "page": chunk["page"],
            "similarity": semantic_score,
            "keyword_score": keyword_score,
            "topic_boost": topic_boost,
            "score": combined_score,
            "distance": 1 - semantic_score,
        })

    results.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    relevant_results = [
        item for item in results
        if item["score"] >= MIN_SCORE
    ]

    return relevant_results[:max(1, top_k)]


def retrieve_agriculture_knowledge(query: str, top_k: int = 3):
    if not isinstance(query, str) or not query.strip():
        return []

    chunks = load_document_chunks()

    if not chunks:
        return []

    query_embedding = generate_embeddings(
        [query.strip()],
        task_type="RETRIEVAL_QUERY",
    )[0]

    return rank_chunks(query, query_embedding, chunks, top_k)


def retrieve_multiple_agriculture_knowledge(
    queries: list[str],
    top_k: int = 5,
):
    valid_queries = [
        query.strip()
        for query in queries
        if isinstance(query, str) and query.strip()
    ]

    if not valid_queries:
        return []

    chunks = load_document_chunks()

    if not chunks:
        return [
            {"query": query, "results": []}
            for query in valid_queries
        ]

    query_embeddings = generate_embeddings(
        valid_queries,
        task_type="RETRIEVAL_QUERY",
    )

    all_results = []

    for query, query_embedding in zip(valid_queries, query_embeddings):
        results = rank_chunks(
            query,
            query_embedding,
            chunks,
            top_k,
        )

        all_results.append({
            "query": query,
            "results": results,
        })

    return all_results
