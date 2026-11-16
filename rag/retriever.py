
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
VECTOR_STORE_DIR = PROJECT_ROOT / "rag" / "vector_store"

COLLECTION_NAME = "agriculture_knowledge"

# ChromaDB uses the same embedding model as ingestion.
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(
    path=str(VECTOR_STORE_DIR)
)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME
)

# Initial distance threshold. We will tune it using test results.
MAX_DISTANCE = 1.0


def retrieve_agriculture_knowledge(
    question: str,
    top_k: int = 3,
) -> list[dict]:
    """Retrieve relevant agricultural knowledge chunks."""

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty.")

    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError("top_k must be a positive integer.")

    total_documents = collection.count()

    if total_documents == 0:
        return []

    query_embedding = embedding_model.encode(
        question.strip()
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, total_documents),
        include=["documents", "metadatas", "distances"],
    )

    retrieved_chunks = []

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances,
    ):
        if not document or distance is None:
            continue

        if distance > MAX_DISTANCE:
            continue

        retrieved_chunks.append(
            {
                "content": document,
                "source": (metadata or {}).get(
                    "source", "Unknown"
                ),
                "distance": distance,
            }
        )

    return retrieved_chunks


if __name__ == "__main__":
    query = "How can farmers improve soil fertility?"

    results = retrieve_agriculture_knowledge(
        query,
        top_k=5,
    )

    print("\nAGRICULTURAL KNOWLEDGE SEARCH")
    print("=" * 50)

    if not results:
        print("No sufficiently close matching chunks found.")

    for index, result in enumerate(results, start=1):
        print(f"\nResult {index}")
        print("Source:", result["source"])
        print("Distance:", result["distance"])
        print("Content:", result["content"][:700])
