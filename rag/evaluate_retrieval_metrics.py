
from rag.retriever import retrieve_agriculture_knowledge


TEST_CASES = {
    "Soil fertility": {
        "question": "How can farmers improve soil fertility?",
        "relevant_sources": ["save_and_grow.pdf"],
    },
    "Pest management": {
        "question": "How can farmers manage crop pests and diseases?",
        "relevant_sources": ["integrated_pest_management.pdf"],
    },
    "Irrigation": {
        "question": "What irrigation practices help conserve water?",
        "relevant_sources": ["save_and_grow.pdf"],
    },
    "Crop nutrients": {
        "question": "What nutrients are important for crop growth?",
        "relevant_sources": ["save_and_grow.pdf"],
    },
    "Soil moisture": {
        "question": "How can farmers conserve soil moisture?",
        "relevant_sources": ["save_and_grow.pdf"],
    },
}


def evaluate():
    k = 5
    total_precision = 0
    total_recall = 0
    total_reciprocal_rank = 0

    print("\nRAG RETRIEVAL METRICS")
    print("=" * 55)

    for topic, test in TEST_CASES.items():

        results = retrieve_agriculture_knowledge(
            test["question"],
            top_k=k,
        )

        relevant_sources = set(test["relevant_sources"])

        retrieved_sources = [
            result["source"] for result in results
        ]

        relevant_found = sum(
            1 for source in retrieved_sources
            if source in relevant_sources
        )

        precision = (
            relevant_found / len(retrieved_sources)
            if retrieved_sources else 0
        )

        recall = (
            relevant_found / len(relevant_sources)
            if relevant_sources else 0
        )

        reciprocal_rank = 0

        for rank, source in enumerate(retrieved_sources, start=1):
            if source in relevant_sources:
                reciprocal_rank = 1 / rank
                break

        total_precision += precision
        total_recall += min(recall, 1)
        total_reciprocal_rank += reciprocal_rank

        print(f"\nTopic: {topic}")
        print(f"Precision@{k}: {precision:.2f}")
        print(f"Recall@{k}: {min(recall, 1):.2f}")
        print(f"Reciprocal Rank: {reciprocal_rank:.2f}")
        print(f"Retrieved: {retrieved_sources}")

    count = len(TEST_CASES)

    print("\n" + "=" * 55)
    print(f"Mean Precision@{k}: {total_precision / count:.2f}")
    print(f"Mean Recall@{k}: {total_recall / count:.2f}")
    print(f"Mean Reciprocal Rank (MRR): {total_reciprocal_rank / count:.2f}")
    print("\nEVALUATION COMPLETED")


if __name__ == "__main__":
    evaluate()
