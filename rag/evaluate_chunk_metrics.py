
from rag.retriever import retrieve_agriculture_knowledge


TEST_CASES = {
    "Soil fertility": {
        "question": "How can farmers improve soil fertility?",
        "keywords": ["soil fertility", "soil nutrients", "fertility levels"],
    },
    "Pest management": {
        "question": "How can farmers manage crop pests and diseases?",
        "keywords": ["pest management", "biological control", "crop diseases"],
    },
    "Irrigation": {
        "question": "What irrigation practices help conserve water?",
        "keywords": ["irrigation", "water stress", "water conservation"],
    },
    "Crop nutrients": {
        "question": "What nutrients are important for crop growth?",
        "keywords": ["nitrogen", "phosphorus", "potassium"],
    },
    "Soil moisture": {
        "question": "How can farmers conserve soil moisture?",
        "keywords": ["soil moisture", "mulching", "water shortage"],
    },
}


def evaluate():
    k = 5
    total_precision = 0
    total_recall = 0
    total_mrr = 0

    print("\nCHUNK-LEVEL RAG EVALUATION")
    print("=" * 55)

    for topic, test in TEST_CASES.items():

        results = retrieve_agriculture_knowledge(
            test["question"],
            top_k=k,
        )

        # Display crop nutrient chunks for inspection
        if topic == "Crop nutrients":
            for i, result in enumerate(results, start=1):
                print(f"\nChunk {i}:")
                print(result["content"][:400])

        relevant_keywords = test["keywords"]
        relevant_count = 0
        reciprocal_rank = 0

        matched_keywords = set()

        for rank, result in enumerate(results, start=1):
            content = result["content"].lower()

            matches = [
                keyword for keyword in relevant_keywords
                if keyword in content
            ]

            matched_keywords.update(matches)

            if matches:
                relevant_count += 1

                if reciprocal_rank == 0:
                    reciprocal_rank = 1 / rank

        precision = (
            relevant_count / len(results)
            if results else 0
        )
        recall = (
            len(matched_keywords) / len(relevant_keywords)
            if relevant_keywords else 0
        )

        total_precision += precision
        total_recall += recall
        total_mrr += reciprocal_rank

        print(f"\nTopic: {topic}")
        print(f"Relevant chunks: {relevant_count}/{k}")
        print(f"Precision@{k}: {precision:.2f}")
        print(f"Keyword Recall: {recall:.2f}")
        print(f"MRR: {reciprocal_rank:.2f}")

    count = len(TEST_CASES)

    print("\n" + "=" * 55)
    print(f"Mean Precision@{k}: {total_precision / count:.2f}")
    print(f"Mean Keyword Recall: {total_recall / count:.2f}")
    print(f"Mean MRR: {total_mrr / count:.2f}")
    print("\nCHUNK EVALUATION COMPLETED")


if __name__ == "__main__":
    evaluate()
