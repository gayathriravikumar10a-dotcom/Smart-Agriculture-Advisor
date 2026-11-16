
from rag.retriever import retrieve_multiple_agriculture_knowledge


TEST_CASES = {
    "Soil fertility": {
        "question": "How can farmers improve soil fertility?",
        "keywords": ["soil", "fertility"],
    },
    "Pest management": {
        "question": "How can farmers manage crop pests and diseases?",
        "keywords": ["pest", "disease", "biological", "management"],
    },
    "Irrigation": {
        "question": "What irrigation practices help conserve water?",
        "keywords": ["irrigation", "water", "drip", "conservation"],
    },
    "Crop nutrients": {
        "question": "Nitrogen phosphorus potassium NPK essential plant nutrients fertilizer crop growth",
        "keywords": ["nitrogen", "phosphorus", "potassium", "nutrients"],
    },
    "Soil moisture": {
        "question": "How can farmers conserve soil moisture?",
        "keywords": ["soil", "moisture", "mulching", "evaporation"],
    },
}


def evaluate():
    total_score = 0

    print("\nLOCAL RAG EVALUATION")
    print("=" * 55)

    topics = list(TEST_CASES.keys())
    questions = [
        TEST_CASES[topic]["question"]
        for topic in topics
    ]

    # Embed all five questions in one request.
    all_results = retrieve_multiple_agriculture_knowledge(
        questions,
        top_k=5,
    )

    for topic, batch_result in zip(topics, all_results):
        test = TEST_CASES[topic]

        # Extract the actual retrieved passages.
        results = batch_result.get("results", [])
        combined_text = " ".join(
            result["content"] for result in results
        ).lower()

        matched = [
            word for word in test["keywords"]
            if word in combined_text
        ]

        score = (
            len(matched) / len(test["keywords"])
        ) * 100

        total_score += score

        print(f"\nTopic: {topic}")
        print(f"Keyword coverage: {score:.2f}%")
        print(f"Matched: {matched}")
        print(f"Results retrieved: {len(results)}")

        if results:
            print("Top retrieved source:", results[0]["source"])

    average = total_score / len(TEST_CASES)

    print("\n" + "=" * 55)
    print(f"Average keyword coverage: {average:.2f}%")
    print("LOCAL EVALUATION COMPLETED")


if __name__ == "__main__":
    evaluate()
