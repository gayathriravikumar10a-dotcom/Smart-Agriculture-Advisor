
from rag.retriever import retrieve_agriculture_knowledge


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

    for topic, test in TEST_CASES.items():

        results = retrieve_agriculture_knowledge(
            test["question"],
            top_k=5,
        )

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
            for i, result in enumerate(results, start=1):
                print(f"\nRetrieved chunk {i}:")
                print(result["content"][:500])
                average = total_score / len(TEST_CASES)

    print("\n" + "=" * 55)
    print(f"Average keyword coverage: {average:.2f}%")
    print("LOCAL EVALUATION COMPLETED")


if __name__ == "__main__":
    evaluate()
