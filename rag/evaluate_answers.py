
from rag.rag_pipeline import generate_rag_answer


TEST_QUESTIONS = [
    "How can farmers improve soil fertility?",
    "How can farmers manage crop pests and diseases?",
    "What irrigation practices help conserve water?",
    "What nutrients are important for crop growth?",
    "How can farmers conserve soil moisture?",
]


def evaluate_answers():
    print("\nRAG ANSWER QUALITY EVALUATION")
    print("=" * 60)

    for index, question in enumerate(TEST_QUESTIONS, start=1):
        print(f"\nTEST {index}: {question}")
        print("-" * 60)

        try:
            answer = generate_rag_answer(question)
            print("\nGENERATED ANSWER:")
            print(answer)

        except Exception as error:
            print("\nAPI REQUEST FAILED")
            print("Error:", str(error)[:300])
            print("Skipping this question.")

    print("\nEVALUATION COMPLETED")


if __name__ == "__main__":
    evaluate_answers()
