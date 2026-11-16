from rag.retriever import retrieve_agriculture_knowledge

questions = [
    "How can farmers improve soil fertility?",
    "Integrated pest management, insect pests, crop diseases, fungal infections, biological control and pest prevention in agriculture",
    "What irrigation practices help conserve water?",
    "What nutrients are important for crop growth?",
    "How can farmers conserve soil moisture?",
]

for question in questions:
    results = retrieve_agriculture_knowledge(question, top_k=3)

    if results:
        print("\nPASSED:", question)
        print("Best distance:", results[0]["distance"])
    else:
        print("\nFAILED:", question)