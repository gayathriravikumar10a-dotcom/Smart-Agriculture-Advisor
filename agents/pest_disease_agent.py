
from agents.crop_recommendation_agent import (
    client,
    generate_gemini_response,
)

from rag.retriever import retrieve_agriculture_knowledge


def analyze_pest_disease(
    crop: str,
    symptoms: str,
    location: str,
    growth_stage: str = "Unknown",
) -> str:
    """Analyze crop symptoms using agricultural knowledge and Gemini."""

    # Validate crop
    if not isinstance(crop, str) or not crop.strip():
        return "Invalid input: Crop cannot be empty."

    # Validate symptoms
    if not isinstance(symptoms, str) or not symptoms.strip():
        return "Invalid input: Symptoms cannot be empty."

    # Validate location
    if not isinstance(location, str) or not location.strip():
        return "Invalid input: Location cannot be empty."

    # Validate growth stage
    if not isinstance(growth_stage, str) or not growth_stage.strip():
        growth_stage = "Unknown"

    # Retrieve agricultural reference information
    question = (
        f"Common pests, plant diseases, symptoms, prevention, "
        f"and integrated pest management for {crop}. "
        f"Observed symptoms: {symptoms}. "
        f"Growth stage: {growth_stage}. "
        f"Location: {location}."
    )

    try:
        results = retrieve_agriculture_knowledge(
            question=question,
            top_k=3,
        )
    except Exception:
        results = []

    # Prepare reference context
    context_parts = []

    for index, result in enumerate(results, start=1):
        context_parts.append(
            f"Source {index}: {result['source']}\n"
            f"Content: {result['content']}"
        )

    agricultural_context = (
        "\n\n".join(context_parts)
        if context_parts
        else "No relevant agricultural reference documents were retrieved."
    )

    prompt = f"""
You are an experienced agricultural pest and plant disease expert.

FARM DETAILS:
Crop: {crop}
Location: {location}
Growth stage: {growth_stage}
Observed symptoms: {symptoms}

AGRICULTURAL REFERENCE MATERIAL:
{agricultural_context}

TASK:
1. Identify possible pests or plant diseases.
2. Explain the likely causes of the symptoms.
3. Suggest safe prevention and management methods.
4. Recommend integrated pest management (IPM).
5. Explain when a farmer should consult an agricultural expert.

IMPORTANT RULES:
- Use relevant reference material when available.
- Do not force unrelated reference information into the answer.
- Do not claim a confirmed diagnosis from symptoms alone.
- Do not recommend unsafe pesticide use or unsupported dosages.
- Do not invent agricultural facts or treatment quantities.
- Mention the reference document names when relevant.
- Recommend local agricultural experts for field-level confirmation.
- For Karnataka, mention Raitha Samparka Kendra (RSK),
  Krishi Vigyan Kendra (KVK), or Department of Agriculture officers.

Present the report in simple, farmer-friendly language.
Use clear headings and practical suggestions.
"""

    return generate_gemini_response(prompt)


if __name__ == "__main__":
    report = analyze_pest_disease(
        crop="Maize",
        symptoms="Yellow leaves",
        location="Karnataka, India",
        growth_stage="Vegetative",
    )

    print("\nPEST AND DISEASE REPORT")
    print("=" * 35)
    print(report)
