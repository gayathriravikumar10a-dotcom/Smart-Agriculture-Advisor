
from agents.crop_recommendation_agent import (
    client,
    generate_gemini_response,
)

from rag.retriever import retrieve_agriculture_knowledge


def analyze_soil(
    soil_type: str,
    soil_ph: float,
    crop: str,
    location: str,
) -> str:
    """Provide soil health guidance using RAG and Gemini."""

    # Validate inputs
    if not isinstance(soil_type, str) or not soil_type.strip():
        return "Invalid input: Soil type cannot be empty."

    if (
        isinstance(soil_ph, bool)
        or not isinstance(soil_ph, (int, float))
        or not 0 <= soil_ph <= 14
    ):
        return "Invalid input: Soil pH must be between 0 and 14."

    if not isinstance(crop, str) or not crop.strip():
        return "Invalid input: Crop cannot be empty."

    if not isinstance(location, str) or not location.strip():
        return "Invalid input: Location cannot be empty."

    # Retrieve relevant agricultural knowledge.
    question = (
        f"Soil fertility, nutrient management, organic matter, "
        f"and soil health for {crop} grown in {soil_type} soil "
        f"with pH {soil_ph}."
    )

    try:
        results = retrieve_agriculture_knowledge(
            query=question,
            top_k=3,
        )
    except Exception:
        results = []

    # Prepare reference context.
    if results:
        context_parts = []

        for index, result in enumerate(results, start=1):
            context_parts.append(
                f"Source {index}: {result['source']}\n"
                f"Information: {result['content']}"
            )

        agricultural_context = "\n\n".join(context_parts)
    else:
        agricultural_context = (
            "No relevant agricultural reference documents "
            "were available. Use general agricultural knowledge "
            "carefully and do not invent facts."
        )

    prompt = f"""
You are the Soil Health and Nutrient Agent in a Smart Agriculture Advisor.

Farm information:
- Location: {location}
- Soil type: {soil_type}
- Soil pH: {soil_ph}
- Intended crop: {crop}

AGRICULTURAL REFERENCE KNOWLEDGE:
{agricultural_context}

Explain the following in simple language:

1. What the reported soil type and pH may indicate.
2. Why soil pH alone cannot determine soil fertility.
3. Which soil-test measurements are useful, such as nitrogen,
   phosphorus, potassium, organic carbon, and electrical conductivity.
4. General ways to maintain soil health, such as suitable organic
   matter, crop rotation, and preventing soil erosion.
5. Why fertilizer and compost quantities depend on soil-test results,
   crop requirements, and local agricultural recommendations.

Important rules:
- Use the reference knowledge when it is relevant.
- Do not treat unrelated crop information as applicable.
- Do not invent soil-test results.
- Do not prescribe exact fertilizer or compost quantities without
  sufficient soil-test and crop-specific information.
- Do not guarantee crop yield.
- Recommend a qualified soil laboratory or local agricultural expert.
- For Karnataka, mention Raitha Samparka Kendra (RSK),
  Krishi Vigyan Kendra (KVK), or Department of Agriculture officers.
- Mention the reference document names when using their information.
- Keep advice practical and easy for a farmer to understand.
"""

    return generate_gemini_response(prompt)


if __name__ == "__main__":
    report = analyze_soil(
        soil_type="Loamy",
        soil_ph=6.5,
        crop="Maize",
        location="Karnataka, India",
    )

    print("\nSOIL HEALTH AND NUTRIENT REPORT")
    print("=" * 40)
    print(report)
