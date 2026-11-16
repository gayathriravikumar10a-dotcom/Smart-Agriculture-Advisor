
from agents.crop_recommendation_agent import (
    client,
    generate_gemini_response,
)

from rag.retriever import retrieve_agriculture_knowledge


def analyze_weather_irrigation(
    crop: str,
    location: str,
    soil_type: str,
    water_availability: str,
    growth_stage: str = "Unknown",
    weather_information: str = "Not provided",
) -> str:
    """Provide weather and irrigation guidance using agricultural RAG."""

    # Validate inputs
    if not isinstance(crop, str) or not crop.strip():
        return "Invalid input: Crop cannot be empty."

    if not isinstance(location, str) or not location.strip():
        return "Invalid input: Location cannot be empty."

    if not isinstance(soil_type, str) or not soil_type.strip():
        return "Invalid input: Soil type cannot be empty."

    if (
        not isinstance(water_availability, str)
        or not water_availability.strip()
    ):
        return "Invalid input: Water availability cannot be empty."

    if not isinstance(growth_stage, str) or not growth_stage.strip():
        return "Invalid input: Growth stage cannot be empty."

    if (
        not isinstance(weather_information, str)
        or not weather_information.strip()
    ):
        return "Invalid input: Weather information cannot be empty."

    # Retrieve agricultural knowledge
    question = (
        f"Irrigation requirements, water management, soil moisture, "
        f"rainfall effects, and water conservation for {crop}. "
        f"Soil type: {soil_type}. "
        f"Growth stage: {growth_stage}. "
        f"Water availability: {water_availability}. "
        f"Location: {location}."
    )

    try:
        results = retrieve_agriculture_knowledge(
            query=question,
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

    # Prepare prompt
    prompt = f"""
You are the Weather and Irrigation Agent in a Smart Agriculture Advisor.

FARM INFORMATION:
- Crop: {crop}
- Location: {location}
- Soil type: {soil_type}
- Water availability: {water_availability}
- Growth stage: {growth_stage}
- Weather information supplied by the user: {weather_information}

AGRICULTURAL REFERENCE MATERIAL:
{agricultural_context}

Provide a practical report in simple farmer-friendly language.

Include:
1. General irrigation considerations for the crop and growth stage.
2. How soil type may affect water retention and irrigation.
3. How rainfall, temperature, and wind can affect water needs.
4. Signs that may indicate overwatering or underwatering.
5. Water conservation practices suitable for consideration.
6. What local weather information the farmer should check next.

IMPORTANT RULES:
- Use relevant agricultural reference material when available.
- Do not force unrelated reference information into the answer.
- Do not claim to have retrieved live weather data.
- Treat weather information as unverified user-provided information.
- Do not invent rainfall, temperature, or forecast values.
- Do not prescribe a rigid irrigation schedule without enough data.
- Explain that water requirements vary with crop variety, soil,
  growth stage, rainfall, and local conditions.
- Mention reference document names when relevant.
- Recommend checking the latest official local weather forecast.
- For Karnataka, farmers can contact RSK, KVK, or Agriculture Officers.
- Use clear, practical language.
"""

    # Generate report
    response = generate_gemini_response(prompt)

    if "quota is exhausted" in response.lower():
        return (
            "Gemini API quota is currently exhausted. "
            "Please try again when quota becomes available."
        )

    return response


if __name__ == "__main__":
    report = analyze_weather_irrigation(
        crop="Maize",
        location="Karnataka, India",
        soil_type="Loamy",
        water_availability="Moderate irrigation",
        growth_stage="Vegetative stage",
        weather_information="No current forecast available",
    )

    print("\nWEATHER AND IRRIGATION REPORT")
    print("=" * 40)
    print(report)

