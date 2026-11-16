
from agents.crop_recommendation_agent import (
    client,
    generate_gemini_response,
)

from rag.retriever import retrieve_agriculture_knowledge


def create_farm_plan(
    location: str,
    intended_crop: str,
    crop_recommendation: str,
    soil_health_report: str,
    pest_disease_report: str,
    weather_irrigation_report: str,
) -> str:
    """Combine five-agent outputs and agricultural references into a farm plan."""

    # Validate inputs
    if not isinstance(location, str) or not location.strip():
        return "Invalid input: Location cannot be empty."

    if not isinstance(intended_crop, str) or not intended_crop.strip():
        return "Invalid input: Intended crop cannot be empty."

    reports = {
        "Crop recommendation": crop_recommendation,
        "Soil health": soil_health_report,
        "Pest and disease": pest_disease_report,
        "Weather and irrigation": weather_irrigation_report,
    }

    for report_name, report in reports.items():
        if not isinstance(report, str) or not report.strip():
            return f"Invalid input: {report_name} report is missing."

    # Retrieve agricultural knowledge
    question = (
        f"Farm planning, crop management, soil health, pest prevention, "
        f"irrigation, and agricultural risk management for {intended_crop} "
        f"in {location}."
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
        else "No additional agricultural reference documents were retrieved."
    )

    # Prepare prompt
    prompt = f"""
You are the Farm Planning and Risk Management Agent in a
Smart Agriculture Advisor.

FARM LOCATION: {location}
FARMER'S INTENDED CROP: {intended_crop}

REPORT FROM AGENT 1 — CROP RECOMMENDATION
{crop_recommendation}

REPORT FROM AGENT 2 — SOIL HEALTH AND NUTRIENTS
{soil_health_report}

REPORT FROM AGENT 3 — PEST AND PLANT DISEASE
{pest_disease_report}

REPORT FROM AGENT 4 — WEATHER AND IRRIGATION
{weather_irrigation_report}

ADDITIONAL AGRICULTURAL REFERENCE MATERIAL:
{agricultural_context}

Create a practical farm plan using these sections:

1. FARM SITUATION
Summarize available information and identify missing details.

2. BEFORE PLANTING
List checks before purchasing seeds or preparing land.

3. SOIL AND NUTRIENT MANAGEMENT
Explain sensible next steps based on the soil report.

4. CROP MONITORING
Suggest practical ways to monitor crop health and pest symptoms.

5. WATER MANAGEMENT
Explain what to check before deciding when to irrigate.

6. KEY RISKS
List risks, why they matter, and ways to reduce them.

7. ACTION CHECKLIST
Provide a short, prioritized checklist.

8. LOCAL SUPPORT
For Karnataka, recommend the local Raitha Samparka Kendra (RSK),
Krishi Vigyan Kendra (KVK), or Agriculture Officer.

IMPORTANT RULES:
- Use the four agent reports as the main farm-specific information.
- Use relevant agricultural reference material when available.
- Do not force unrelated reference information into the plan.
- Do not invent weather forecasts, soil-test values, or market prices.
- Do not guarantee crop yield or profit.
- Do not prescribe unsupported fertilizer or pesticide quantities.
- Clearly identify uncertainty and missing information.
- Mention reference document names when relevant.
- Recommend local agricultural expertise when necessary.
- Use clear, practical language suitable for farmers.
"""

    # Generate the combined farm plan
    response = generate_gemini_response(prompt)

    if "quota is exhausted" in response.lower():
        return (
            "Gemini API quota is currently exhausted. "
            "Please try again when quota becomes available."
        )

    return response


if __name__ == "__main__":
    plan = create_farm_plan(
        location="Karnataka, India",
        intended_crop="Maize",
        crop_recommendation="Maize is a possible crop for the supplied conditions.",
        soil_health_report="Loamy soil with pH 6.5. Soil testing is recommended.",
        pest_disease_report="Monitor yellow leaves and inspect plants for pests and disease.",
        weather_irrigation_report="Moderate irrigation is available. Check local rainfall forecasts.",
    )

    print("\nFARM PLANNING AND RISK MANAGEMENT REPORT")
    print("=" * 50)
    print(plan)
