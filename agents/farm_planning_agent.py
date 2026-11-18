
from agents.crop_recommendation_agent import generate_gemini_response
from rag.retriever import retrieve_agriculture_knowledge


def shorten_report(report: str, max_chars: int = 1800) -> str:
    """Limit report size before combining agent outputs."""
    if not isinstance(report, str):
        return "Report unavailable."

    report = report.strip()

    if len(report) > max_chars:
        return report[:max_chars] + "\n[Report shortened]"

    return report


def create_farm_plan(
    location: str,
    intended_crop: str,
    crop_recommendation: str,
    soil_health_report: str,
    pest_disease_report: str,
    weather_irrigation_report: str,
) -> str:
    """Combine agent outputs into a practical farm plan."""

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

    for name, report in reports.items():
        if not isinstance(report, str) or not report.strip():
            return f"Invalid input: {name} report is missing."

    # Retrieve only two relevant reference chunks.
    question = (
        f"Practical farm management, soil health, pest prevention "
        f"and irrigation for {intended_crop} in {location}."
    )

    try:
        results = retrieve_agriculture_knowledge(
            query=question,
            top_k=2,
        )
    except Exception as error:
        print(f"Farm planning retrieval warning: {error}")
        results = []

    context_parts = []

    for index, result in enumerate(results, start=1):
        source = str(result.get("source", "Agricultural reference"))
        content = str(result.get("content", ""))[:1000]

        if content:
            context_parts.append(
                f"Source {index}: {source}\n{content}"
            )

    agricultural_context = (
        "\n\n".join(context_parts)
        if context_parts
        else "No additional references retrieved."
    )

    # Keep the combined prompt within a smaller size.
    prompt = f"""
You are an expert agricultural farm planning advisor
for Indian farmers.

LOCATION: {location}
INTENDED CROP: {intended_crop}

CROP RECOMMENDATION:
{shorten_report(crop_recommendation)}

SOIL HEALTH:
{shorten_report(soil_health_report)}

PEST AND DISEASE:
{shorten_report(pest_disease_report)}

WEATHER AND IRRIGATION:
{shorten_report(weather_irrigation_report)}

AGRICULTURAL REFERENCES:
{agricultural_context}

Prepare a practical farm plan with these eight sections:

1. FARM SITUATION
Summarize the farm conditions and missing information.

2. BEFORE PLANTING
List important preparation checks.

3. SOIL AND NUTRIENT MANAGEMENT
Give practical soil improvement steps.

4. CROP MONITORING
Explain how to monitor crop health and pests.

5. WATER MANAGEMENT
Give safe irrigation guidance.

6. KEY RISKS
Identify major risks and prevention methods.

7. ACTION CHECKLIST
Give a prioritized checklist of farmer actions.

8. LOCAL SUPPORT
Recommend Raitha Samparka Kendra, KVK,
or the local Agriculture Officer in Karnataka.

RULES:
- Use the four agent reports as the main information.
- Use references only when relevant.
- Do not invent weather, prices, soil test results or yields.
- Do not prescribe unsupported fertilizer or pesticide quantities.
- Clearly mention uncertainty.
- Use simple, practical language.
- Keep the complete plan concise, around 700 words maximum.
"""

    response = generate_gemini_response(prompt)

    if not response or "temporarily unavailable" in response.lower():
        return (
            "Farm planning could not be completed because the AI service "
            "is temporarily unavailable. Please try again."
        )

    return response
