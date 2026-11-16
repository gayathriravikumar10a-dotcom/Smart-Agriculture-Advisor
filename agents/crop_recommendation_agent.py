
from pathlib import Path
import os
import time

from dotenv import load_dotenv
from google import genai


# Load environment variables
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found. Check your project's .env file."
    )

client = genai.Client(api_key=API_KEY)

MODEL_NAME = "gemini-3.7-flash"


def generate_gemini_response(prompt: str) -> str:
    """
    Shared Gemini request handler.

    Handles quota errors immediately and retries temporary
    service errors only once.
    """

    max_attempts = 2

    for attempt in range(max_attempts):
        try:
            print("Contacting Gemini...")

            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
            )

            if response.text:
                return response.text

            return "Gemini returned an empty response."

        except Exception as error:
            error_message = str(error).lower()

            # Handle quota exhaustion FIRST.
            # Do not retry when the daily quota is exhausted.
            quota_exhausted = (
                "resource_exhausted" in error_message
                or "quota exceeded" in error_message
                or "daily limit" in error_message
                or "per day" in error_message
            )

            if quota_exhausted:
                print("Gemini API quota is exhausted.")
                return (
                    "Analysis failed: Gemini API quota is exhausted. "
                    "Please try again after the quota resets."
                )

            # Handle temporary service errors.
            temporary_error = (
                "503" in error_message
                or "429" in error_message
                or "500" in error_message
                or "502" in error_message
                or "service unavailable" in error_message
            )

            if temporary_error and attempt < max_attempts - 1:
                print("Temporary Gemini error. Retrying once in 3 seconds...")
                time.sleep(3)

            elif temporary_error:
                print("Gemini is currently unavailable.")
                return (
                    "Analysis failed: Gemini is temporarily unavailable "
                    "or rate-limited. Please try again later."
                )

            else:
                print(f"Gemini request failed: {error}")
                return (
                    "Analysis failed: The Gemini request failed. "
                    f"Error: {error}"
                )

    return "Analysis failed: Unable to generate a response."


def recommend_crops(
    soil_type: str,
    soil_ph: float,
    water_availability: str,
    season: str,
    location: str,
) -> str:
    """Recommend potentially suitable crops for a farm."""

    # Validate soil type
    if not isinstance(soil_type, str) or not soil_type.strip():
        return "Invalid input: Soil type cannot be empty."

    # Validate soil pH
    if (
        isinstance(soil_ph, bool)
        or not isinstance(soil_ph, (int, float))
        or not 0 <= soil_ph <= 14
    ):
        return "Invalid input: Soil pH must be between 0 and 14."

    # Validate water availability
    if (
        not isinstance(water_availability, str)
        or not water_availability.strip()
    ):
        return "Invalid input: Water availability cannot be empty."

    # Validate season
    if not isinstance(season, str) or not season.strip():
        return "Invalid input: Season cannot be empty."

    # Validate location
    if not isinstance(location, str) or not location.strip():
        return "Invalid input: Location cannot be empty."

    prompt = f"""
You are an agricultural crop recommendation expert.

FARM DETAILS:
Location: {location}
Soil type: {soil_type}
Soil pH: {soil_ph}
Water availability: {water_availability}
Season: {season}

TASK:
1. Suggest potentially suitable crops for these conditions.
2. Explain why each crop may be suitable.
3. Mention important soil and water considerations.
4. Do not invent local weather or soil-test results.
5. Do not prescribe exact fertilizer quantities without
   sufficient soil-test information.

LOCAL AGRICULTURAL SUPPORT IN KARNATAKA:
- Raitha Samparka Kendra (RSK): Local farming advice and
  seed availability information.
- Krishi Vigyan Kendra (KVK): Soil-test interpretation and
  agricultural guidance.
- Department of Agriculture officers: Local crop variety
  recommendations.

NEXT STEP:
Recommend confirming crop and seed variety suitability with
local agricultural experts before purchasing seeds or preparing
the land.

Present the answer in clear sections and simple language.
"""

    return generate_gemini_response(prompt)


if __name__ == "__main__":
    report = recommend_crops(
        soil_type="Loamy",
        soil_ph=6.5,
        water_availability="Moderate irrigation",
        season="Kharif",
        location="Karnataka, India",
    )

    print("\nCROP RECOMMENDATION REPORT")
    print("=" * 35)
    print(report)
