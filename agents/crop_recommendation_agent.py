
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env")

client = genai.Client(
    api_key=API_KEY,
    http_options={"timeout": 30000},
)
MODEL_NAME = "gemini-3.8-flash"


def generate_gemini_response(prompt: str) -> str:
    """
    Shared Gemini request handler.
    Handles quota exhaustion and temporary service errors.
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

            return "The AI service returned an empty response. Please try again."

        except Exception as error:
            error_message = str(error).lower()

            quota_exhausted = (
                "resource_exhausted" in error_message
                or "quota exceeded" in error_message
                or "daily limit" in error_message
                or "per day" in error_message
            )

            if quota_exhausted:
                print("Gemini daily quota exhausted.")

                return (
                    "Our AI agricultural advisory service has temporarily "
                    "reached its daily usage limit. Your farm information "
                    "and knowledge base are safe. Please try again after "
                    "the API quota resets."
                )

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
                print(f"Gemini failed after retry: {error}")
                return (
                    "The AI service is temporarily unavailable. "
                    "Please try again later."
                )

            else:
                print(f"Gemini ERROR TYPE: {type(error).__name__}")
                print(f"Gemini ERROR DETAILS: {error}")
                return (
                    "The AI service could not complete your request. "
                    "Please try again later."
                )

    return "Unable to generate an agricultural recommendation."


def recommend_crops(
    soil_type: str,
    soil_ph: float,
    water_availability: str,
    season: str,
    location: str,
) -> str:
    """Generate crop recommendations using Gemini."""

    prompt = f"""
    You are an agricultural crop recommendation expert.

    Recommend suitable crops based on the following farm details:
    Location: {location}
    Soil type: {soil_type}
    Soil pH: {soil_ph}
    Water availability: {water_availability}
    Growing season: {season}

    Provide:
    1. Three to five suitable crop recommendations.
    2. Reasons why each crop is suitable.
    3. Basic soil and water requirements.
    4. Important cultivation precautions.

    Give practical advice suitable for Indian farmers.
    Do not invent local weather conditions or guarantee crop yields.
    """

    return generate_gemini_response(prompt)
