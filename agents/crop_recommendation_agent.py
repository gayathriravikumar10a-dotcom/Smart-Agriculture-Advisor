
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

API_KEY = os.getenv("GROQ_API_KEY")
if not API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env")

client = Groq(api_key=API_KEY)
MODEL_NAME = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")


def generate_groq_response(prompt: str) -> str:
    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert agricultural advisor. "
                            "Give clear, practical and safe advice "
                            "suitable for Indian farmers. "
                            "Do not invent facts or guarantee crop yields."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0.3,
                max_tokens=2048,
            )
            answer = response.choices[0].message.content
            if answer:
                return answer.strip()
            return "The AI service returned an empty response. Please try again."
        except Exception as error:
            status_code = getattr(error, "status_code", None)
            print(f"Groq error: {type(error).__name__}: {error}")
            if status_code == 401:
                return "Groq authentication failed. Please check your GROQ_API_KEY."
            if status_code == 429:
                return "Groq API rate limit reached. Please wait and try again."
            if status_code in (500, 502, 503, 504) and attempt == 0:
                print("Temporary Groq error. Retrying in 3 seconds...")
                time.sleep(3)
                continue
            return "The AI service is temporarily unavailable. Please try again later."
    return "Unable to generate an agricultural recommendation."


def generate_gemini_response(prompt: str) -> str:
    return generate_groq_response(prompt)


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

    return generate_groq_response(prompt)
