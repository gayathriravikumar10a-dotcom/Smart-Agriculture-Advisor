from pathlib import Path
from dotenv import load_dotenv
from google import genai

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env", override=True)

client = genai.Client()

response = client.models.generate_content(
    model="gemini-3.8-flash",
    contents="Give one simple tip for growing healthy tomato plants."
)

print(response.text)