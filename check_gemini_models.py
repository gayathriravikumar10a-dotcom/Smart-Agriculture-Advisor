
from pathlib import Path
from dotenv import load_dotenv
from google import genai

project_root = Path(__file__).resolve().parent
load_dotenv(project_root / ".env", override=True)

client = genai.Client()

try:
    print("Checking available Gemini models...\n")

    for model in client.models.list():
        name = getattr(model, "name", "") or ""

        if "flash" in name.lower() and "gemini" in name.lower():
            print(name)

except Exception as error:
    print("Could not list models.")
    print(type(error).__name__, str(error))
