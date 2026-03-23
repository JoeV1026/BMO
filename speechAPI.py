import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

Gemini_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

if genai is not None:
    client = genai.Client(api_key=Gemini_API_KEY)
else:
    client = None


def brain(text: str) -> str:
    user_text = (text or "").strip()

    try:
        if client is not None:
            response = client.models.generate_content(model=MODEL, contents=user_text)
            answer = response.text.strip()
        return answer
    except Exception:
        return ""