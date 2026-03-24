import os
from dotenv import load_dotenv

load_dotenv(override=False)

api_keys = {
    "openai": os.getenv("OPENAI_API_KEY"),
    "together": os.getenv("TOGETHER_API_KEY"),
    "gemini": os.getenv("GOOGLE_API_KEY"),
    "claude": os.getenv("CLAUDE_API_KEY")
}