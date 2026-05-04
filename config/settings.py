from dotenv import load_dotenv
import os

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL   = "gemini-2.5-flash"

# Memory settings
MEMORY_MAX_MESSAGES = 10

# Agent settings
AGENT_VERBOSE = True