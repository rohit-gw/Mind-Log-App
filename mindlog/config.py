"""
MindLog — Configuration
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

# ── API Keys ─────────────────────────────────────────────────────────────────
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

# Which LLM provider to use: "gemini" | "openai"
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "gemini")
LLM_MODEL: str = os.getenv("LLM_MODEL", "gemini-1.5-flash")

# ── Database ─────────────────────────────────────────────────────────────────
DB_PATH: str = str(Path(__file__).parent / "mindlog.db")

# ── Pattern Detection ────────────────────────────────────────────────────────
ROLLING_WINDOW_DAYS: int = 7          # days to compute recent average
MIN_ENTRIES_FOR_BASELINE: int = 5     # need at least N entries before detecting
NUDGE_DROP_THRESHOLD: float = 1.8    # if recent avg is this many points below baseline
SUSTAINED_DAYS: int = 5               # how many recent entries must be low

# ── Mood Scale ───────────────────────────────────────────────────────────────
MOOD_MIN: int = 1
MOOD_MAX: int = 10

# ── App ──────────────────────────────────────────────────────────────────────
APP_TITLE: str = "MindLog"
APP_ICON: str = "🧠"
MAX_HISTORY_MESSAGES: int = 12   # messages kept in LLM context window

# ── Safety ───────────────────────────────────────────────────────────────────
HELPLINE_TEXT: str = (
    "If you're going through a difficult time, talking to someone can help. "
    "**iCall (India):** 9152987821 | "
    "**Vandrevala Foundation:** 1860-2662-345 (24/7) | "
    "**Snehi:** 044-24640050"
)
