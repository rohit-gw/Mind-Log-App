"""
MindLog — Sentiment & Mood Scoring Engine
Uses VADER for fast polarity, plus keyword emotion detection.
"""
from __future__ import annotations

import re
from typing import NamedTuple

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from config import MOOD_MIN, MOOD_MAX

_analyzer = SentimentIntensityAnalyzer()

# ── Emotion keyword lexicon ───────────────────────────────────────────────────
_EMOTION_LEXICON: dict[str, list[str]] = {
    "anxious":   ["anxious", "anxiety", "worried", "worry", "nervous", "panic", "stressed", "stress", "overwhelmed"],
    "tired":     ["tired", "exhausted", "drained", "fatigue", "sleepy", "burnout", "worn out", "weary"],
    "sad":       ["sad", "unhappy", "miserable", "depressed", "hopeless", "crying", "cried", "tears", "grief"],
    "angry":     ["angry", "frustrated", "annoyed", "furious", "rage", "mad", "irritated", "upset"],
    "lonely":    ["lonely", "alone", "isolated", "disconnected", "empty", "invisible"],
    "content":   ["content", "okay", "fine", "alright", "decent", "calm", "peaceful", "settled"],
    "happy":     ["happy", "joyful", "excited", "great", "wonderful", "amazing", "fantastic", "love", "loved"],
    "grateful":  ["grateful", "thankful", "blessed", "appreciate", "appreciated"],
    "hopeful":   ["hopeful", "optimistic", "looking forward", "excited about", "can't wait"],
    "proud":     ["proud", "accomplished", "achieved", "succeeded", "nailed", "did it"],
}


class MoodResult(NamedTuple):
    mood_score: float       # 1.0 – 10.0
    compound: float         # VADER raw: -1 … +1
    sentiment: str          # "positive" | "neutral" | "negative"
    emotions: list[str]     # detected emotion labels
    pos: float
    neu: float
    neg: float


def score_entry(text: str) -> MoodResult:
    """
    Score a journal entry.

    Returns a MoodResult with mood_score on a 1-10 scale,
    raw VADER compound, sentiment label, and detected emotions.
    """
    scores = _analyzer.polarity_scores(text)
    compound: float = scores["compound"]

    # Map compound (-1…+1) → mood score (1…10)
    mood_score = _compound_to_mood(compound)

    # Sentiment label
    if compound >= 0.05:
        sentiment = "positive"
    elif compound <= -0.05:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    # Detect emotions
    emotions = _detect_emotions(text)

    return MoodResult(
        mood_score=round(mood_score, 2),
        compound=round(compound, 4),
        sentiment=sentiment,
        emotions=emotions,
        pos=scores["pos"],
        neu=scores["neu"],
        neg=scores["neg"],
    )


def _compound_to_mood(compound: float) -> float:
    """
    Normalise VADER compound (-1…+1) → mood score (1…10).
    Uses a slightly non-linear mapping so the middle range
    is 4-6 for neutral text (rather than collapsing everything to 5.5).
    """
    # Linear mapping first: (-1, +1) → (1, 10)
    linear = (compound + 1) / 2 * (MOOD_MAX - MOOD_MIN) + MOOD_MIN
    return max(MOOD_MIN, min(MOOD_MAX, linear))


def _detect_emotions(text: str) -> list[str]:
    """Return a list of detected emotion labels from the lexicon."""
    text_lower = text.lower()
    found = []
    for emotion, keywords in _EMOTION_LEXICON.items():
        for kw in keywords:
            if re.search(r"\b" + re.escape(kw) + r"\b", text_lower):
                found.append(emotion)
                break
    return found


def mood_label(score: float) -> str:
    """Return a human-readable label for a mood score."""
    if score >= 8.0:
        return "😄 Great"
    elif score >= 6.5:
        return "🙂 Good"
    elif score >= 5.0:
        return "😐 Okay"
    elif score >= 3.5:
        return "😔 Low"
    else:
        return "😢 Tough"


def mood_color(score: float) -> str:
    """Return a hex colour for a mood score (for charts/UI)."""
    if score >= 8.0:
        return "#00C9B1"   # teal
    elif score >= 6.5:
        return "#7BC67E"   # green
    elif score >= 5.0:
        return "#FFD600"   # yellow
    elif score >= 3.5:
        return "#FF9B6A"   # orange
    else:
        return "#FF6B9D"   # pink-red
