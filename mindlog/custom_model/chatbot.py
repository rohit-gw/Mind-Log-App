"""
MindLog Custom Model — Inference Engine (Chatbot)

Architecture:
  Step 1: Intent Classifier (TF-IDF + Logistic Regression)
          → Predict the emotional intent of the user's message
  Step 2: Response Retriever (TF-IDF cosine similarity)
          → Find the most contextually relevant response for that intent
  Step 3: Follow-up injector
          → Append a reflective question if the response doesn't already end with one
  Step 4: Context memory (last 3 turns)
          → Avoid repeating the same response twice in a row

All inference runs fully offline on CPU — no API calls needed.
"""
from __future__ import annotations

import os
import random
import re
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE       = Path(__file__).parent
MODEL_DIR  = HERE / "saved_model"

INTENT_MODEL_PATH    = MODEL_DIR / "intent_classifier.joblib"
RESPONSE_VEC_PATH    = MODEL_DIR / "response_vectorizer.joblib"
RESPONSE_MATRIX_PATH = MODEL_DIR / "response_matrix.npy"
RESPONSE_MAP_PATH    = MODEL_DIR / "response_map.joblib"


# ── Fallback responses (before model is trained) ──────────────────────────────
_FALLBACK = [
    "Tell me more — what's been going on for you today?",
    "I'm here and listening. What's on your mind?",
    "That sounds like a lot to hold. What feels most important to share right now?",
    "I'd like to understand more. Can you tell me what's been happening?",
    "What's sitting with you most as you write this?",
]

# ── Reflective follow-up questions (appended when needed) ─────────────────────
_FOLLOW_UPS = {
    "tired":        ["What's been taking the most out of you?", "When did you last truly rest?"],
    "anxious":      ["What's the worry that keeps coming back?", "What feels most uncertain right now?"],
    "sad":          ["What's been heaviest on your heart?", "Has anything helped, even slightly?"],
    "angry":        ["What's at the center of this frustration?", "How long has this been building?"],
    "lonely":       ["When did you last feel genuinely connected?", "What kind of connection do you miss most?"],
    "happy":        ["What made today feel different?", "What's one thing you'd want to carry forward?"],
    "grateful":     ["What are you most thankful for today?", "What did you notice that you might have overlooked before?"],
    "hopeful":      ["What's giving you that sense of hope?", "What do you think is changing?"],
    "overwhelmed":  ["What's piling up the most?", "Is there one thing you could put down, even temporarily?"],
    "proud":        ["What did it take to get there?", "What did you learn about yourself through this?"],
    "relationship": ["What do you think the real issue is?", "What outcome would feel like resolution?"],
    "work_school":  ["What's the most pressing thing you're up against?", "What would feel like relief?"],
    "general":      ["What's one thing you noticed today?", "What's been taking up your mental space, even quietly?"],
}

_DEFAULT_FOLLOW_UP = [
    "What feels most important about that?",
    "How has that been affecting you?",
    "What would help most right now?",
    "Is there more you want to say about that?",
]


class MindLogChatbot:
    """
    Retrieval-based chatbot with intent classification.

    The model is loaded lazily on first use (call `load()` explicitly
    or let it auto-load on first `chat()` call).
    """

    def __init__(self) -> None:
        self._intent_model = None
        self._resp_vec     = None
        self._resp_matrix  = None
        self._resp_map     = None
        self._loaded       = False

        # Context memory: track last 3 responses to avoid repetition
        self._recent_responses: list[str] = []

    def is_trained(self) -> bool:
        return INTENT_MODEL_PATH.exists()

    def load(self) -> bool:
        """Load saved model artifacts. Returns True on success."""
        if self._loaded:
            return True
        if not self.is_trained():
            return False
        try:
            self._intent_model = joblib.load(INTENT_MODEL_PATH)
            self._resp_vec     = joblib.load(RESPONSE_VEC_PATH)
            self._resp_matrix  = np.load(RESPONSE_MATRIX_PATH)
            self._resp_map     = joblib.load(RESPONSE_MAP_PATH)
            self._loaded = True
            return True
        except Exception:
            return False

    def predict_intent(self, text: str) -> tuple[str, float]:
        """
        Predict the intent tag and confidence for a user message.
        Returns (tag, confidence_0_to_1).
        """
        if not self._loaded:
            return "general", 0.0

        probs  = self._intent_model.predict_proba([text.lower()])[0]
        labels = self._intent_model.classes_
        idx    = int(np.argmax(probs))
        return labels[idx], float(probs[idx])

    def retrieve_response(self, text: str, intent: str, confidence: float) -> str:
        """
        Find the best response:
          - High confidence (>0.55): filter to intent's responses, pick best cosine match
          - Lower confidence:        search across all responses
        Avoids repeating recently used responses.
        """
        if not self._loaded:
            return random.choice(_FALLBACK)

        query_vec = self._resp_vec.transform([text.lower()]).toarray()
        sims      = cosine_similarity(query_vec, self._resp_matrix)[0]

        # Build candidate list
        if confidence > 0.55:
            # Only consider responses matching the predicted intent
            candidates = [
                (i, sims[i]) for i, (tag, _) in enumerate(self._resp_map)
                if tag == intent
            ]
        else:
            # Broader search
            candidates = list(enumerate(sims))

        if not candidates:
            candidates = list(enumerate(sims))

        # Sort by similarity descending
        candidates.sort(key=lambda x: x[1], reverse=True)

        # Pick highest-similarity response that wasn't recently used
        for idx, score in candidates[:8]:
            tag, response = self._resp_map[idx]
            if response not in self._recent_responses:
                self._recent_responses.append(response)
                if len(self._recent_responses) > 6:
                    self._recent_responses.pop(0)
                return response

        # Fallback: return top match regardless
        top_idx = candidates[0][0]
        return self._resp_map[top_idx][1]

    def _add_follow_up(self, response: str, intent: str) -> str:
        """Always append a reflective question to make the chat interactive."""
        # Ensure response ends with punctuation before adding a question
        if not response.rstrip().endswith((".", "!", "?")):
            response = response.rstrip() + "."
        pool = _FOLLOW_UPS.get(intent, _DEFAULT_FOLLOW_UP)
        question = random.choice(pool)
        return f"{response} {question}"

    def chat(
        self,
        user_message: str,
        history: Optional[list[dict]] = None,
    ) -> str:
        """
        Generate a response to user_message.

        Args:
            user_message: The user's journal text.
            history:      List of {"role": ..., "content": ...} (not used heavily,
                          but kept for API compatibility with llm_engine interface).
        Returns:
            Response string.
        """
        if not self._loaded:
            self.load()

        if not self._loaded:
            return (
                "The custom model hasn't been trained yet. "
                "Please run the trainer first: python custom_model/trainer.py"
            )

        intent, confidence = self.predict_intent(user_message)
        response = self.retrieve_response(user_message, intent, confidence)
        response = self._add_follow_up(response, intent)

        return response

    def explain(self, user_message: str) -> dict:
        """Return a debug explanation of how the response was chosen."""
        if not self._loaded:
            self.load()
        intent, confidence = self.predict_intent(user_message)
        return {
            "predicted_intent": intent,
            "confidence": round(confidence, 3),
            "top_intents": self._top_intents(user_message),
        }

    def _top_intents(self, text: str, n: int = 3) -> list[dict]:
        if not self._loaded:
            return []
        probs  = self._intent_model.predict_proba([text.lower()])[0]
        labels = self._intent_model.classes_
        ranked = sorted(zip(labels, probs), key=lambda x: x[1], reverse=True)
        return [{"intent": t, "confidence": round(float(p), 3)} for t, p in ranked[:n]]


# ── Module-level singleton ────────────────────────────────────────────────────
_chatbot: Optional[MindLogChatbot] = None


def get_chatbot() -> MindLogChatbot:
    global _chatbot
    if _chatbot is None:
        _chatbot = MindLogChatbot()
        _chatbot.load()
    return _chatbot
