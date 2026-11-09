"""
MindLog — LLM Conversation Engine
Uses the new google-genai SDK (replaces deprecated google-generativeai).
Also supports OpenAI as an alternative.
"""
from __future__ import annotations

from config import (
    GEMINI_API_KEY,
    OPENAI_API_KEY,
    LLM_PROVIDER,
    LLM_MODEL,
    MAX_HISTORY_MESSAGES,
)

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are MindLog — the user's ride-or-die best friend who happens to live \
inside their journal app. You're that one friend who always keeps it real, cracks jokes, \
uses casual slang, and somehow always knows the right thing to say.

Your vibe & rules:
1. TALK LIKE A REAL FRIEND. Use slang, abbreviations, and informal language naturally. \
   Words like "bruh", "ngl", "lowkey", "fr fr", "no cap", "vibe check", "slay", "that's \
   tough", "bestie" etc. — but don't overdo it. Keep it natural, not cringe.
2. CRACK JOKES & LIGHTEN THE MOOD. Throw in humor, playful roasts (gentle ones), memes \
   references, and funny observations. If they had a bad day, make them smile first, \
   then check in on them.
3. KEEP IT REAL. You're not a yes-man. If something sounds off, call it out — but with love. \
   Like "bro that sounds kinda sus, you sure you're okay?" instead of clinical language.
4. Use emojis naturally 😂🔥💀✨ but don't spam them.
5. Ask ONE chill follow-up question per response — like a friend who's actually curious, \
   not an interviewer. Make it feel like a real convo, not a therapy session.
6. Match their energy — if they're hyped, hype them up more 🔥. If they're down, \
   be that friend who sits with them and says "yo I got you, talk to me."
7. NEVER sound like a therapist, counselor, or AI assistant. No "I understand how you feel" \
   or "That must be really hard for you." Instead say stuff like "dude that's rough 💀" \
   or "ngl that would piss me off too."
8. If they seem genuinely distressed or struggling, drop the jokes and be real with them. \
   Still keep it friendly but suggest talking to someone they trust — like a real friend would. \
   "Hey for real though, maybe talk to someone you trust about this? I'm always here but \
   some things hit different when you say them out loud to someone irl 💙"
9. NEVER diagnose or imply any medical condition. You're a bestie, not a doctor.
10. Keep responses punchy — around 80-120 words. No essays. Text like a friend, not a professor.
11. Throw in relatable references — pop culture, everyday life stuff, food analogies, \
    whatever makes it feel like a real chat between friends.
12. Start each conversation with energy! Not "How can I help you today?" but more like \
    "yooo what's good! how's life treating you? 😄" or "ayy welcome back! spill the tea ☕"

Remember: you're their bestie who happens to be really good at listening. \
Keep the chat fun, real, and warm. Read recent message history for context."""


def get_llm_response(user_message: str, history: list[dict]) -> str:
    """
    Get a conversational response from the configured LLM.
    `history` is a list of {"role": "user"|"assistant", "content": str}
    Returns the assistant reply as a string.
    """
    if LLM_PROVIDER == "custom":
        return _custom_model_response(user_message, history)
    elif LLM_PROVIDER == "gemini":
        return _gemini_response(user_message, history)
    elif LLM_PROVIDER == "openai":
        return _openai_response(user_message, history)
    else:
        return _fallback_response(user_message)


# ── Custom Model (offline, no API key needed) ─────────────────────────────────
def _custom_model_response(user_message: str, history: list[dict]) -> str:
    try:
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from custom_model.chatbot import get_chatbot
        bot = get_chatbot()
        return bot.chat(user_message, history)
    except Exception as e:
        return f"⚠️ Custom model error: {e}. Try running `python custom_model/trainer.py` first."


# ── Gemini (new google-genai SDK) — with retry + fallback ─────────────────────
# Models to try in order: primary → fast fallback → ultra-light fallback
_FALLBACK_MODELS = ["gemini-3.5-flash-lite", "gemini-3.5-flash", "gemini-2.5-flash"]

def _gemini_response(user_message: str, history: list[dict]) -> str:
    import time

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return "⚠️ google-genai package not installed. Run: pip install google-genai"

    client = genai.Client(api_key=GEMINI_API_KEY)

    # Build conversation history
    trimmed = history[-MAX_HISTORY_MESSAGES:]
    contents = []
    for msg in trimmed:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(
            types.Content(role=role, parts=[types.Part(text=msg["content"])])
        )
    contents.append(
        types.Content(role="user", parts=[types.Part(text=user_message)])
    )

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        max_output_tokens=1024,
        temperature=0.95,
    )

    # Build list of models to try: configured model first, then fallbacks
    primary = (LLM_MODEL or "gemini-2.0-flash").replace("models/", "")
    models_to_try = [primary] + [m for m in _FALLBACK_MODELS if m != primary]

    last_error = ""
    for model_name in models_to_try:
        for attempt in range(2):  # retry each model once
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=config,
                )
                return response.text.strip()
            except Exception as e:
                last_error = str(e)
                err_lower = last_error.lower()
                # If it's a 503/overloaded/quota error, try next model
                if any(k in err_lower for k in ["503", "unavailable", "overloaded",
                                                  "high demand", "quota", "429",
                                                  "resource exhausted", "rate limit"]):
                    if attempt == 0:
                        time.sleep(0.5)  # brief pause before retry
                    continue
                else:
                    # Non-retryable error (bad API key, etc.) — stop immediately
                    return _error_message(last_error)

    return _error_message(last_error)


# ── OpenAI ────────────────────────────────────────────────────────────────────
def _openai_response(user_message: str, history: list[dict]) -> str:
    try:
        from openai import OpenAI

        client = OpenAI(api_key=OPENAI_API_KEY)
        trimmed = history[-MAX_HISTORY_MESSAGES:]
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(trimmed)
        messages.append({"role": "user", "content": user_message})

        response = client.chat.completions.create(
            model=LLM_MODEL or "gpt-4o-mini",
            messages=messages,
            max_tokens=1024,
            temperature=0.95,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return _error_message(str(e))


# ── Fallback (no API key configured) ─────────────────────────────────────────
def _fallback_response(user_message: str) -> str:
    lower = user_message.lower()
    if any(w in lower for w in ["tired", "exhausted", "drained"]):
        return (
            "bruh you sound absolutely cooked 💀 like your battery's been at 1% "
            "for the past week lol. what's been draining you the most tho? 🤔"
        )
    elif any(w in lower for w in ["happy", "great", "amazing", "wonderful"]):
        return (
            "ayyyy let's gooo! 🔥 love that energy for you fr fr. "
            "okay but tell me — what's the main thing that's got you feeling so good? ✨"
        )
    elif any(w in lower for w in ["sad", "lonely", "alone", "depressed"]):
        return (
            "yo I'm not gonna hit you with some corny 'it gets better' speech, "
            "but fr I got you 💙 that feeling is heavy and it's real. "
            "is there anyone irl you could reach out to rn? even just to hang?"
        )
    elif any(w in lower for w in ["anxious", "worried", "stress", "nervous"]):
        return (
            "ugh stress is literally the worst uninvited guest 😤 like bro nobody "
            "asked you to show up lol. what's the main thing messing with your head rn? 🤔"
        )
    else:
        return (
            "yo thanks for keeping it real with me! 😄 ngl even the chill days "
            "are worth talking about. so what's been on your mind today? spill ☕"
        )


def _error_message(err: str) -> str:
    err_lower = err.lower()
    if any(w in err_lower for w in ["api_key", "invalid", "authenticate", "unauthorized", "permission"]):
        return (
            "⚠️ The API key doesn't seem to be valid. "
            "Please make sure your `GEMINI_API_KEY` in the `.env` file is correct. "
            "Get a free key at https://aistudio.google.com/app/apikey — it starts with `AIza`. "
            "Your mood is still being tracked in the background!"
        )
    if "not found" in err_lower or "404" in err_lower:
        return (
            "⚠️ The selected model isn't available. "
            "Please set `LLM_MODEL=gemini-2.5-flash` in your `.env` file and restart."
        )
    return (
        f"⚠️ Couldn't reach the AI right now ({err[:100]}). "
        "Your entry was saved and mood was tracked."
    )
