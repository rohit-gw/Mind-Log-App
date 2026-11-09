"""
MindLog — Voice Engine
Speech-to-Text via SpeechRecognition (Google free Web API)
Text-to-Speech via gTTS (Google TTS, free, no key needed)
"""
from __future__ import annotations

import io
import re


# ── Translation Helper ────────────────────────────────────────────────────────
def translate_text(text: str, target_lang: str) -> str:
    """
    Translate text into target language using Google Translate GTX endpoint.
    Fast (~100ms), reliable, zero API key needed.
    """
    if not text or target_lang == "en":
        return text

    import urllib.parse
    import requests

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    # Endpoint 1: Google Translate GTX endpoint
    try:
        url = (
            f"https://translate.googleapis.com/translate_a/single"
            f"?client=gtx&sl=auto&tl={target_lang}&dt=t&q={urllib.parse.quote(text)}"
        )
        res = requests.get(url, headers=headers, timeout=6)
        if res.status_code == 200:
            data = res.json()
            translated = "".join([segment[0] for segment in data[0] if segment and segment[0]])
            if translated.strip():
                return translated.strip()
    except Exception as e:
        print(f"[MindLog Translation Warning] Primary endpoint error: {e}")

    # Endpoint 2: Fallback Chrome dict client endpoint
    try:
        url2 = (
            f"https://clients5.google.com/translate_a/t"
            f"?client=dict-chrome-ex&sl=auto&tl={target_lang}&q={urllib.parse.quote(text)}"
        )
        res2 = requests.get(url2, headers=headers, timeout=6)
        if res2.status_code == 200:
            data2 = res2.json()
            if isinstance(data2, list) and data2 and isinstance(data2[0], str):
                return data2[0].strip()
            elif isinstance(data2, list) and data2 and isinstance(data2[0], list):
                return "".join([s[0] for s in data2 if s and isinstance(s, list) and s[0]]).strip()
    except Exception as e:
        print(f"[MindLog Translation Warning] Fallback endpoint error: {e}")

    return text


# ── Text → Speech ─────────────────────────────────────────────────────────────
def speak(text: str, lang: str = "en") -> bytes:
    """
    Convert text to MP3 audio bytes using gTTS.
    If lang != 'en', the text is first translated into that language
    so the speech is in the actual target language, not English with an accent.
    Returns raw MP3 bytes that can be fed to st.audio().
    """
    from gtts import gTTS

    # Strip markdown formatting so it reads naturally
    clean = re.sub(r"\*{1,2}|_{1,2}|`{1,3}|#{1,6}\s?", "", text)
    clean = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", clean)   # links → text

    # Strip emojis and other non-ASCII symbols that TTS can't pronounce
    clean = re.sub(
        r"[\U0001F600-\U0001F64F"   # emoticons
        r"\U0001F300-\U0001F5FF"    # misc symbols & pictographs
        r"\U0001F680-\U0001F6FF"    # transport & map
        r"\U0001F1E0-\U0001F1FF"    # flags
        r"\U00002702-\U000027B0"    # dingbats
        r"\U000024C2-\U0001F251"    # enclosed chars
        r"\U0001F900-\U0001F9FF"    # supplemental symbols
        r"\U0001FA00-\U0001FA6F"    # chess symbols
        r"\U0001FA70-\U0001FAFF"    # symbols extended-A
        r"\U00002600-\U000026FF"    # misc symbols
        r"\U0000FE00-\U0000FE0F"    # variation selectors
        r"\U0000200D"               # zero width joiner
        r"]+", " ", clean
    )

    clean = re.sub(r"\s+", " ", clean).strip()
    clean = clean[:600]  # cap length for TTS

    if not clean:
        clean = "Hey, I'm here for you!"

    # ── Actual Translation so it speaks the true language, NOT an accent ──
    speech_text = clean
    if lang and lang != "en":
        try:
            speech_text = translate_text(clean, target_lang=lang)
        except Exception as e:
            print(f"[MindLog Translation Error] {e}")
            speech_text = clean

    try:
        tts = gTTS(text=speech_text, lang=lang, slow=False)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp.read()
    except Exception as e:
        if lang != "en":
            try:
                tts = gTTS(text=clean, lang="en", slow=False)
                fp = io.BytesIO()
                tts.write_to_fp(fp)
                fp.seek(0)
                return fp.read()
            except Exception:
                pass
        raise e


# ── Speech → Text ─────────────────────────────────────────────────────────────
def transcribe(audio_file) -> tuple[str, str]:
    """
    Transcribe audio (file-like object / UploadedFile from st.audio_input).

    Returns:
        (transcript, error_message)
        On success: (text, "")
        On failure: ("", error_msg)
    """
    import speech_recognition as sr

    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 300
    recognizer.pause_threshold = 0.8

    try:
        with sr.AudioFile(audio_file) as source:
            # Adjust for ambient noise briefly
            recognizer.adjust_for_ambient_noise(source, duration=0.3)
            audio_data = recognizer.record(source)

        text = recognizer.recognize_google(audio_data, language="en-IN")
        return text.strip(), ""

    except sr.UnknownValueError:
        return "", "Hmm, I couldn't quite make that out. Try speaking a little closer to the mic."
    except sr.RequestError as e:
        return "", f"Speech service unavailable: {e}. Check your internet connection."
    except Exception as e:
        return "", f"Audio error: {str(e)[:100]}"
