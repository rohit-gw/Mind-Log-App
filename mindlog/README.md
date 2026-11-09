# 🧠 MindLog — AI-Powered Mental Wellness Journal

A conversational journaling companion that listens, reflects, and tracks your emotional wellbeing over time.

---

## Quick Start

### 1. Install dependencies
```bash
cd mindlog
pip install -r requirements.txt
```

### 2. Set your API key
```bash
# Copy the example env file
copy .env.example .env

# Open .env and add your Gemini API key:
# Get one free at https://aistudio.google.com/app/apikey
GEMINI_API_KEY=your_key_here
```

### 3. Run the app
```bash
streamlit run app.py
```

The app opens at **http://localhost:8501**

---

## Features

| Feature | Description |
|---------|-------------|
| 💬 **Conversational journaling** | Chat with an empathetic LLM that reflects back, never lectures |
| 🔍 **Silent mood scoring** | Every entry is scored 1–10 using VADER sentiment analysis |
| 📈 **Mood trend chart** | Daily averages plotted as an interactive Plotly line graph |
| 📊 **Weekly averages** | Bar chart of weekly mood averages |
| 🔍 **Pattern detection** | Rolling-average algorithm detects sustained low-mood trends |
| 💙 **Supportive nudge** | Gentle, non-diagnostic prompt when patterns are detected |
| 🗄️ **SQLite storage** | All entries stored locally — no cloud, no sharing |

---

## Project Structure

```
mindlog/
├── app.py               # Main Streamlit app
├── config.py            # Configuration & constants
├── database.py          # SQLite CRUD & analytics queries
├── sentiment_engine.py  # VADER mood scoring + emotion detection
├── llm_engine.py        # Gemini/OpenAI conversation engine
├── pattern_detector.py  # Rolling average & nudge logic
├── requirements.txt
├── .env.example         # Environment variable template
└── mindlog.db           # Auto-created SQLite database
```

---

## Ethical Design Principles

> **MindLog is a wellness journaling tool, not a mental health diagnostic app.**

- ❌ Never diagnoses any condition
- ❌ Never uses clinical language like "depression" or "anxiety disorder"
- ✅ Frames all observations as *patterns*, not conclusions
- ✅ Encourages connection with trusted people or professionals when relevant
- ✅ Surfaces helpline resources alongside any supportive nudge
- ✅ All data stays local (SQLite on your machine)

---

## LLM Configuration

By default MindLog uses **Google Gemini 1.5 Flash** (free tier, fast).

To switch to OpenAI, edit `.env`:
```
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=your_key_here
```

If no API key is set, the app uses a built-in rule-based fallback and still tracks mood.

---

## Pattern Detection Logic

```
recent_avg  = mean of last 7 entries
baseline    = mean of all earlier entries
drop        = baseline - recent_avg

if drop > 1.8  OR  last 5 entries all below baseline:
    → show supportive nudge
```

Day-of-week and keyword frequency patterns are also surfaced in the sidebar.
