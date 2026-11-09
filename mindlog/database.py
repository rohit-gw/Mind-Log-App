"""
MindLog — SQLite database layer
"""
import sqlite3
from datetime import datetime, date
from typing import Optional
import pandas as pd
from config import DB_PATH


# ── Schema ────────────────────────────────────────────────────────────────────
_CREATE_ENTRIES = """
CREATE TABLE IF NOT EXISTS entries (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  TEXT    NOT NULL,           -- ISO-8601 datetime
    entry_date  TEXT    NOT NULL,           -- YYYY-MM-DD
    content     TEXT    NOT NULL,
    mood_score  REAL    NOT NULL,           -- 1.0 – 10.0
    compound    REAL    NOT NULL,           -- VADER raw compound  -1 … 1
    sentiment   TEXT    NOT NULL,           -- "positive" | "neutral" | "negative"
    emotions    TEXT    DEFAULT ''          -- comma-separated detected words
);
"""

_CREATE_NUDGES = """
CREATE TABLE IF NOT EXISTS nudges (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    triggered_at TEXT   NOT NULL,
    reason      TEXT    NOT NULL
);
"""

_CREATE_CHAT_MESSAGES = """
CREATE TABLE IF NOT EXISTS chat_messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  TEXT    NOT NULL,
    role        TEXT    NOT NULL,           -- "user" | "assistant"
    content     TEXT    NOT NULL,
    mood_score  REAL,                       -- only for assistant messages
    sentiment   TEXT,
    emotions    TEXT
);
"""


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create tables if they don't exist yet."""
    with _connect() as conn:
        conn.execute(_CREATE_ENTRIES)
        conn.execute(_CREATE_NUDGES)
        conn.execute(_CREATE_CHAT_MESSAGES)
        conn.commit()


# ── Write ─────────────────────────────────────────────────────────────────────
def save_entry(
    content: str,
    mood_score: float,
    compound: float,
    sentiment: str,
    emotions: str = "",
    timestamp: Optional[datetime] = None,
) -> int:
    """Insert a journal entry. Returns the new row id."""
    ts = timestamp or datetime.now()
    with _connect() as conn:
        cur = conn.execute(
            """INSERT INTO entries
               (created_at, entry_date, content, mood_score, compound, sentiment, emotions)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (ts.isoformat(), ts.date().isoformat(),
             content, round(mood_score, 2), round(compound, 4),
             sentiment, emotions),
        )
        conn.commit()
        return cur.lastrowid


def log_nudge(reason: str) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO nudges (triggered_at, reason) VALUES (?, ?)",
            (datetime.now().isoformat(), reason),
        )
        conn.commit()


# ── Read ──────────────────────────────────────────────────────────────────────
def get_all_entries() -> pd.DataFrame:
    """Return all entries as a DataFrame sorted by date asc."""
    with _connect() as conn:
        df = pd.read_sql_query(
            "SELECT * FROM entries ORDER BY created_at ASC", conn
        )
    if df.empty:
        return df
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    return df


def get_recent_entries(n: int = 10) -> pd.DataFrame:
    """Return the N most recent entries."""
    with _connect() as conn:
        df = pd.read_sql_query(
            "SELECT * FROM entries ORDER BY created_at DESC LIMIT ?", conn, params=(n,)
        )
    if df.empty:
        return df
    df["created_at"] = pd.to_datetime(df["created_at"])
    return df


def get_daily_averages() -> pd.DataFrame:
    """Return one row per day with avg mood_score."""
    with _connect() as conn:
        df = pd.read_sql_query(
            """SELECT entry_date,
                      AVG(mood_score) AS avg_mood,
                      COUNT(*)        AS entry_count
               FROM entries
               GROUP BY entry_date
               ORDER BY entry_date ASC""",
            conn,
        )
    if df.empty:
        return df
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    return df


def get_weekly_averages() -> pd.DataFrame:
    """Return one row per ISO week."""
    df = get_all_entries()
    if df.empty:
        return df
    df["week"] = df["created_at"].dt.to_period("W").dt.start_time
    return df.groupby("week")["mood_score"].mean().reset_index().rename(
        columns={"mood_score": "avg_mood"}
    )


def get_word_frequency(n_recent: int = 50) -> dict:
    """Return word counts from the N most recent entries (for pattern detection)."""
    import re
    from collections import Counter

    df = get_recent_entries(n_recent)
    if df.empty:
        return {}
    stop_words = {
        "i", "the", "a", "and", "to", "was", "is", "it", "my", "me",
        "in", "of", "that", "so", "but", "just", "have", "had", "with",
        "for", "on", "at", "this", "today", "day", "really", "very",
        "got", "get", "feel", "felt", "bit", "little", "been", "not",
        "am", "are", "do", "did", "been", "an", "we", "he", "she",
    }
    all_words = []
    for text in df["content"]:
        words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
        all_words.extend([w for w in words if w not in stop_words])
    return dict(Counter(all_words).most_common(20))


def total_entries() -> int:
    with _connect() as conn:
        row = conn.execute("SELECT COUNT(*) FROM entries").fetchone()
        return row[0]


# ── Chat Message Persistence ──────────────────────────────────────────────────
def save_chat_message(
    role: str,
    content: str,
    mood_score: float = None,
    sentiment: str = None,
    emotions: str = None,
) -> int:
    """Save a single chat message (user or assistant) to the database."""
    ts = datetime.now().isoformat()
    with _connect() as conn:
        cur = conn.execute(
            """INSERT INTO chat_messages
               (created_at, role, content, mood_score, sentiment, emotions)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (ts, role, content, mood_score, sentiment, emotions),
        )
        conn.commit()
        return cur.lastrowid


def load_chat_messages() -> list[dict]:
    """Load all chat messages from the database, sorted by creation time."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT role, content, mood_score, sentiment, emotions "
            "FROM chat_messages ORDER BY id ASC"
        ).fetchall()

    messages = []
    for row in rows:
        msg = {"role": row["role"], "content": row["content"]}
        if row["mood_score"] is not None:
            msg["mood_score"] = row["mood_score"]
        if row["sentiment"] is not None:
            msg["sentiment"] = row["sentiment"]
        if row["emotions"] is not None:
            msg["emotions"] = [e.strip() for e in row["emotions"].split(",") if e.strip()]
        messages.append(msg)
    return messages


def clear_chat_messages() -> None:
    """Clear all chat messages (for a 'new conversation' feature)."""
    with _connect() as conn:
        conn.execute("DELETE FROM chat_messages")
        conn.commit()
