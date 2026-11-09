"""
MindLog — Pattern & Anomaly Detection
Rolling average baseline comparison + keyword frequency analysis.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from config import (
    ROLLING_WINDOW_DAYS,
    MIN_ENTRIES_FOR_BASELINE,
    NUDGE_DROP_THRESHOLD,
    SUSTAINED_DAYS,
)


@dataclass
class PatternReport:
    has_pattern: bool = False
    nudge_message: str = ""
    recent_avg: Optional[float] = None
    baseline_avg: Optional[float] = None
    drop: Optional[float] = None
    frequent_words: dict = field(default_factory=dict)
    day_of_week_insight: str = ""
    streak_info: str = ""


def analyse(all_entries: pd.DataFrame, word_freq: dict) -> PatternReport:
    """
    Run all pattern checks on the entries DataFrame.
    Returns a PatternReport describing any detected patterns.
    """
    report = PatternReport(frequent_words=word_freq)

    if all_entries.empty or len(all_entries) < MIN_ENTRIES_FOR_BASELINE:
        return report

    df = all_entries.sort_values("created_at").copy()
    df["created_at"] = pd.to_datetime(df["created_at"])

    # ── 1. Sustained low-mood trend ──────────────────────────────────────────
    recent = df.tail(ROLLING_WINDOW_DAYS)
    older  = df.iloc[:-ROLLING_WINDOW_DAYS] if len(df) > ROLLING_WINDOW_DAYS else pd.DataFrame()

    recent_avg  = recent["mood_score"].mean()
    baseline_avg = df["mood_score"].mean() if older.empty else older["mood_score"].mean()

    report.recent_avg   = round(recent_avg, 2)
    report.baseline_avg = round(baseline_avg, 2)
    report.drop         = round(baseline_avg - recent_avg, 2)

    # Check sustained low: last N entries all below baseline
    last_n = df.tail(SUSTAINED_DAYS)["mood_score"].tolist()
    sustained_low = all(s < baseline_avg for s in last_n) and len(last_n) >= SUSTAINED_DAYS

    if recent_avg < (baseline_avg - NUDGE_DROP_THRESHOLD) or sustained_low:
        report.has_pattern = True
        report.nudge_message = _build_nudge(recent_avg, baseline_avg, report.drop)

    # ── 2. Day-of-week pattern ───────────────────────────────────────────────
    if len(df) >= 14:
        df["weekday"] = df["created_at"].dt.day_name()
        dow_avg = df.groupby("weekday")["mood_score"].mean()
        worst_day = dow_avg.idxmin()
        best_day  = dow_avg.idxmax()
        worst_score = dow_avg[worst_day]
        best_score  = dow_avg[best_day]

        if (best_score - worst_score) > 1.5:
            report.day_of_week_insight = (
                f"Your mood tends to be lower on **{worst_day}s** "
                f"(avg {worst_score:.1f}) and higher on **{best_day}s** "
                f"(avg {best_score:.1f})."
            )

    # ── 3. Frequent negative emotion words ──────────────────────────────────
    negative_flags = ["tired", "anxious", "lonely", "sad", "angry"]
    flagged = [w for w in negative_flags if w in word_freq and word_freq[w] >= 3]
    if flagged:
        words_str = ", ".join(f"**{w}**" for w in flagged)
        report.streak_info = f"You've mentioned {words_str} quite a bit this week."

    return report


def _build_nudge(recent_avg: float, baseline_avg: float, drop: float) -> str:
    """Craft a warm, non-diagnostic nudge message."""
    if recent_avg < 3.5:
        return (
            "It looks like the past week or so has felt pretty heavy. "
            "You don't have to carry that alone — is there someone you'd want to talk to about how you're feeling? "
            "Even a short conversation with someone you trust can help."
        )
    elif drop > 2.5:
        return (
            "I've noticed your entries have felt a bit heavier lately compared to your usual. "
            "That's completely okay — life has its seasons. "
            "Is there something specific that's been weighing on you? "
            "And if it would help, talking to someone you trust might bring some relief."
        )
    else:
        return (
            "Looking at your recent entries, things seem to have felt a bit more challenging than usual. "
            "It might be worth checking in with yourself — or even reaching out to a friend. "
            "Small steps matter."
        )
