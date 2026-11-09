"""
MindLog — Main Streamlit Application  (with Voice Assistant)
Run with: streamlit run app.py   (from inside mindlog/ folder)
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

import database as db
import sentiment_engine as se
import pattern_detector as pd_engine
import llm_engine as llm
import voice_engine as ve
from config import APP_TITLE, APP_ICON, HELPLINE_TEXT

LANG_DICT = {
    "en": "English",
    "hi": "Hindi",
    "kn": "Kannada",
    "fr": "French",
    "es": "Spanish",
    "de": "German",
    "ja": "Japanese",
}

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Mobile smooth touch scrolling & GPU acceleration */
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    -webkit-overflow-scrolling: touch !important;
    touch-action: pan-y !important;
    overscroll-behavior-y: contain !important;
}

.stApp { background-color: #0D1B2A; color: #CCD6E0; }

section[data-testid="stSidebar"] {
    background-color: #0A1520;
    border-right: 1px solid #1A3A5C;
}
section[data-testid="stSidebar"] * { color: #CCD6E0 !important; }

.stChatMessage {
    background-color: #142A40 !important;
    border-radius: 12px;
    overflow: visible !important;
    max-height: none !important;
    height: auto !important;
}
[data-testid="stChatMessageContent"] {
    color: #CCD6E0 !important;
    overflow: visible !important;
    max-height: none !important;
    height: auto !important;
    white-space: normal !important;
    word-wrap: break-word !important;
    overflow-wrap: break-word !important;
    text-overflow: unset !important;
}
[data-testid="stChatMessageContent"] p {
    white-space: normal !important;
    word-wrap: break-word !important;
    overflow-wrap: break-word !important;
    text-overflow: unset !important;
}
/* Ensure the chat message container expands fully */
.stChatMessage > div {
    overflow: visible !important;
    max-height: none !important;
}
.stChatInputContainer { background-color: #0D1B2A !important; }
textarea { background-color: #142A40 !important; color: #FFFFFF !important;
           border-radius: 10px !important; }

[data-testid="stMetricLabel"] { color: #00C9B1 !important; font-size:0.8rem !important; }
[data-testid="stMetricValue"] { color: #FFFFFF !important; }

/* Voice banner */
.voice-active {
    background: linear-gradient(135deg, #0D2A3A, #0D1B2A);
    border: 1px solid #00C9B1;
    border-radius: 14px;
    padding: 16px 20px;
    margin: 10px 0 6px 0;
    text-align: center;
}
.voice-transcript {
    background: #142A40;
    border-left: 4px solid #9B5DE5;
    border-radius: 10px;
    padding: 12px 16px;
    margin: 6px 0;
    font-style: italic;
    color: #CCD6E0;
}
.nudge-box {
    background: linear-gradient(135deg,#1A2E1A,#142A20);
    border-left: 4px solid #00C9B1;
    border-radius: 10px;
    padding: 14px 18px;
    margin: 10px 0;
    color: #CCD6E0;
}
.helpline-box {
    background: #142A40;
    border-left: 4px solid #9B5DE5;
    border-radius: 10px;
    padding: 10px 16px;
    margin: 6px 0;
    font-size: 0.85rem;
    color: #CCD6E0;
}
.pattern-insight {
    background: #142A40;
    border-left: 4px solid #FFD600;
    border-radius: 10px;
    padding: 10px 16px;
    margin: 6px 0;
    color: #CCD6E0;
    font-size: 0.9rem;
}
h1,h2,h3 { color: #FFFFFF !important; }
.stMarkdown p { color: #CCD6E0; }
hr { border-color: #1A3A5C; }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0D1B2A; }
::-webkit-scrollbar-thumb { background: #1A3A5C; border-radius: 3px; }

/* Audio player style */
audio { width: 100%; border-radius: 8px; margin-top: 6px; }
[data-testid="stAudio"] { margin-top: 4px; }

/* Tab styling */
.stTabs [data-baseweb="tab-list"] { background-color: #142A40; border-radius: 10px; padding: 4px; }
.stTabs [data-baseweb="tab"] { color: #CCD6E0 !important; border-radius: 8px; }
.stTabs [aria-selected="true"] { background-color: #00C9B1 !important; color: #0D1B2A !important; }
</style>
""", unsafe_allow_html=True)

# ── Initialise ────────────────────────────────────────────────────────────────
db.init_db()

if "messages"      not in st.session_state:
    # Load persisted chat history from the database
    st.session_state.messages = db.load_chat_messages()
if "last_mood"     not in st.session_state:
    # Restore last mood from the most recent assistant message
    last_mood = None
    for msg in reversed(st.session_state.messages):
        if msg.get("mood_score") is not None:
            last_mood = msg["mood_score"]
            break
    st.session_state.last_mood = last_mood
if "nudge_shown"   not in st.session_state: st.session_state.nudge_shown   = False
if "last_audio"    not in st.session_state: st.session_state.last_audio    = None
if "voice_reply"   not in st.session_state: st.session_state.voice_reply   = None
if "processed_key" not in st.session_state: st.session_state.processed_key = None


# ── Shared: process one journal entry (text → score → LLM → optionally TTS) ──
def process_entry(user_text: str, auto_speak: bool = False, tts_lang: str = "en", is_voice: bool = False) -> str:
    """Score, save, get LLM reply, optionally generate TTS. Returns reply string."""
    # 1. Sentiment scoring (silent)
    result = se.score_entry(user_text)
    st.session_state.last_mood = result.mood_score

    # 2. Save to DB
    db.save_entry(
        content=user_text,
        mood_score=result.mood_score,
        compound=result.compound,
        sentiment=result.sentiment,
        emotions=", ".join(result.emotions),
    )

    # 3. Build LLM history
    llm_history = [{"role": m["role"], "content": m["content"]}
                   for m in st.session_state.messages]

    # 4. LLM response
    reply = llm.get_llm_response(user_text, llm_history)

    # 5. Optionally generate TTS audio
    audio_bytes = None
    if auto_speak:
        try:
            audio_bytes = ve.speak(reply, lang=tts_lang)
            st.session_state.voice_reply = audio_bytes
        except Exception as e:
            print(f"[MindLog TTS Error] {e}")
            st.session_state.voice_reply = None
            audio_bytes = None

    # 6. Store messages (session state + database for persistence)
    display_user_text = f"🎤 {user_text}" if is_voice else user_text
    user_msg = {"role": "user", "content": display_user_text}
    assistant_msg = {
        "role": "assistant",
        "content": reply,
        "mood_score": result.mood_score,
        "emotions": result.emotions,
        "sentiment": result.sentiment,
        "audio": audio_bytes,
        "audio_lang": tts_lang if audio_bytes else None,
        "autoplay": True if audio_bytes else False,
    }
    st.session_state.messages.append(user_msg)
    st.session_state.messages.append(assistant_msg)

    # Persist to database
    db.save_chat_message(role="user", content=display_user_text)
    db.save_chat_message(
        role="assistant",
        content=reply,
        mood_score=result.mood_score,
        sentiment=result.sentiment,
        emotions=", ".join(result.emotions) if result.emotions else "",
    )

    # 7. Pattern check
    fresh = db.get_all_entries()
    wf = db.get_word_frequency(30)
    pat = pd_engine.analyse(fresh, wf)
    if pat.has_pattern and not st.session_state.nudge_shown:
        st.session_state.nudge_shown = True
        db.log_nudge(pat.nudge_message)

    return reply


# ── Chart builders ────────────────────────────────────────────────────────────
def build_mood_chart(df_daily: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if df_daily.empty:
        fig.add_annotation(
            text="Journal a few entries to see your mood trend!",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(color="#CCD6E0", size=13),
        )
    else:
        fig.add_hrect(y0=5, y1=7, fillcolor="rgba(0,201,177,0.06)", layer="below", line_width=0)
        fig.add_trace(go.Scatter(
            x=df_daily["entry_date"], y=df_daily["avg_mood"],
            mode="lines+markers",
            line=dict(color="#00C9B1", width=2.5, shape="spline"),
            marker=dict(
                size=8,
                color=df_daily["avg_mood"],
                colorscale=[[0,"#FF6B9D"],[0.4,"#FFD600"],[0.7,"#7BC67E"],[1,"#00C9B1"]],
                cmin=1, cmax=10, line=dict(color="#0D1B2A", width=1.5),
            ),
            name="Daily Avg Mood",
            hovertemplate="<b>%{x|%b %d}</b><br>Mood: %{y:.1f}/10<extra></extra>",
        ))
        if len(df_daily) >= 3:
            df_daily = df_daily.copy()
            df_daily["rolling"] = df_daily["avg_mood"].rolling(3, min_periods=1).mean()
            fig.add_trace(go.Scatter(
                x=df_daily["entry_date"], y=df_daily["rolling"],
                mode="lines", line=dict(color="#9B5DE5", width=1.5, dash="dot"),
                name="3-day avg", hoverinfo="skip",
            ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#CCD6E0", family="Segoe UI"),
        xaxis=dict(gridcolor="#1A3A5C", tickfont=dict(size=10), title=""),
        yaxis=dict(gridcolor="#1A3A5C", range=[0.5,10.5], title="Mood",
                   tickfont=dict(size=10), dtick=2),
        legend=dict(font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=10,r=10,t=10,b=10), height=230,
    )
    return fig


def build_weekly_bar(df_weekly: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if df_weekly.empty:
        return fig
    colors = [se.mood_color(v) for v in df_weekly["avg_mood"]]
    fig.add_trace(go.Bar(
        x=df_weekly["week"].dt.strftime("W/C %b %d"),
        y=df_weekly["avg_mood"], marker_color=colors,
        hovertemplate="<b>%{x}</b><br>Avg: %{y:.1f}/10<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#CCD6E0", family="Segoe UI"),
        xaxis=dict(gridcolor="#1A3A5C", tickfont=dict(size=9), title=""),
        yaxis=dict(gridcolor="#1A3A5C", range=[0,11], title="", tickfont=dict(size=9)),
        margin=dict(l=10,r=10,t=10,b=10), height=180,
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Dashboard + Voice Settings
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"## {APP_ICON} {APP_TITLE}")
    st.markdown("*Your private mood journal*")
    st.divider()

    # ── Voice Settings ─────────────────────────────────────────────────────
    st.markdown("### 🎤 Voice Settings")
    voice_auto_speak = st.toggle(
        "Auto-play AI voice reply",
        value=True,
        help="When on, MindLog will speak its response aloud after each entry.",
    )
    LANG_DICT = {
        "en": "English 🇬🇧",
        "hi": "Hindi 🇮🇳",
        "kn": "Kannada 🇮🇳",
        "fr": "French 🇫🇷",
        "es": "Spanish 🇪🇸",
        "de": "German 🇩🇪",
        "ja": "Japanese 🇯🇵",
    }
    tts_lang = st.selectbox(
        "Voice language",
        options=list(LANG_DICT.keys()),
        format_func=lambda x: LANG_DICT.get(x, x),
        index=0,
        help="MindLog translates and speaks its response in this language.",
    )
    st.divider()

    # ── Stats ──────────────────────────────────────────────────────────────
    all_entries = db.get_all_entries()
    total = db.total_entries()
    daily_df = db.get_daily_averages()
    weekly_df = db.get_weekly_averages()

    col1, col2 = st.columns(2)
    col1.metric("📓 Entries", total)
    if not all_entries.empty:
        recent_score = all_entries.tail(1)["mood_score"].values[0]
        prev_score   = all_entries.tail(2).head(1)["mood_score"].values[0] if total > 1 else recent_score
        delta = round(recent_score - prev_score, 1)
        col2.metric("Last Mood", f"{recent_score:.1f}/10",
                    f"{'+' if delta>0 else ''}{delta}", delta_color="normal")
        overall_avg = round(all_entries["mood_score"].mean(), 1)
        st.markdown(f"**Overall Avg:** `{overall_avg}/10` {se.mood_label(overall_avg)}")

    st.divider()

    # ── Charts ─────────────────────────────────────────────────────────────
    st.markdown("**📈 Mood Trend**")
    st.plotly_chart(build_mood_chart(daily_df), width="stretch", key="trend")

    if not weekly_df.empty and len(weekly_df) > 1:
        st.markdown("**📊 Weekly Averages**")
        st.plotly_chart(build_weekly_bar(weekly_df), width="stretch", key="weekly")

    st.divider()

    # ── Pattern insights ────────────────────────────────────────────────────
    if total >= 5:
        wf  = db.get_word_frequency(30)
        pat = pd_engine.analyse(all_entries, wf)
        if pat.day_of_week_insight:
            st.markdown(f'<div class="pattern-insight">📅 {pat.day_of_week_insight}</div>',
                        unsafe_allow_html=True)
        if pat.streak_info:
            st.markdown(f'<div class="pattern-insight">🔍 {pat.streak_info}</div>',
                        unsafe_allow_html=True)
        if pat.recent_avg and pat.baseline_avg:
            st.markdown(f"**Recent 7-day avg:** `{pat.recent_avg}/10` | "
                        f"**Baseline:** `{pat.baseline_avg}/10`")

    if total >= 3:
        wf = db.get_word_frequency(20)
        if wf:
            top = list(wf.items())[:6]
            words_str = " · ".join(f"`{w}` ×{c}" for w, c in top)
            st.markdown(f"**🔡 Common themes:** {words_str}")

    st.divider()
    st.caption("🛡️ *MindLog is a journaling tool, not a mental health service.*")

    # ── New Conversation button ────────────────────────────────────────────
    st.divider()
    if st.button("🗑️ New Conversation", use_container_width=True,
                 help="Clear chat history and start a fresh conversation"):
        db.clear_chat_messages()
        st.session_state.messages = []
        st.session_state.last_mood = None
        st.session_state.nudge_shown = False
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN — Header
# ═══════════════════════════════════════════════════════════════════════════════
hcol, mcol = st.columns([3, 1])
with hcol:
    st.markdown(f"# {APP_ICON} {APP_TITLE}")
    st.markdown("*your AI friend who actually listens 💙*")
with mcol:
    if st.session_state.last_mood is not None:
        score = st.session_state.last_mood
        st.markdown(
            f"<div style='text-align:right;padding-top:8px;'>"
            f"<span style='color:#CCD6E0;font-size:0.85rem;'>Last entry</span><br>"
            f"<span style='color:{se.mood_color(score)};font-size:1.6rem;font-weight:bold;'>"
            f"{score:.1f}/10</span><br>"
            f"<span style='font-size:1.2rem;'>{se.mood_label(score)}</span></div>",
            unsafe_allow_html=True,
        )

st.divider()

# ── Nudge banner ─────────────────────────────────────────────────────────────
if total >= 5 and not st.session_state.nudge_shown:
    ae = db.get_all_entries()
    wf = db.get_word_frequency(30)
    pat = pd_engine.analyse(ae, wf)
    if pat.has_pattern:
        st.session_state.nudge_shown = True
        db.log_nudge(pat.nudge_message)
        st.markdown(f'<div class="nudge-box">💙 <b>A gentle check-in</b><br>{pat.nudge_message}</div>',
                    unsafe_allow_html=True)
        st.markdown(f'<div class="helpline-box">📞 {HELPLINE_TEXT}</div>',
                    unsafe_allow_html=True)

# ── Chat history ──────────────────────────────────────────────────────────────
for idx, msg in enumerate(st.session_state.messages):
    avatar = APP_ICON if msg["role"] == "assistant" else "🧑"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg.get("mood_score"):
            score = msg["mood_score"]
            em_str = " · ".join(msg.get("emotions", [])) or "—"
            st.caption(
                f"Mood: **{score:.1f}/10** {se.mood_label(score)} "
                f"| Sentiment: *{msg.get('sentiment','—')}* "
                f"| Emotions: *{em_str}*"
            )

        # ── Audio voice player & play option for assistant replies ────────────
        if msg["role"] == "assistant":
            audio_data = msg.get("audio")
            audio_lang = msg.get("audio_lang", "en")
            should_autoplay = msg.pop("autoplay", False)

            if audio_data:
                col_aud, col_btn = st.columns([5, 1])
                with col_aud:
                    st.audio(audio_data, format="audio/mp3", autoplay=should_autoplay)
                    if audio_lang and audio_lang != "en":
                        lang_label = LANG_DICT.get(audio_lang, audio_lang)
                        st.caption(f"🗣️ *Spoken in {lang_label}*")
                with col_btn:
                    if st.button("🔄", key=f"re_voice_{idx}", help=f"Re-generate voice in {LANG_DICT.get(tts_lang, tts_lang)}"):
                        with st.spinner(f"Speaking in {LANG_DICT.get(tts_lang, tts_lang)}…"):
                            new_audio = ve.speak(msg["content"], lang=tts_lang)
                            if new_audio:
                                msg["audio"] = new_audio
                                msg["audio_lang"] = tts_lang
                                msg["autoplay"] = True
                                st.rerun()
            else:
                col_btn, _ = st.columns([1, 4])
                with col_btn:
                    lang_label = LANG_DICT.get(tts_lang, "English")
                    btn_text = "🔊 Play Voice" if tts_lang == "en" else f"🔊 Play in {lang_label}"
                    if st.button(btn_text, key=f"play_voice_{idx}", help=f"Listen to this reply in {lang_label}"):
                        with st.spinner(f"Generating voice in {lang_label}…"):
                            new_audio = ve.speak(msg["content"], lang=tts_lang)
                            if new_audio:
                                msg["audio"] = new_audio
                                msg["audio_lang"] = tts_lang
                                msg["autoplay"] = True
                                st.rerun()

# ── Welcome ────────────────────────────────────────────────────────────────────
if not st.session_state.messages:
    welcome_text = (
        "yooo what's good! 😄 welcome to **MindLog** — I'm basically your journal "
        "that talks back lol\n\n"
        "you can **type** whatever's on your mind below, or hit the **🎤 mic tab** "
        "if you're too lazy to type (no judgment 😂)\n\n"
        "**so tell me — how's life treating you today?** ✨"
    )
    with st.chat_message("assistant", avatar=APP_ICON):
        st.markdown(welcome_text)
        if "welcome_audio" in st.session_state and st.session_state.welcome_audio:
            st.audio(st.session_state.welcome_audio, format="audio/mp3",
                     autoplay=st.session_state.pop("welcome_autoplay", False))
        else:
            col_w_btn, _ = st.columns([1, 4])
            with col_w_btn:
                if st.button("🔊 Play Voice", key="play_welcome_voice", help="Listen to welcome message"):
                    with st.spinner("Generating voice…"):
                        wa = ve.speak(welcome_text, lang=tts_lang)
                        if wa:
                            st.session_state.welcome_audio = wa
                            st.session_state.welcome_autoplay = True
                            st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# INPUT AREA — Two tabs: Type  |  Speak
# ═══════════════════════════════════════════════════════════════════════════════
tab_text, tab_voice = st.tabs(["✍️  Type your entry", "🎤  Speak your entry"])

# ────────────────────────────────────────────────────────────────────────────
# TAB 1 — Text input (original chat input)
# ────────────────────────────────────────────────────────────────────────────
with tab_text:
    if prompt := st.chat_input("what's on your mind? spill it 💬",
                                key="text_input"):
        with st.spinner("Thinking…"):
            reply = process_entry(prompt, auto_speak=voice_auto_speak, tts_lang=tts_lang)
        st.rerun()


# ────────────────────────────────────────────────────────────────────────────
# TAB 2 — Voice input
# ────────────────────────────────────────────────────────────────────────────
with tab_voice:
    st.markdown(
        '<div class="voice-active">'
        '<span style="font-size:2.5rem;">🎤</span><br>'
        '<span style="color:#00C9B1;font-weight:600;font-size:1.1rem;">Voice Mode Active</span><br>'
        '<span style="color:#CCD6E0;font-size:0.9rem;">'
        'Click the mic below, speak your journal entry, then click Stop Recording.</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    audio_value = st.audio_input(
        label="🎙️ Hold to record — speak naturally",
        key="mic_input",
    )

    if audio_value is not None:
        # Only process if this is a NEW recording (avoid reprocessing on reruns)
        audio_id = id(audio_value)
        if st.session_state.processed_key != audio_id:
            st.session_state.processed_key = audio_id

            with st.spinner("🔍 Transcribing your voice…"):
                transcript, err = ve.transcribe(audio_value)

            if err:
                st.warning(f"🎤 {err}")
            elif transcript:
                with st.spinner("Thinking…"):
                    reply = process_entry(transcript, auto_speak=True, tts_lang=tts_lang, is_voice=True)
                st.rerun()
    else:
        # Instructions when no audio yet
        st.markdown("""
        <div style="text-align:center; padding: 20px; color: #CCD6E0; opacity: 0.7;">
        <p>💡 <b>Tips for best results:</b></p>
        <p>• Speak clearly and at a normal pace</p>
        <p>• Works best with a quiet background</p>
        <p>• You can say things like:<br>
        <i>"Today was really exhausting, I had back-to-back meetings and couldn't catch a break."</i></p>
        </div>
        """, unsafe_allow_html=True)
