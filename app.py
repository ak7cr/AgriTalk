"""Agri Talk: multilingual voice-first agricultural assistant."""

import html
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from agritalk.assistant import ask_agritalk
from agritalk.i18n import get_ui
from agritalk.language import LANGUAGE_NAMES
from agritalk.llm import DEFAULT_MODEL, AnthropicChatClient
from agritalk.speech import WHISPER_MODEL_DEFAULT, make_speech_audio, transcribe_audio
from agritalk.voice_recorder import voice_recorder

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

st.set_page_config(
    page_title="Agri Talk | Agricultural Voice Assistant",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)


def get_setting(name: str, default: str = "") -> str:
    """Read a setting from Streamlit secrets first, then environment variables."""
    try:
        value = st.secrets.get(name)
    except Exception:
        value = None
    return str(value or os.getenv(name, default)).strip()


def has_real_key(value: str) -> bool:
    """Reject empty values and the placeholder shipped in .env.example."""
    return bool(value) and not value.startswith("replace_with")


def render_answer_header(label: str, tooltip: str = "") -> None:
    """Render the answer heading. A yellow dot marks an unsourced answer; the
    tooltip explains it on hover."""
    if not tooltip:
        st.subheader(label)
        return

    escaped = html.escape(tooltip, quote=True)
    st.markdown(
        """
        <style>
        .agritalk-answer-header { display: flex; align-items: center; gap: .5rem; margin-bottom: .5rem; }
        .agritalk-answer-header h3 { margin: 0; padding: 0; font-size: 1.5rem; font-weight: 600; line-height: 1.3; }
        .agritalk-unsourced-dot { width: 10px; height: 10px; border-radius: 50%; background: #F5C518;
            box-shadow: 0 0 0 1px rgba(0, 0, 0, .25); cursor: help; flex: none; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="agritalk-answer-header"><h3>{html.escape(label)}</h3>'
        f'<span class="agritalk-unsourced-dot" role="img" title="{escaped}" aria-label="{escaped}"></span></div>',
        unsafe_allow_html=True,
    )


def language_display(code: str, ui_text: dict) -> str:
    labels = {
        "auto": ui_text["auto_detect"],
        "same": ui_text["same_as_question"],
        "en": ui_text["english"],
        "hi": ui_text["hindi"],
        "ta": ui_text["tamil"],
    }
    return labels.get(code, code)


def process_question(question: str, input_language: str, response_preference: str, ui_text: dict):
    # Remove the previous result before starting a new request so an error cannot
    # leave a stale answer visible below the new question.
    st.session_state.pop("agritalk_last_result", None)
    st.session_state.pop("agritalk_last_origin", None)
    api_key = get_setting("ANTHROPIC_API_KEY")
    if not has_real_key(api_key):
        st.error(ui_text["key_missing"])
        return

    model = get_setting("ANTHROPIC_MODEL", DEFAULT_MODEL)
    client = AnthropicChatClient(api_key=api_key)
    try:
        with st.spinner(ui_text["working"]):
            result = ask_agritalk(
                question=question,
                input_language=input_language,
                response_preference=response_preference,
                client=client,
                model=model,
            )
        # Every answer has text. Voice playback is best-effort; a TTS outage
        # should not erase an otherwise successful answer.
        try:
            result["audio_bytes"] = make_speech_audio(
                result["answer"], result["response_language"]
            )
            result["audio_error"] = None
        except Exception:
            result["audio_bytes"] = None
            result["audio_error"] = ui_text["audio_error"]

        st.session_state["agritalk_last_result"] = result
        st.session_state["agritalk_last_origin"] = "voice" if st.session_state.get("agritalk_current_origin") == "voice" else "text"
        st.session_state.pop("agritalk_last_error", None)
    except Exception as exc:
        # Show a useful error without exposing any secret values.
        st.session_state["agritalk_last_error"] = f"{ui_text['answer_error']}\n\nTechnical detail: {type(exc).__name__}: {exc}"


# Top-level interface language is independent of the language used for a question.
st.sidebar.title("🌱 Agri Talk")
interface_options = {"English": "en", "हिन्दी": "hi", "தமிழ்": "ta"}
selected_interface = st.sidebar.selectbox(
    "Interface language / இடைமுக மொழி / इंटरफ़ेस की भाषा",
    options=list(interface_options.keys()),
    index=0,
    key="agritalk_interface_language",
)
ui_language = interface_options[selected_interface]
t = get_ui(ui_language)

st.sidebar.markdown("---")
question_language = st.sidebar.selectbox(
    t["question_language"],
    options=["auto", "en", "hi", "ta"],
    index=0,
    format_func=lambda code: language_display(code, t),
    key="agritalk_question_language",
)
response_preference = st.sidebar.selectbox(
    t["answer_language"],
    options=["same", "en", "hi", "ta"],
    index=0,
    format_func=lambda code: language_display(code, t),
    key="agritalk_response_language",
)

with st.sidebar.expander(t["about_title"], expanded=False):
    st.write(t["about_body"])
with st.sidebar.expander(t["try_examples"], expanded=False):
    st.markdown(f"- {t['example_storage']}")
    st.markdown(f"- {t['example_soil']}")
    st.markdown(f"- {t['example_pest']}")
    st.markdown(f"- {t['example_wheat']}")
    st.markdown(f"- {t['example_water']}")

st.title("🌱 " + t["title"])
st.markdown(t["subtitle"])
st.caption(t["privacy_note"])

voice_tab, text_tab = st.tabs([t["voice_tab"], t["text_tab"]])

with voice_tab:
    st.write(t["voice_instructions"])
    st.caption(t["recording_limit"])
    st.caption(t["whisper_note"])
    capture = voice_recorder(
        start_label=t["record_start"],
        stop_label=t["record_stop"],
        key="agritalk_mic_recorder",
    )
    if capture and capture.get("bytes"):
        if capture.get("id") != st.session_state.get("agritalk_voice_capture_id"):
            st.session_state["agritalk_voice_capture_id"] = capture.get("id")
            st.session_state["agritalk_voice_audio"] = capture["bytes"]
            st.session_state["agritalk_voice_mime_type"] = capture.get("mime_type", "audio/webm")
            st.session_state.pop("agritalk_voice_transcript", None)

    voice_audio = st.session_state.get("agritalk_voice_audio")
    if voice_audio:
        st.audio(voice_audio, format=st.session_state.get("agritalk_voice_mime_type", "audio/webm"))
        st.caption(t["voice_ready"])
        if st.button(t["convert_speech"], key="agritalk_convert_speech"):
            try:
                stt_model = get_setting("WHISPER_MODEL", WHISPER_MODEL_DEFAULT)
                with st.spinner(t["transcribing"]):
                    st.session_state["agritalk_voice_transcript"] = transcribe_audio(
                        voice_audio,
                        input_language=question_language,
                        model_size=stt_model,
                    )
                st.session_state.pop("agritalk_last_error", None)
            except Exception as exc:
                st.session_state["agritalk_last_error"] = (
                    f"{t['speech_error']}\n\nTechnical detail: {type(exc).__name__}: {exc}"
                )

        if "agritalk_voice_transcript" in st.session_state:
            transcript = st.text_area(
                t["transcript"],
                key="agritalk_voice_transcript",
                height=100,
            )
            if st.button(t["ask_voice"], type="primary", key="agritalk_ask_voice"):
                st.session_state["agritalk_current_origin"] = "voice"
                st.session_state["agritalk_last_transcript"] = transcript
                process_question(transcript, question_language, response_preference, t)

with text_tab:
    st.write(t["text_instructions"])
    with st.form("agritalk_text_question_form", clear_on_submit=False):
        typed_question = st.text_area(
            t["question_label"],
            placeholder=t["question_placeholder"],
            height=110,
            key="agritalk_typed_question",
        )
        submitted = st.form_submit_button(t["ask_button"], type="primary", use_container_width=True)
    if submitted:
        st.session_state["agritalk_current_origin"] = "text"
        st.session_state.pop("agritalk_last_transcript", None)
        process_question(typed_question, question_language, response_preference, t)

if st.session_state.get("agritalk_last_error"):
    st.error(st.session_state["agritalk_last_error"])

last_result = st.session_state.get("agritalk_last_result")
if last_result:
    st.markdown("---")
    if st.session_state.get("agritalk_last_origin") == "voice":
        st.subheader(t["transcript"])
        st.info(last_result["question"])

    unsourced = last_result.get("answer_mode") == "general"
    render_answer_header(t["answer"], t["general_notice"] if unsourced else "")
    st.markdown(last_result["answer"])

    if last_result.get("audio_bytes"):
        st.caption(t["audio_caption"])
        st.audio(last_result["audio_bytes"], format="audio/mp3")
    elif last_result.get("audio_error"):
        st.info(last_result["audio_error"])

    docs = last_result.get("documents", [])
    st.subheader(t["sources"])
    if docs:
        for index, document in enumerate(docs, start=1):
            with st.expander(f"[S{index}] {document['title']}  ·  {t['relevance']}: {document['score']}"):
                st.write(document["content"])
                st.markdown(f"**{t['source_label']}:** [{document['source_title']}]({document['source_url']})")
    else:
        st.info(t["no_sources"])

    with st.expander(t["disclaimer_title"], expanded=False):
        st.write(t["disclaimer_body"])

    if st.button(t["clear_button"], key="agritalk_clear_answer"):
        st.session_state.pop("agritalk_last_result", None)
        st.session_state.pop("agritalk_last_origin", None)
        st.session_state.pop("agritalk_last_transcript", None)
        st.session_state.pop("agritalk_last_error", None)
        st.rerun()

st.markdown("---")
st.caption(t["language_hint"])
st.caption("Agri Talk student prototype · Built with Streamlit, Claude (Anthropic), local Whisper, RAG retrieval and gTTS")
