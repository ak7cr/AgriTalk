"""Speech-to-text and text-to-speech helpers."""

import os
from functools import lru_cache
from io import BytesIO
from typing import Optional

from gtts import gTTS

from agritalk.language import GTTS_LANGUAGE_CODES, WHISPER_LANGUAGE_CODES

WHISPER_MODEL_DEFAULT = "tiny"


def _load_whisper_model(model_size: str):
    # Imported lazily so text-only use does not pay the ctranslate2 startup cost.
    from faster_whisper import WhisperModel

    return WhisperModel(model_size, device="cpu", compute_type="int8")


@lru_cache(maxsize=2)
def _get_whisper_model(model_size: str):
    return _load_whisper_model(model_size)


def transcribe_audio(
    audio_bytes: bytes,
    input_language: str = "auto",
    model_size: Optional[str] = None,
) -> str:
    """Transcribe recorded audio locally with Whisper, hinting at its language.

    The first call downloads the selected model once and reuses it afterwards.
    """
    if not audio_bytes:
        raise ValueError("No audio was supplied. Record a question first.")

    size = model_size or os.getenv("WHISPER_MODEL", WHISPER_MODEL_DEFAULT)
    model = _get_whisper_model(size)
    segments, _info = model.transcribe(
        BytesIO(audio_bytes),
        language=WHISPER_LANGUAGE_CODES.get(input_language),
        vad_filter=True,
    )

    transcript = " ".join(segment.text.strip() for segment in segments).strip()
    if not transcript:
        raise ValueError("Speech recognition returned an empty transcript.")
    return transcript


def make_speech_audio(text: str, language_code: str) -> bytes:
    """Generate an MP3 reply through gTTS for English, Hindi, or Tamil.

    gTTS relies on an internet connection and a third-party service. It is used
    here for an educational prototype, not as a production SLA-backed service.
    """
    clean_text = (text or "").strip()
    if not clean_text:
        raise ValueError("There is no answer text to speak.")

    tts_language = GTTS_LANGUAGE_CODES.get(language_code, "en")
    mp3_buffer = BytesIO()
    speech = gTTS(text=clean_text, lang=tts_language, slow=False)
    speech.write_to_fp(mp3_buffer)
    return mp3_buffer.getvalue()
