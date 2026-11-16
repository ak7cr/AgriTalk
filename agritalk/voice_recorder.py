"""Browser based voice recorder with a live waveform visualizer."""

import base64
from pathlib import Path

import streamlit.components.v1 as components

_component = components.declare_component(
    "agritalk_voice_recorder", path=str(Path(__file__).with_name("voice_component"))
)


def voice_recorder(start_label: str, stop_label: str, key: str):
    """Return the latest audio capture as bytes, or None before the first capture."""
    value = _component(start_label=start_label, stop_label=stop_label, key=key, default=None)
    if not isinstance(value, dict) or not value.get("audio_base64"):
        return None
    try:
        return {
            "bytes": base64.b64decode(value["audio_base64"]),
            "id": value.get("id"),
            "mime_type": value.get("mime_type", "audio/webm"),
        }
    except (ValueError, TypeError):
        return None
