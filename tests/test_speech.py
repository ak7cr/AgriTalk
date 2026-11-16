"""Tests for the local Whisper speech-to-text path."""

from types import SimpleNamespace

import pytest

from agritalk import speech


class FakeModel:
    def __init__(self, texts):
        self.texts = texts
        self.calls = []

    def transcribe(self, audio, language=None, vad_filter=False):
        self.calls.append({"language": language, "vad_filter": vad_filter})
        return (SimpleNamespace(text=text) for text in self.texts), SimpleNamespace()


def _patch_model(monkeypatch, texts):
    model = FakeModel(texts)
    used_sizes = []

    def fake_get(size):
        used_sizes.append(size)
        return model

    monkeypatch.setattr(speech, "_get_whisper_model", fake_get)
    return model, used_sizes


def test_empty_audio_is_rejected():
    with pytest.raises(ValueError):
        speech.transcribe_audio(b"")


def test_language_hint_is_forwarded_and_segments_are_joined(monkeypatch):
    model, _ = _patch_model(monkeypatch, [" भीगे अनाज ", "सुखाएँ।"])

    transcript = speech.transcribe_audio(b"audio", input_language="hi", model_size="tiny")

    assert transcript == "भीगे अनाज सुखाएँ।"
    assert model.calls == [{"language": "hi", "vad_filter": True}]


def test_auto_language_sends_no_language_hint(monkeypatch):
    model, _ = _patch_model(monkeypatch, ["hello"])

    speech.transcribe_audio(b"audio", input_language="auto", model_size="tiny")

    assert model.calls[0]["language"] is None


def test_model_size_from_environment_when_not_passed(monkeypatch):
    _, used_sizes = _patch_model(monkeypatch, ["hello"])
    monkeypatch.setenv("WHISPER_MODEL", "base")

    speech.transcribe_audio(b"audio")

    assert used_sizes == ["base"]


def test_blank_transcript_raises(monkeypatch):
    _patch_model(monkeypatch, ["   ", ""])

    with pytest.raises(ValueError):
        speech.transcribe_audio(b"audio", model_size="tiny")
