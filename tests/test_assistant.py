"""Tests for answer generation paths (sourced and general guidance)."""

from types import SimpleNamespace

import pytest

from agritalk import assistant


class FakeCompletions:
    def __init__(self, content):
        self._content = content
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self._content))]
        )


class FakeClient:
    def __init__(self, content):
        self.chat = SimpleNamespace(completions=FakeCompletions(content))

    @property
    def requests(self):
        return self.chat.completions.requests


def _document():
    return {
        "id": "grain_storage",
        "title": "Grain storage",
        "content": "Dry grain to safe moisture before storage.",
        "topics": ["storage"],
        "source_title": "ICAR",
        "source_url": "https://example.org/icar",
        "score": 0.5,
    }


def test_general_answer_is_requested_without_source_markers():
    client = FakeClient("Store crops in a dry, ventilated place. Confirm locally.")

    answer = assistant.generate_general_answer(
        question="Which crops suit winter in India?",
        response_language="en",
        client=client,
        model="claude-haiku-4-5",
    )

    assert answer == "Store crops in a dry, ventilated place. Confirm locally."
    prompt = client.requests[0]["messages"][-1]["content"]
    assert "No verified knowledge-base source" in prompt
    assert "Which crops suit winter in India?" in prompt


def test_ask_agritalk_uses_general_mode_when_nothing_is_retrieved(monkeypatch):
    monkeypatch.setattr(assistant, "retrieve_documents", lambda *args, **kwargs: [])
    client = FakeClient("General guidance about rabi crops.")

    result = assistant.ask_agritalk(
        question="Which crops suit winter in India?",
        input_language="en",
        response_preference="same",
        client=client,
        model="claude-haiku-4-5",
    )

    assert result["answer_mode"] == "general"
    assert result["answer"] == "General guidance about rabi crops."
    assert result["has_sources"] is False
    assert result["documents"] == []


def test_ask_agritalk_uses_sourced_mode_when_documents_exist(monkeypatch):
    monkeypatch.setattr(assistant, "retrieve_documents", lambda *args, **kwargs: [_document()])
    client = FakeClient("Dry the grain first [S1].")

    result = assistant.ask_agritalk(
        question="How do I store wheat?",
        input_language="en",
        response_preference="same",
        client=client,
        model="claude-haiku-4-5",
    )

    assert result["answer_mode"] == "sourced"
    assert result["answer"] == "Dry the grain first [S1]."
    assert result["has_sources"] is True


@pytest.mark.parametrize("language", ["en", "hi", "ta"])
def test_fallback_used_when_model_returns_nothing(language):
    client = FakeClient("   ")

    answer = assistant.generate_general_answer(
        question="How do I irrigate wheat?",
        response_language=language,
        client=client,
        model="claude-haiku-4-5",
    )

    assert answer == assistant.FALLBACKS[language]
