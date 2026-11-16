"""Tests for the Anthropic chat adapter used by the assistant pipeline."""

from types import SimpleNamespace

from agritalk.llm import AnthropicChatClient


class FakeMessages:
    def __init__(self, blocks):
        self._blocks = blocks
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return SimpleNamespace(content=self._blocks)


class FakeAnthropic:
    def __init__(self, blocks):
        self.messages = FakeMessages(blocks)


def _client(blocks, **kwargs):
    fake = FakeAnthropic(blocks)
    return AnthropicChatClient(api_key="test", client=fake, **kwargs), fake


def test_system_message_is_sent_separately_and_text_is_joined():
    client, fake = _client(
        [
            SimpleNamespace(type="text", text="Use [S1]. "),
            SimpleNamespace(type="tool_use", text="ignored"),
            SimpleNamespace(type="text", text="Consult your KVK."),
        ]
    )

    result = client.chat.completions.create(
        model="claude-haiku-4-5",
        messages=[
            {"role": "system", "content": "Be cautious."},
            {"role": "user", "content": "How do I store wheat?"},
        ],
        temperature=0.0,
    )

    assert result.choices[0].message.content == "Use [S1]. Consult your KVK."
    request = fake.messages.requests[0]
    assert request["system"] == "Be cautious."
    assert request["messages"] == [{"role": "user", "content": "How do I store wheat?"}]
    assert request["temperature"] == 0.0
    assert request["model"] == "claude-haiku-4-5"


def test_no_system_message_omits_system_parameter():
    client, fake = _client([SimpleNamespace(type="text", text="ok")])

    client.chat.completions.create(
        model="claude-haiku-4-5",
        messages=[{"role": "user", "content": "hi"}],
    )

    assert "system" not in fake.messages.requests[0]


def test_max_tokens_defaults_and_override():
    client, fake = _client([SimpleNamespace(type="text", text="ok")], max_tokens=256)
    messages = [{"role": "user", "content": "hi"}]

    client.chat.completions.create(model="claude-haiku-4-5", messages=messages)
    client.chat.completions.create(model="claude-haiku-4-5", messages=messages, max_tokens=64)

    assert fake.messages.requests[0]["max_tokens"] == 256
    assert fake.messages.requests[1]["max_tokens"] == 64


def test_empty_content_returns_empty_string():
    client, _ = _client([])

    result = client.chat.completions.create(
        model="claude-haiku-4-5", messages=[{"role": "user", "content": "hi"}]
    )

    assert result.choices[0].message.content == ""
