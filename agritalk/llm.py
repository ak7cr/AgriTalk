"""Chat model client for the Anthropic Messages API.

The assistant pipeline expects an OpenAI-style
``client.chat.completions.create(...)`` call that returns
``response.choices[0].message.content``. This small adapter keeps that shape on
top of Anthropic so the pipeline does not need provider-specific branches.
"""

from typing import Any, Dict, List, Optional, Protocol

from anthropic import Anthropic

DEFAULT_MODEL = "claude-haiku-4-5"
DEFAULT_MAX_TOKENS = 1024

CHAT_ROLES = {"user", "assistant"}


class ChatClient(Protocol):
    """Anything exposing ``chat.completions.create`` like the Groq SDK."""

    chat: Any


class _Message:
    def __init__(self, content: str) -> None:
        self.content = content


class _Choice:
    def __init__(self, message: _Message) -> None:
        self.message = message


class _Completion:
    def __init__(self, content: str) -> None:
        self.choices = [_Choice(_Message(content))]


class _Completions:
    def __init__(self, client: Anthropic, max_tokens: int) -> None:
        self._client = client
        self._max_tokens = max_tokens

    def create(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
    ) -> _Completion:
        system_parts = [
            message.get("content", "")
            for message in messages
            if message.get("role") == "system"
        ]
        chat_messages = [
            {"role": message["role"], "content": message.get("content", "")}
            for message in messages
            if message.get("role") in CHAT_ROLES
        ]

        request: Dict[str, Any] = {
            "model": model,
            "messages": chat_messages,
            "temperature": temperature,
            "max_tokens": max_tokens or self._max_tokens,
        }
        if system_parts:
            request["system"] = "\n\n".join(system_parts)

        response = self._client.messages.create(**request)
        text = "".join(
            block.text
            for block in response.content
            if getattr(block, "type", None) == "text"
        )
        return _Completion(text)


class _Chat:
    def __init__(self, completions: _Completions) -> None:
        self.completions = completions


class AnthropicChatClient:
    """OpenAI-style chat client backed by Anthropic Claude."""

    def __init__(
        self,
        api_key: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        client: Optional[Anthropic] = None,
    ) -> None:
        anthropic_client = client or Anthropic(api_key=api_key)
        self.chat = _Chat(_Completions(anthropic_client, max_tokens))
