"""LLM provider adapters. Each exposes the same ``generate`` interface."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .config import DEFAULT_MODELS


class ProviderError(RuntimeError):
    """Raised for configuration problems or failed API calls."""


@dataclass
class ProviderResponse:
    text: str
    model: str
    usage: dict | None = None
    finish_reason: str | None = None


class BaseProvider(ABC):
    name: str = "base"

    def __init__(self, model: str | None = None):
        self.model = model or DEFAULT_MODELS[self.name]

    @abstractmethod
    def generate(
        self,
        system: str,
        user: str,
        temperature: float,
        max_tokens: int | None = None,
    ) -> ProviderResponse: ...


class OpenAIProvider(BaseProvider):
    name = "openai"

    def __init__(self, model: str | None = None, api_key: str | None = None):
        super().__init__(model)
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ProviderError("OPENAI_API_KEY is not set (see .env.example).")
        try:
            from openai import OpenAI
        except ImportError:
            raise ProviderError("Install the OpenAI SDK: pip install openai") from None
        self._client = OpenAI(api_key=api_key)

    def generate(self, system, user, temperature, max_tokens=None) -> ProviderResponse:
        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
        }
        if max_tokens:
            kwargs["max_completion_tokens"] = max_tokens
        try:
            resp = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            raise ProviderError(f"OpenAI request failed: {exc}") from exc

        choice = resp.choices[0]
        usage = None
        if resp.usage:
            usage = {
                "input_tokens": resp.usage.prompt_tokens,
                "output_tokens": resp.usage.completion_tokens,
            }
        return ProviderResponse(
            text=(choice.message.content or "").strip(),
            model=resp.model or self.model,
            usage=usage,
            finish_reason=choice.finish_reason,
        )


class GeminiProvider(BaseProvider):
    name = "gemini"

    def __init__(self, model: str | None = None, api_key: str | None = None):
        super().__init__(model)
        api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ProviderError("GEMINI_API_KEY (or GOOGLE_API_KEY) is not set.")
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise ProviderError("Install the Gemini SDK: pip install google-genai") from None
        self._types = types
        self._client = genai.Client(api_key=api_key)

    def generate(self, system, user, temperature, max_tokens=None) -> ProviderResponse:
        config = self._types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        try:
            resp = self._client.models.generate_content(
                model=self.model, contents=user, config=config
            )
        except Exception as exc:
            raise ProviderError(f"Gemini request failed: {exc}") from exc

        try:
            text = (resp.text or "").strip()
        except Exception:  # blocked / empty candidates
            text = ""
        if not text:
            raise ProviderError("Gemini returned no text (response may have been blocked).")

        usage = None
        meta = getattr(resp, "usage_metadata", None)
        if meta:
            usage = {
                "input_tokens": getattr(meta, "prompt_token_count", None),
                "output_tokens": getattr(meta, "candidates_token_count", None),
            }
        finish = None
        if resp.candidates:
            finish = str(getattr(resp.candidates[0], "finish_reason", None))
        return ProviderResponse(text=text, model=self.model, usage=usage, finish_reason=finish)


class MockProvider(BaseProvider):
    """Offline provider for tests and demos. Echoes what it was asked to do."""

    name = "mock"

    def __init__(self, model: str | None = None, **_):
        super().__init__(model)
        self.calls: list[dict] = []

    def generate(self, system, user, temperature, max_tokens=None) -> ProviderResponse:
        self.calls.append(
            {"system": system, "user": user, "temperature": temperature, "max_tokens": max_tokens}
        )
        text = (
            "# Mock Output\n\n"
            f"This is a **mock** response generated at temperature {temperature}.\n\n"
            "- First point\n- Second point\n\n"
            "Visit [Example](https://example.com) for more."
        )
        return ProviderResponse(
            text=text,
            model=self.model,
            usage={"input_tokens": len(system + user) // 4, "output_tokens": len(text) // 4},
            finish_reason="stop",
        )


PROVIDERS = {"openai": OpenAIProvider, "gemini": GeminiProvider, "mock": MockProvider}


def get_provider(name: str, model: str | None = None) -> BaseProvider:
    key = name.lower()
    if key not in PROVIDERS:
        raise ProviderError(f"Unknown provider {name!r}. Choose from: {', '.join(PROVIDERS)}")
    return PROVIDERS[key](model=model)
