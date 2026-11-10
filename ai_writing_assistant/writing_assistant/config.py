"""Configuration: env loading, default models, temperature presets."""

from __future__ import annotations

import os

try:  # optional dependency
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

# Override these with OPENAI_MODEL / GEMINI_MODEL in your environment or .env
# (model names change often, so they are deliberately not hard-wired).
DEFAULT_MODELS = {
    "openai": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    "gemini": os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
    "mock": "mock-1",
}

MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 2.0  # valid range for both OpenAI and Gemini

# Named presets so users don't have to guess numbers.
TEMPERATURE_PRESETS = {
    "precise": 0.2,   # factual, consistent, minimal variation
    "balanced": 0.7,  # good general-purpose default
    "creative": 1.0,  # more varied wording and ideas
    "wild": 1.4,      # experimental; may drift or get odd
}


def validate_temperature(value: float) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Temperature must be a number, got {value!r}") from None
    if not MIN_TEMPERATURE <= value <= MAX_TEMPERATURE:
        raise ValueError(
            f"Temperature must be between {MIN_TEMPERATURE} and {MAX_TEMPERATURE}, got {value}"
        )
    return value


def resolve_temperature(
    temperature: float | None = None,
    creativity: str | None = None,
    default: float = TEMPERATURE_PRESETS["balanced"],
) -> float:
    """Pick the final temperature.

    Precedence: explicit ``temperature`` > ``creativity`` preset > ``default``
    (usually the template's own default).
    """
    if temperature is not None:
        return validate_temperature(temperature)
    if creativity is not None:
        key = creativity.lower()
        if key not in TEMPERATURE_PRESETS:
            raise ValueError(
                f"Unknown creativity preset {creativity!r}. "
                f"Choose from: {', '.join(TEMPERATURE_PRESETS)}"
            )
        return TEMPERATURE_PRESETS[key]
    return validate_temperature(default)
