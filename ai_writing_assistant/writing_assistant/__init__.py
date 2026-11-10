"""AI Writing Assistant: prompt templates + OpenAI/Gemini providers."""

from .assistant import GenerationResult, WritingAssistant
from .config import TEMPERATURE_PRESETS, resolve_temperature
from .providers import get_provider
from .templates import TEMPLATES

__all__ = [
    "WritingAssistant",
    "GenerationResult",
    "TEMPLATES",
    "TEMPERATURE_PRESETS",
    "resolve_temperature",
    "get_provider",
]
__version__ = "1.0.0"
