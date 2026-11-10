"""Core orchestration: template -> prompt -> provider -> formatted result."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .config import resolve_temperature
from .formatting import FORMAT_INSTRUCTIONS, FORMATS, format_output
from .providers import BaseProvider
from .templates import TEMPLATES, PromptTemplate, TemplateError


@dataclass
class GenerationResult:
    task: str
    provider: str
    model: str
    temperature: float
    output_format: str
    raw: str          # model output as returned
    formatted: str    # raw converted to the requested format
    usage: dict | None = None
    finish_reason: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class WritingAssistant:
    def __init__(self, provider: BaseProvider):
        self.provider = provider

    @staticmethod
    def get_template(task: str) -> PromptTemplate:
        try:
            return TEMPLATES[task]
        except KeyError:
            raise TemplateError(
                f"Unknown task {task!r}. Available: {', '.join(TEMPLATES)}"
            ) from None

    def build_prompts(
        self,
        task: str,
        fields: dict,
        output_format: str = "markdown",
        extra_instructions: str | None = None,
    ) -> tuple[str, str]:
        """Render the (system, user) prompts without calling the API."""
        if output_format not in FORMATS:
            raise ValueError(f"Unknown format {output_format!r}. Choose from: {', '.join(FORMATS)}")
        system, user = self.get_template(task).render(fields)
        system = f"{system}\n\n{FORMAT_INSTRUCTIONS[output_format]}"
        if extra_instructions:
            user = f"{user}\n\nAdditional instructions from the user: {extra_instructions.strip()}"
        return system, user

    def generate(
        self,
        task: str,
        fields: dict,
        *,
        temperature: float | None = None,
        creativity: str | None = None,
        output_format: str = "markdown",
        max_tokens: int | None = None,
        extra_instructions: str | None = None,
    ) -> GenerationResult:
        template = self.get_template(task)
        temp = resolve_temperature(temperature, creativity, default=template.default_temperature)
        system, user = self.build_prompts(task, fields, output_format, extra_instructions)

        resp = self.provider.generate(
            system=system,
            user=user,
            temperature=temp,
            max_tokens=max_tokens or template.default_max_tokens,
        )

        meta = {
            "task": task,
            "provider": self.provider.name,
            "model": resp.model,
            "temperature": temp,
        }
        return GenerationResult(
            task=task,
            provider=self.provider.name,
            model=resp.model,
            temperature=temp,
            output_format=output_format,
            raw=resp.text,
            formatted=format_output(resp.text, output_format, meta),
            usage=resp.usage,
            finish_reason=resp.finish_reason,
        )
