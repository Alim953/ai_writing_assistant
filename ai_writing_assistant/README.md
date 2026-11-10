# AI Writing Assistant

A Python (CLI + library) writing assistant backed by **OpenAI** or **Google Gemini**, with
task-specific prompt templates, temperature controls, and output formatting. No frontend.

## Tasks

| Task      | What it does                          | Default temperature |
|-----------|---------------------------------------|---------------------|
| `blog`    | Structured blog post + meta description | 0.8 |
| `email`   | Subject line + ready-to-send email    | 0.5 |
| `explain` | Code explanation at a chosen skill level | 0.2 |
| `social`  | Platform-tailored posts (X/LinkedIn/Instagram/Facebook), multiple variants | 1.0 |

Factual tasks run cool, creative ones run warm. Override any time.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env      # add OPENAI_API_KEY and/or GEMINI_API_KEY
```

Default models are `gpt-4o-mini` and `gemini-2.5-flash`. Model names change often, so set
`OPENAI_MODEL` / `GEMINI_MODEL` in `.env` (or pass `-m`) to whatever your account supports.

## Usage

```bash
python -m writing_assistant templates          # list tasks, fields, defaults

# Blog post with Gemini, creative preset, saved as HTML
python -m writing_assistant blog --topic "Intro to composting" --audience "beginners" \
    --words 600 --keywords "compost, kitchen waste" -p gemini -c creative -f html -o post.html

# Email, exact temperature
python -m writing_assistant email --purpose "Ask for a deadline extension" \
    --recipient "Priya (project manager)" --sender "Sam" --length short -t 0.4

# Explain code from a file or stdin
python -m writing_assistant explain --code-file app.py --language Python --level beginner
cat app.py | python -m writing_assistant explain --level expert --focus "error handling"

# Social posts
python -m writing_assistant social --topic "Our eco bottle launches Friday" \
    --platform twitter --variants 3 --cta "Pre-order now"

# See exactly what would be sent (no API call, no key needed)
python -m writing_assistant blog --topic "Test" --dry-run
```

### Common flags

| Flag | Meaning |
|------|---------|
| `-p/--provider` | `openai` (default), `gemini`, or `mock` (offline) |
| `-m/--model` | Model override |
| `-t/--temperature` | Exact value, 0 to 2 |
| `-c/--creativity` | Preset: `precise`=0.2, `balanced`=0.7, `creative`=1.0, `wild`=1.4 |
| `-f/--format` | `markdown` (default), `plain`, `html`, `json` |
| `-o/--output` | Write to a file |
| `--max-tokens` | Cap generated tokens |
| `--extra` | Extra instructions appended to the prompt |
| `--dry-run` / `--show-meta` | Preview prompts / print model, temperature, usage to stderr |

Temperature precedence: `-t` > `-c` > the task's default.

### Output formats

- **markdown**: as returned by the model.
- **plain**: the model is asked for plain text, and any leftover Markdown is stripped.
- **html**: Markdown converted to escaped HTML (headings, lists, code, emphasis, links).
- **json**: `{"task", "provider", "model", "temperature", "content"}`, handy for piping into other tools.

## As a library

```python
from writing_assistant import WritingAssistant, get_provider

assistant = WritingAssistant(get_provider("gemini"))
result = assistant.generate(
    "blog",
    {"topic": "Intro to composting", "words": "600"},
    creativity="creative",
    output_format="html",
)
print(result.formatted, result.temperature, result.usage)
```

## Adding a template

Add a `PromptTemplate` in `writing_assistant/templates.py` and include it in `TEMPLATES`. Its
`fields` automatically become validated CLI flags, so no other changes are needed.

## Tests

```bash
python -m pytest -q
```

Tests use the built-in `mock` provider, so they need no API keys or network.

## Notes

- Some newer or reasoning-style OpenAI models only accept the default temperature. If you get
  a temperature error, switch models or use a model that supports it.
- On Gemini models that "think", reasoning tokens count toward the output cap. If output is
  cut short, raise `--max-tokens`.
- The OpenAI and Gemini calls were written against their current SDKs but not run against the
  live APIs here (no keys in this environment); the template, formatting, and CLI layers are
  tested with the mock provider.
