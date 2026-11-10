import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from writing_assistant import WritingAssistant, get_provider, resolve_temperature
from writing_assistant.cli import main
from writing_assistant.formatting import format_output, to_html, to_plain
from writing_assistant.providers import ProviderError
from writing_assistant.templates import TEMPLATES, TemplateError

SAMPLES = {
    "blog": {"topic": "remote work"},
    "email": {"purpose": "request a meeting"},
    "explain": {"code": "print('hi')", "language": "Python"},
    "social": {"topic": "our launch"},
}


# ---- temperature -----------------------------------------------------------
def test_temperature_precedence():
    assert resolve_temperature(0.3, "wild", default=0.9) == 0.3
    assert resolve_temperature(None, "precise", default=0.9) == 0.2
    assert resolve_temperature(None, None, default=0.9) == 0.9


@pytest.mark.parametrize("bad", [-0.1, 2.1, "abc"])
def test_temperature_rejects_bad_values(bad):
    with pytest.raises(ValueError):
        resolve_temperature(bad)


def test_unknown_preset():
    with pytest.raises(ValueError):
        resolve_temperature(None, "nope")


# ---- templates -------------------------------------------------------------
@pytest.mark.parametrize("task", SAMPLES)
def test_every_template_renders_without_leftover_placeholders(task):
    system, user = TEMPLATES[task].render(SAMPLES[task])
    import re
    assert not re.search(r"\{[a-z_]+\}", (system + user).split("````")[0])
    assert system and user


def test_required_field_enforced():
    with pytest.raises(TemplateError):
        TEMPLATES["blog"].render({})


def test_choice_validation_and_unknown_field():
    with pytest.raises(TemplateError):
        TEMPLATES["social"].render({"topic": "x", "platform": "myspace"})
    with pytest.raises(TemplateError):
        TEMPLATES["blog"].render({"topic": "x", "bogus": "y"})


def test_braces_in_user_input_are_safe():
    _, user = TEMPLATES["explain"].render({"code": "d = {'a': 1}\nf'{x}'"})
    assert "d = {'a': 1}" in user and "f'{x}'" in user


def test_platform_rules_applied():
    _, user = TEMPLATES["social"].render({"topic": "x", "platform": "twitter"})
    assert "280" in user


# ---- formatting ------------------------------------------------------------
MD = "# Title\n\nSome **bold** and *italic* with [a link](https://x.com).\n\n- one\n- two\n\n```\ncode <b>\n```"


def test_plain_strips_markdown():
    out = to_plain(MD)
    assert "#" not in out and "**" not in out and "```" not in out
    assert "a link (https://x.com)" in out and "• one" in out


def test_html_conversion_and_escaping():
    out = to_html(MD + "\n\n<script>alert(1)</script>")
    assert "<h1>Title</h1>" in out
    assert "<strong>bold</strong>" in out and "<em>italic</em>" in out
    assert '<a href="https://x.com">a link</a>' in out
    assert "<ul>" in out and "<li>one</li>" in out
    assert "&lt;b&gt;" in out
    assert "<script>" not in out


def test_json_envelope():
    out = json.loads(format_output("hello", "json", {"task": "blog"}))
    assert out == {"task": "blog", "content": "hello"}


def test_unknown_format():
    with pytest.raises(ValueError):
        format_output("x", "pdf")


# ---- assistant end to end (mock provider) ----------------------------------
def test_generate_uses_template_default_temperature():
    provider = get_provider("mock")
    a = WritingAssistant(provider)
    a.generate("explain", SAMPLES["explain"])
    a.generate("social", SAMPLES["social"])
    assert provider.calls[0]["temperature"] == TEMPLATES["explain"].default_temperature
    assert provider.calls[1]["temperature"] == TEMPLATES["social"].default_temperature
    assert provider.calls[0]["temperature"] < provider.calls[1]["temperature"]


def test_overrides_and_format_instruction():
    provider = get_provider("mock")
    a = WritingAssistant(provider)
    r = a.generate("blog", SAMPLES["blog"], creativity="creative", output_format="plain",
                   max_tokens=123, extra_instructions="Keep it spicy")
    call = provider.calls[0]
    assert r.temperature == 1.0 and call["max_tokens"] == 123
    assert "plain text" in call["system"]
    assert "Keep it spicy" in call["user"]
    assert "#" not in r.formatted


def test_unknown_task():
    with pytest.raises(TemplateError):
        WritingAssistant(get_provider("mock")).generate("poem", {})


def test_missing_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(ProviderError):
        get_provider("openai")
    with pytest.raises(ProviderError):
        get_provider("gemini")


# ---- CLI -------------------------------------------------------------------
def test_cli_mock_run(capsys):
    assert main(["blog", "--topic", "tea", "-p", "mock", "-f", "html", "-c", "precise"]) == 0
    assert "<h1>Mock Output</h1>" in capsys.readouterr().out


def test_cli_dry_run_needs_no_key(capsys, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert main(["email", "--purpose", "follow up", "--length", "short", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "--- SYSTEM ---" in out and "3-5 sentences" in out


def test_cli_code_file_and_output_file(tmp_path, capsys):
    src = tmp_path / "a.py"
    src.write_text("x = 1")
    dest = tmp_path / "out.json"
    rc = main(["explain", "--code-file", str(src), "-p", "mock", "-f", "json", "-o", str(dest)])
    assert rc == 0
    data = json.loads(dest.read_text())
    assert data["task"] == "explain" and "content" in data


def test_cli_errors_are_clean(capsys, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert main(["social", "--topic", "x", "-p", "openai"]) == 1
    assert "OPENAI_API_KEY" in capsys.readouterr().err
    assert main(["explain", "-p", "mock"]) == 1  # no code supplied (stdin is not a tty in pytest)
    assert "Error" in capsys.readouterr().err


def test_cli_rejects_out_of_range_temperature():
    with pytest.raises(SystemExit):
        main(["blog", "--topic", "x", "-p", "mock", "-t", "3"])
