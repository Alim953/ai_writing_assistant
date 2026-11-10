"""Command-line interface. Flags for each task are generated from the template fields."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .assistant import WritingAssistant
from .config import MAX_TEMPERATURE, MIN_TEMPERATURE, TEMPERATURE_PRESETS
from .formatting import FORMATS
from .providers import PROVIDERS, ProviderError, get_provider
from .templates import TEMPLATES, TemplateError


def _temperature(value: str) -> float:
    try:
        t = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not a number") from None
    if not MIN_TEMPERATURE <= t <= MAX_TEMPERATURE:
        raise argparse.ArgumentTypeError(
            f"must be between {MIN_TEMPERATURE} and {MAX_TEMPERATURE}"
        )
    return t


def _common_parent() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    g = p.add_argument_group("model")
    g.add_argument("-p", "--provider", choices=list(PROVIDERS), default="openai")
    g.add_argument("-m", "--model", help="Override the provider's default model")

    t = p.add_argument_group("temperature (default: tuned per task)")
    mx = t.add_mutually_exclusive_group()
    mx.add_argument(
        "-t", "--temperature", type=_temperature, metavar="0-2",
        help="Exact temperature; lower = focused, higher = more varied",
    )
    mx.add_argument(
        "-c", "--creativity", choices=list(TEMPERATURE_PRESETS),
        help="Preset: " + ", ".join(f"{k}={v}" for k, v in TEMPERATURE_PRESETS.items()),
    )

    o = p.add_argument_group("output")
    o.add_argument("-f", "--format", choices=FORMATS, default="markdown", dest="output_format")
    o.add_argument("-o", "--output", metavar="FILE", help="Write the result to a file")
    o.add_argument("--max-tokens", type=int, help="Cap on generated tokens")
    o.add_argument("--extra", metavar="TEXT", help="Extra instructions appended to the prompt")
    o.add_argument("--dry-run", action="store_true", help="Print the prompts; make no API call")
    o.add_argument("--show-meta", action="store_true", help="Print model/temperature/usage to stderr")
    return p


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="writing-assistant",
        description="AI writing assistant: blog posts, emails, code explanations, social posts.",
    )
    sub = parser.add_subparsers(dest="task", required=True, metavar="TASK")
    parent = _common_parent()

    for key, tpl in TEMPLATES.items():
        sp = sub.add_parser(key, parents=[parent], help=tpl.description, description=tpl.description)
        for f in tpl.fields:
            flag = "--" + f.name.replace("_", "-")
            if f.from_file:
                sp.add_argument(flag, dest=f.name, help=f"{f.help} (or use {flag}-file / stdin)")
                sp.add_argument(f"{flag}-file", dest=f"{f.name}_file", metavar="PATH",
                                help=f"Read {f.name} from a file")
                continue
            kw = {"dest": f.name, "choices": f.choices}
            default_note = f" [default: {f.default}]" if f.default else ""
            kw["help"] = f"{f.help}{default_note}"
            if f.required:
                kw["required"] = True
            sp.add_argument(flag, **kw)

    sub.add_parser("templates", help="List tasks, their fields and default temperatures")
    return parser


def _list_templates() -> None:
    for tpl in TEMPLATES.values():
        print(f"{tpl.key:8} {tpl.title}  (default temperature {tpl.default_temperature})")
        for f in tpl.fields:
            req = "required" if f.required else f"default: {f.default}"
            ch = f" choices: {', '.join(f.choices)};" if f.choices else ""
            print(f"    --{f.name.replace('_', '-'):<12} {f.help} [{ch} {req}]".replace("[ ", "["))
        print()
    print("Creativity presets: " + ", ".join(f"{k}={v}" for k, v in TEMPERATURE_PRESETS.items()))


def _collect_fields(args: argparse.Namespace) -> dict:
    tpl = TEMPLATES[args.task]
    fields: dict = {}
    for f in tpl.fields:
        value = getattr(args, f.name, None)
        if f.from_file:
            path = getattr(args, f"{f.name}_file", None)
            if path:
                value = Path(path).read_text(encoding="utf-8")
            elif not value and not sys.stdin.isatty():
                value = sys.stdin.read()
        if value is not None:
            fields[f.name] = value
    return fields


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.task == "templates":
        _list_templates()
        return 0

    try:
        fields = _collect_fields(args)
        # Provider is only constructed when needed so --dry-run works without API keys.
        if args.dry_run:
            assistant = WritingAssistant(get_provider("mock"))
            system, user = assistant.build_prompts(
                args.task, fields, args.output_format, args.extra
            )
            print(f"--- SYSTEM ---\n{system}\n\n--- USER ---\n{user}")
            return 0

        assistant = WritingAssistant(get_provider(args.provider, args.model))
        result = assistant.generate(
            args.task,
            fields,
            temperature=args.temperature,
            creativity=args.creativity,
            output_format=args.output_format,
            max_tokens=args.max_tokens,
            extra_instructions=args.extra,
        )
    except (TemplateError, ProviderError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.output:
        Path(args.output).write_text(result.formatted + "\n", encoding="utf-8")
        print(f"Saved to {args.output}", file=sys.stderr)
    else:
        print(result.formatted)

    if args.show_meta:
        print(
            f"[{result.provider}/{result.model} temperature={result.temperature} "
            f"usage={result.usage} finish={result.finish_reason}]",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
