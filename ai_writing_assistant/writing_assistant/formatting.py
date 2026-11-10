"""Output formatting: markdown (as-is), plain text, HTML, or a JSON envelope."""

from __future__ import annotations

import html
import json
import re

FORMATS = ("markdown", "plain", "html", "json")

# Instruction appended to the system prompt so the model writes in a suitable style.
FORMAT_INSTRUCTIONS = {
    "markdown": "Format the response in Markdown.",
    "plain": "Respond in plain text only: no Markdown symbols such as #, **, or backticks.",
    "html": "Format the response in Markdown (it will be converted to HTML afterwards).",
    "json": "Format the response in Markdown.",
}


def to_plain(text: str) -> str:
    """Strip Markdown syntax, keeping the readable content."""
    text = re.sub(r"^```[^\n]*\n?|^````[^\n]*\n?", "", text, flags=re.M)  # code fences
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.M)                     # headings
    text = re.sub(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", r"\1 (\2)", text) # links
    text = re.sub(r"(\*\*|__)(.+?)\1", r"\2", text)                        # bold
    text = re.sub(r"(?<![\*\w])\*(?!\s)(.+?)(?<!\s)\*(?!\*)", r"\1", text) # italics
    text = re.sub(r"`([^`]+)`", r"\1", text)                               # inline code
    text = re.sub(r"^\s*[-*+]\s+", "• ", text, flags=re.M)                 # bullets
    return text.strip()


def _inline(text: str) -> str:
    text = html.escape(text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)", r"<em>\1</em>", text)
    text = re.sub(
        r"\[([^\]]+)\]\((https?://[^)\s]+)\)", r'<a href="\2">\1</a>', text
    )
    return text


def to_html(text: str) -> str:
    """Minimal, dependency-free Markdown -> HTML (headings, lists, code, emphasis, links)."""
    out: list[str] = []
    para: list[str] = []
    code: list[str] = []
    in_code = False
    list_tag: str | None = None

    def flush_para():
        if para:
            out.append(f"<p>{_inline(' '.join(para))}</p>")
            para.clear()

    def close_list():
        nonlocal list_tag
        if list_tag:
            out.append(f"</{list_tag}>")
            list_tag = None

    def open_list(tag: str):
        nonlocal list_tag
        if list_tag != tag:
            close_list()
            out.append(f"<{tag}>")
            list_tag = tag

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            if in_code:
                out.append(f"<pre><code>{html.escape(chr(10).join(code))}</code></pre>")
                code.clear()
                in_code = False
            else:
                flush_para()
                close_list()
                in_code = True
            continue
        if in_code:
            code.append(line)
            continue
        if not stripped:
            flush_para()
            close_list()
            continue
        if m := re.match(r"^(#{1,6})\s+(.*)$", stripped):
            flush_para()
            close_list()
            n = len(m.group(1))
            out.append(f"<h{n}>{_inline(m.group(2))}</h{n}>")
        elif m := re.match(r"^[-*+]\s+(.*)$", stripped):
            flush_para()
            open_list("ul")
            out.append(f"<li>{_inline(m.group(1))}</li>")
        elif m := re.match(r"^\d+[.)]\s+(.*)$", stripped):
            flush_para()
            open_list("ol")
            out.append(f"<li>{_inline(m.group(1))}</li>")
        else:
            close_list()
            para.append(stripped)

    if in_code:  # unterminated fence
        out.append(f"<pre><code>{html.escape(chr(10).join(code))}</code></pre>")
    flush_para()
    close_list()
    return "\n".join(out)


def format_output(text: str, fmt: str, meta: dict | None = None) -> str:
    fmt = fmt.lower()
    if fmt not in FORMATS:
        raise ValueError(f"Unknown format {fmt!r}. Choose from: {', '.join(FORMATS)}")
    if fmt == "markdown":
        return text
    if fmt == "plain":
        return to_plain(text)
    if fmt == "html":
        return to_html(text)
    return json.dumps({**(meta or {}), "content": text}, indent=2, ensure_ascii=False)
