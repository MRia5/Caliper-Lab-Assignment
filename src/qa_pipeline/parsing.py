from __future__ import annotations

import html
import re
from html.parser import HTMLParser
from pathlib import Path


class FilingHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1
        if tag in {"p", "div", "br", "tr", "table", "h1", "h2", "h3", "li"}:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._skip_depth:
            self._skip_depth -= 1
        if tag in {"p", "div", "tr", "table", "h1", "h2", "h3", "li"}:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            stripped = data.strip()
            if stripped:
                self._parts.append(stripped)
                self._parts.append(" ")

    def text(self) -> str:
        return normalize_text("".join(self._parts))


def load_document(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    if "Your Request Originates from an Undeclared Automated Tool" in raw:
        raise ValueError(
            "The input file is an SEC automated-tool block page, not the filing. "
            "Open the SEC URL in a browser and save the filing HTML into raw_inputs/."
        )
    if looks_like_html(raw):
        parser = FilingHTMLParser()
        parser.feed(raw)
        return parser.text()
    return normalize_text(raw)


def looks_like_html(text: str) -> bool:
    sample = text[:2000].lower()
    return "<html" in sample or "<ix:" in sample or "<document" in sample or "<body" in sample


def normalize_text(text: str) -> str:
    text = html.unescape(text)
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()
