from __future__ import annotations

import re
from pathlib import Path

from .models import Chunk


ITEM_RE = re.compile(
    r"(?im)^(item\s+(?:1a|1b|1c|1|2|3|4|5|6|7a|7|8|9a|9b|9c|9|10|11|12|13|14|15|16)\.?\s+.+)$"
)


def chunk_document(
    text: str,
    source_path: Path,
    max_words: int = 900,
    overlap_words: int = 120,
) -> list[Chunk]:
    sections = split_sections(text)
    chunks: list[Chunk] = []

    for section_index, (title, section_text, start_char) in enumerate(sections):
        words = section_text.split()
        if not words:
            continue

        step = max(1, max_words - overlap_words)
        part_index = 0
        for word_start in range(0, len(words), step):
            word_end = min(len(words), word_start + max_words)
            chunk_text = " ".join(words[word_start:word_end]).strip()
            if len(chunk_text.split()) < 80:
                continue
            chunk_id = f"chunk-{section_index:03d}-{part_index:03d}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    source_file=source_path.name,
                    section_title=title,
                    section_index=section_index,
                    text=chunk_text,
                    start_char=start_char,
                    end_char=start_char + len(section_text),
                )
            )
            part_index += 1
            if word_end == len(words):
                break

    return chunks


def split_sections(text: str) -> list[tuple[str, str, int]]:
    matches = list(ITEM_RE.finditer(text))
    if not matches:
        return [("Full document", text, 0)]

    sections: list[tuple[str, str, int]] = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        title = clean_section_title(match.group(1))
        body = text[start:end].strip()
        sections.append((title, body, start))
    return sections


def clean_section_title(title: str) -> str:
    return re.sub(r"\s+", " ", title).strip(" .")

