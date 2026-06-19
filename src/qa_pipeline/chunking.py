from __future__ import annotations

import re
from pathlib import Path

from .models import Chunk


ITEM_RE = re.compile(
    r"(?im)^(item\s+(?:1a|1b|1c|1|2|3|4|5|6|7a|7|8|9a|9b|9c|9|10|11|12|13|14|15|16)\.?\s+.+)$"
)

SEMANTIC_HEADINGS_BY_ITEM = {
    "Item 1.": [
        "Company Background",
        "Products",
        "Services",
        "Segments",
        "Markets and Distribution",
        "Competition",
        "Supply of Components",
        "Research and Development",
        "Intellectual Property",
        "Business Seasonality and Product Introductions",
        "Human Capital",
        "Available Information",
    ],
    "Item 1A.": [
        "Macroeconomic and Industry Risks",
        "Legal and Regulatory Compliance Risks",
        "Technology and Intellectual Property Risks",
        "Business Risks",
        "General Risks",
    ],
    "Item 7.": [
        "Fiscal Period",
        "Macroeconomic Conditions",
        "Segment Operating Performance",
        "Products and Services Performance",
        "Net Sales",
        "Gross Margin",
        "Operating Expenses",
        "Other Income/(Expense), Net",
        "Provision for Income Taxes",
        "Liquidity and Capital Resources",
        "Critical Accounting Estimates",
    ],
    "Item 8.": [
        "Consolidated Statements of Operations",
        "Consolidated Statements of Comprehensive Income",
        "Consolidated Balance Sheets",
        "Consolidated Statements of Shareholders Equity",
        "Consolidated Statements of Cash Flows",
        "Note 1",
        "Note 2 Revenue",
        "Note 3 Earnings Per Share",
        "Note 4 Financial Instruments",
        "Note 5 Consolidated Financial Statement Details",
        "Note 6 Debt",
        "Note 7 Income Taxes",
        "Note 8 Leases",
        "Note 9 Shareholders Equity",
        "Note 10 Share-Based Compensation",
        "Note 11 Commitments, Contingencies and Supply Concentrations",
        "Note 12 Segment Information and Geographic Data",
        "Note 13 Segment Information and Geographic Data",
    ],
}


def chunk_document(
    text: str,
    source_path: Path,
    max_words: int = 900,
    overlap_words: int = 120,
) -> list[Chunk]:
    sections = split_sections(text)
    chunks: list[Chunk] = []

    semantic_sections = split_semantic_sections(sections)

    for section_index, (title, topic, section_text, start_char) in enumerate(semantic_sections):
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
                    topic=topic,
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


def split_semantic_sections(sections: list[tuple[str, str, int]]) -> list[tuple[str, str, str, int]]:
    semantic_sections: list[tuple[str, str, str, int]] = []

    for title, section_text, start_char in sections:
        headings = headings_for_item(title)
        if not headings:
            semantic_sections.append((title, title, section_text, start_char))
            continue

        matches = find_heading_matches(section_text, headings)
        if not matches:
            semantic_sections.append((title, title, section_text, start_char))
            continue

        prefix = section_text[: matches[0][0]].strip()
        if len(prefix.split()) >= 80:
            semantic_sections.append((title, title, prefix, start_char))

        for index, (match_start, heading) in enumerate(matches):
            match_end = matches[index + 1][0] if index + 1 < len(matches) else len(section_text)
            body = section_text[match_start:match_end].strip()
            if len(body.split()) < 80:
                continue
            semantic_sections.append((f"{title} > {heading}", heading, body, start_char + match_start))

    return semantic_sections


def headings_for_item(title: str) -> list[str]:
    for item_prefix, headings in SEMANTIC_HEADINGS_BY_ITEM.items():
        if title.startswith(item_prefix):
            return headings
    return []


def find_heading_matches(text: str, headings: list[str]) -> list[tuple[int, str]]:
    matches: list[tuple[int, str]] = []
    for heading in headings:
        pattern = re.compile(rf"(?<![A-Za-z]){re.escape(heading)}(?![A-Za-z])")
        for match in pattern.finditer(text):
            matches.append((match.start(), heading))

    matches.sort(key=lambda item: item[0])
    deduped: list[tuple[int, str]] = []
    last_position = -1
    for position, heading in matches:
        if position == last_position:
            continue
        deduped.append((position, heading))
        last_position = position
    return deduped


def clean_section_title(title: str) -> str:
    return re.sub(r"\s+", " ", title).strip(" .")
