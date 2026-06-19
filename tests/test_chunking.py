from pathlib import Path

from qa_pipeline.chunking import chunk_document, split_sections


def test_split_sections_uses_item_headings() -> None:
    text = """
Item 1. Business
Apple sells products and services.

Item 1A. Risk Factors
The company faces market and supply risks.
"""
    sections = split_sections(text)
    assert [section[0] for section in sections] == ["Item 1. Business", "Item 1A. Risk Factors"]


def test_chunk_document_keeps_section_titles() -> None:
    body = " ".join(["Revenue increased because services grew."] * 120)
    text = f"Item 7. Management Discussion\n{body}"
    chunks = chunk_document(text, Path("sample.htm"), max_words=100, overlap_words=10)
    assert chunks
    assert chunks[0].section_title == "Item 7. Management Discussion"
    assert chunks[0].source_file == "sample.htm"

