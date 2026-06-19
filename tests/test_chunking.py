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
    assert chunks[0].topic == "Item 7. Management Discussion"
    assert chunks[0].source_file == "sample.htm"


def test_chunk_document_splits_known_semantic_headings() -> None:
    net_sales = " ".join(["Net sales increased because services and products grew."] * 50)
    gross_margin = " ".join(["Gross margin changed due to product mix and costs."] * 50)
    text = f"Item 7. Management Discussion\nNet Sales {net_sales} Gross Margin {gross_margin}"
    chunks = chunk_document(text, Path("sample.htm"), max_words=120, overlap_words=10)
    titles = {chunk.section_title for chunk in chunks}
    assert "Item 7. Management Discussion > Net Sales" in titles
    assert "Item 7. Management Discussion > Gross Margin" in titles
