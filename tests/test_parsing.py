from qa_pipeline.parsing import FilingHTMLParser, normalize_text


def test_html_parser_removes_script_and_keeps_text() -> None:
    parser = FilingHTMLParser()
    parser.feed("<html><script>ignore()</script><body><p>Apple Inc.</p><p>Revenue grew.</p></body></html>")
    text = parser.text()
    assert "ignore" not in text
    assert "Apple Inc." in text
    assert "Revenue grew." in text


def test_normalize_text_collapses_whitespace() -> None:
    assert normalize_text("A&nbsp;&nbsp;B\n\n\n C") == "A B\n\nC"

