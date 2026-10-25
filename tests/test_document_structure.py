"""Heading-classification and table-validation heuristics."""

from app.service.document_processor import DocumentProcessor

BODY = 11.0


def test_body_size_text_is_never_a_heading():
    # Bold at body size is a label/table header, not a heading — this used to
    # explode hierarchy lists (29 "headings" in a 6-page PDF).
    assert DocumentProcessor._heading_level("نکته مهم", BODY, 1.0, BODY) is None
    assert DocumentProcessor._heading_level("متن معمولی", BODY, 0.0, BODY) is None


def test_modest_size_bump_requires_bold():
    assert DocumentProcessor._heading_level("مقدمه", 12.0, 0.0, BODY) is None
    assert DocumentProcessor._heading_level("مقدمه", 12.0, 0.9, BODY) == 2


def test_large_size_is_heading_even_without_bold():
    assert DocumentProcessor._heading_level("فصل اول: کلیات", 16.0, 0.0, BODY) == 1


def test_sentence_like_and_numeric_lines_rejected():
    assert DocumentProcessor._heading_level("این یک جمله کامل است.", 14.0, 1.0, BODY) is None
    assert DocumentProcessor._heading_level("۱۴۰۲/۰۳/۲۵", 14.0, 1.0, BODY) is None
    assert DocumentProcessor._heading_level("x" * 200, 14.0, 1.0, BODY) is None


def test_valid_sparse_table_accepted():
    # Borderless extraction often yields empty cells; 40% fill must pass.
    table = [
        ["ستون ۱", "ستون ۲", "ستون ۳"],
        ["a", None, "c"],
        ["d", "e", ""],
    ]
    assert DocumentProcessor()._is_valid_table(table)


def test_diagram_box_rejected():
    table = [
        ["a", "", "", ""],
        ["", "", "", ""],
        ["", "b", "", ""],
    ]
    assert not DocumentProcessor()._is_valid_table(table)


def test_callout_box_rejected():
    table = [
        ["توضیح", "مقدار"],
        ["متن بسیار طولانی " * 10, "x"],
    ]
    assert not DocumentProcessor()._is_valid_table(table)


def test_single_row_or_column_rejected():
    assert not DocumentProcessor()._is_valid_table([["a", "b"]])
    assert not DocumentProcessor()._is_valid_table([["a"], ["b"]])
    assert not DocumentProcessor()._is_valid_table(None)


def test_rows_to_markdown_pads_ragged_rows_and_escapes_pipes():
    rows = [
        ["نام", "مقدار"],
        ["الف", "1 | 2"],
        ["ب"],
    ]
    markdown = DocumentProcessor._rows_to_markdown(rows)
    lines = markdown.split("\n")
    assert lines[0] == "| نام | مقدار |"
    assert "1 \\| 2" in lines[2]
    assert lines[3].count("|") == lines[0].count("|")


def test_docx_reports_heading_count_separately_from_depth(tmp_path):
    from docx import Document

    doc = Document()
    doc.add_heading("Introduction", level=1)
    doc.add_heading("First topic", level=2)
    doc.add_heading("Second topic", level=2)
    path = tmp_path / "headings.docx"
    doc.save(path)

    _, stats = DocumentProcessor()._process_docx(path, "fixture", path.name)
    assert stats["hierarchy_levels"] == 2
    assert stats["heading_count"] == 3
