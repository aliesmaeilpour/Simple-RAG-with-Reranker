from app.service.document_processor import DocumentProcessor


def test_doc_id_stable_for_same_content_and_name():
    first = DocumentProcessor.generate_doc_id(b"content", "report.pdf")
    second = DocumentProcessor.generate_doc_id(b"content", "report.pdf")
    assert first == second


def test_doc_id_changes_with_content_or_name():
    base = DocumentProcessor.generate_doc_id(b"content", "report.pdf")
    assert DocumentProcessor.generate_doc_id(b"other", "report.pdf") != base
    assert DocumentProcessor.generate_doc_id(b"content", "other.pdf") != base


def test_doc_id_is_short_hex():
    doc_id = DocumentProcessor.generate_doc_id(b"x", "a.pdf")
    assert len(doc_id) == 24
    int(doc_id, 16)  # must be hexadecimal
