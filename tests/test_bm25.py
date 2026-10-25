import pytest

from app.retrieval.bm25 import BM25Encoder


def test_tokenize_normalizes_persian_digits_and_letters():
    # Arabic yeh/kaf become Persian; Persian and Arabic-Indic digits become ASCII
    tokens = BM25Encoder.tokenize("سال۱۴۰۲ ي ك ٥")
    assert tokens == ["سال1402", "ی", "ک", "5"]


def test_tokenize_splits_zwnj_and_strips_diacritics():
    # ZWNJ becomes a space; the fathatan diacritic is removed
    tokens = BM25Encoder.tokenize("می‌کردم مدرسهٌ")
    assert tokens == ["می", "کردم", "مدرسه"]


def test_tokenize_empty_and_punctuation_only():
    assert BM25Encoder.tokenize("") == []
    assert BM25Encoder.tokenize("!!! ... ، ،") == []


def test_encode_query_dedupes_tokens():
    query = BM25Encoder().encode_query("سلام سلام")
    assert len(query.indices) == len(query.values) == 1
    assert query.values == [1.0]


def test_encode_document_empty_text():
    vector = BM25Encoder().encode_document("   ")
    assert vector.indices == []
    assert vector.values == []


def test_encode_document_indices_sorted_and_values_positive():
    vector = BM25Encoder().encode_document("متن نمونه برای آزمایش متن")
    assert vector.indices == sorted(vector.indices)
    assert all(value > 0 for value in vector.values)


def test_token_ids_are_deterministic():
    assert BM25Encoder._token_id("سلام") == BM25Encoder._token_id("سلام")


def test_invalid_parameters_rejected():
    with pytest.raises(ValueError):
        BM25Encoder(k1=-1)
    with pytest.raises(ValueError):
        BM25Encoder(b=2.0)
    with pytest.raises(ValueError):
        BM25Encoder(average_length=0)
