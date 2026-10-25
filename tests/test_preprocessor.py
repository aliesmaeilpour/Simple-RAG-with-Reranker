from app.service.document_processor import ChunkQualityScorer, TextPreprocessor


def test_normalize_whitespace_collapses_spaces_and_blank_lines():
    text = "سلام\tدنیا\r\n\r\n\r\nمتن  جدید"
    assert TextPreprocessor.normalize_whitespace(text) == "سلام دنیا\n\nمتن جدید"


def test_normalize_persian_unifies_arabic_variants():
    assert TextPreprocessor.normalize_persian("ي ك") == "ی ک"


def test_normalize_persian_collapses_repeated_zwnj():
    text = f"می{'‌' * 3}شود"
    assert TextPreprocessor.normalize_persian(text) == f"می{'‌'}شود"


def test_remove_headers_footers_strips_page_markers():
    text = "صفحه 12\nبدنه متن اینجا است\n7"
    result = TextPreprocessor.remove_headers_footers(text)
    lines = result.split("\n")
    assert "صفحه 12" not in lines
    assert "7" not in lines
    assert "بدنه متن اینجا است" in lines


def test_is_boilerplate_detects_page_numbers_and_short_text():
    assert TextPreprocessor.is_boilerplate("42")
    assert TextPreprocessor.is_boilerplate("صفحه 10")
    assert not TextPreprocessor.is_boilerplate("این یک متن طولانی و معنادار است.")


def test_short_chunk_is_filtered():
    quality = ChunkQualityScorer.score_chunk("کوتاه")
    assert not ChunkQualityScorer.should_keep_chunk(quality)


def test_normal_chunk_is_kept():
    text = (
        "این یک متن نمونه است که طول کافی دارد و برای آزمایش کیفیت "
        "قطعه‌بندی به کار می‌رود. جمله دوم برای افزایش چگالی اطلاعاتی."
    )
    quality = ChunkQualityScorer.score_chunk(text)
    assert ChunkQualityScorer.should_keep_chunk(quality)
