"""PDF table regressions, including extraction from real generated pages."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import pymupdf

from app.service.document_processor import DocumentProcessor


ROWS = [["Product", "Quantity"], ["Apples", "12"], ["Oranges", "24"]]


def candidate(bbox, rows=None):
    return SimpleNamespace(bbox=bbox, extract=lambda: rows or ROWS)


def draw_table(page, top=100, ruled=True):
    if ruled:
        for y in (top, top + 30, top + 60, top + 90):
            page.draw_line((50, y), (350, y))
        for x in (50, 200, 350):
            page.draw_line((x, top), (x, top + 90))
    for index, row in enumerate(ROWS):
        for x, text in zip((60, 210), row):
            page.insert_text((x, top + 20 + 30 * index), text, fontsize=11)


class PdfTableTests(unittest.TestCase):
    def setUp(self):
        self.processor = DocumentProcessor()

    def extract_candidates(self, primary, fallback):
        page = Mock()
        page.find_tables.return_value = SimpleNamespace(tables=primary)
        plumber = Mock()
        plumber.find_tables.return_value = fallback
        out = []
        rects = self.processor._extract_page_tables(page, 1, out, lambda _: plumber)
        return out, rects

    def test_primary_duplicate_and_contained_detections_count_once(self):
        out, rects = self.extract_candidates([
            candidate((50, 100, 350, 190)),
            candidate((51, 101, 349, 189)),
            candidate((50, 100, 200, 160)),
        ], [])
        self.assertEqual(len(out), 1)
        self.assertEqual(len(rects), 1)

    def test_fallback_prefers_complete_table_over_contained_fragment(self):
        out, rects = self.extract_candidates([], [
            candidate((50, 100, 200, 160), ROWS[:2]),
            candidate((50, 100, 350, 190)),
        ])
        self.assertEqual(len(out), 1)
        self.assertEqual(tuple(rects[0]), (50, 100, 350, 190))
        self.assertIn("Oranges", out[0]["content"])

    def test_identical_tables_at_separate_locations_are_preserved(self):
        out, _ = self.extract_candidates([], [
            candidate((50, 100, 350, 190)),
            candidate((50, 300, 350, 390)),
        ])
        self.assertEqual(len(out), 2)
        self.assertEqual([table["table_index"] for table in out], [0, 1])

    def test_bad_candidate_does_not_hide_valid_table(self):
        out, _ = self.extract_candidates([
            candidate(("invalid", 1, 2, 3)),
            candidate((50, 100, 350, 190)),
        ], [])
        self.assertEqual(len(out), 1)

    def test_output_uses_page_order_after_preferring_larger_regions(self):
        out, rects = self.extract_candidates([
            candidate((50, 100, 250, 150), [["First", "Value"], ["a", "b"]]),
            candidate((50, 300, 350, 390)),
        ], [])
        self.assertEqual(len(out), 2)
        self.assertIn("First", out[0]["content"])
        self.assertEqual([table["table_index"] for table in out], [0, 1])
        self.assertLess(rects[0].y0, rects[1].y0)

    def test_empty_rows_or_columns_do_not_create_a_table(self):
        self.assertFalse(self.processor._is_valid_table([["a", "b"], [None, None]]))
        self.assertFalse(self.processor._is_valid_table([["a", None], ["b", None]]))
        self.assertTrue(self.processor._is_valid_table([["Name", "Value"], ["a", 0]]))

    def test_detection_failure_keeps_page_text(self):
        with pymupdf.open() as doc:
            page = doc.new_page()
            page.insert_text((50, 70), "Ordinary document text remains available.")
            get_plumber = Mock(side_effect=RuntimeError("unavailable"))
            out = []
            rects = self.processor._extract_page_tables(page, 1, out, get_plumber)
            lines = self.processor._extract_page_lines(page, rects)
        self.assertEqual(out, [])
        self.assertIn("Ordinary document text", lines[0][0])

    def extract_document(self, build):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.pdf"
            with pymupdf.open() as doc:
                build(doc)
                doc.save(path)
            self.processor.upload_dir = Path(directory) / "artifacts"
            return self.processor._extract_pdf_content(path, "fixture")

    def test_six_pages_with_only_one_table(self):
        def build(doc):
            for index in range(6):
                page = doc.new_page()
                page.insert_text((50, 50), f"Section {index + 1}")
                if index == 2:
                    draw_table(page)
                else:
                    # Repeated word alignments look like columns to the text
                    # strategy, but the spaces between words are not gutters.
                    for line in range(8):
                        page.insert_text(
                            (50, 100 + line * 20),
                            "Normal paragraph text has several words on each line.",
                            fontsize=11,
                        )

        extraction = self.extract_document(build)
        self.assertEqual(len(extraction.tables), 1)
        self.assertEqual(extraction.tables[0]["page_number"], 3)
        self.assertIn("Normal paragraph text", extraction.pages[0][1])

    def test_borderless_table_is_extracted(self):
        extraction = self.extract_document(
            lambda doc: draw_table(doc.new_page(), ruled=False)
        )
        self.assertEqual(len(extraction.tables), 1)
        self.assertIn("Apples", extraction.tables[0]["content"])

    def test_two_column_article_is_not_a_borderless_table(self):
        def build(doc):
            page = doc.new_page()
            for x in (50, 320):
                for row in range(15):
                    page.insert_text(
                        (x, 100 + row * 14),
                        "This paragraph has many words in each line of text.",
                        fontsize=9,
                    )

        extraction = self.extract_document(build)
        self.assertEqual(extraction.tables, [])
        self.assertIn("This paragraph", extraction.pages[0][1])

    def test_captioned_raster_table_is_preserved_and_marked_for_ocr(self):
        def build(doc):
            with pymupdf.open() as source:
                table_page = source.new_page()
                draw_table(table_page)
                raster = table_page.get_pixmap(clip=(45, 95, 355, 195)).tobytes("png")
            page = doc.new_page()
            page.insert_text((180, 70), "TABLE I")
            page.insert_text((120, 85), "Product quantities")
            page.insert_image((50, 100, 360, 200), stream=raster)
            page.insert_text((50, 250), "Body text below the table stays available.")

        extraction = self.extract_document(build)
        self.assertEqual(len(extraction.tables), 1)
        table = extraction.tables[0]
        self.assertEqual(table["extraction_method"], "image")
        self.assertTrue(table["requires_ocr"])
        self.assertIn("TABLE I", table["content"])
        self.assertIn("Product quantities", table["content"])
        self.assertEqual(len(extraction.images), 1)
        self.assertIn("Body text below", extraction.pages[0][1])

    def test_figure_caption_does_not_turn_an_image_into_a_table(self):
        def build(doc):
            with pymupdf.open() as source:
                source.new_page()
                raster = source[0].get_pixmap(clip=(0, 0, 100, 100)).tobytes("png")
            page = doc.new_page()
            page.insert_image((50, 100, 150, 200), stream=raster)
            page.insert_text((50, 220), "Fig. 1. This is a figure, not a table.")

        extraction = self.extract_document(build)
        self.assertEqual(extraction.tables, [])
        self.assertEqual(len(extraction.images), 1)

    def test_bookmark_paragraphs_are_rejected_and_depth_is_not_heading_count(self):
        def build(doc):
            page = doc.new_page()
            page.insert_text((50, 50), "Introduction")
            page.insert_text((50, 100), "Details")
            page.insert_text((50, 150), "Conclusion")
            doc.set_toc([
                [1, "Introduction", 1],
                [2, "Details", 1],
                [1, "Conclusion", 1],
                [2, "A conclusion section is not required. " * 6, 1],
                [1, "[1] J. Author, An example reference.", 1],
            ])

        extraction = self.extract_document(build)
        self.assertEqual([h.title for h in extraction.hierarchy], ["Introduction", "Details", "Conclusion"])
        with patch.object(self.processor, "_extract_pdf_content", return_value=extraction):
            _, stats = self.processor._process_pdf(Path("fixture.pdf"), "fixture", "fixture.pdf")
        self.assertEqual(stats["heading_count"], 3)
        self.assertEqual(stats["hierarchy_levels"], 2)

    @unittest.skipUnless(
        (Path(__file__).resolve().parents[1] / "uploads/Transactions-instructions-only.pdf").exists(),
        "User-supplied PDF is not part of the repository",
    )
    def test_supplied_transactions_pdf(self):
        from app.schemas.models import DocumentMetadata

        source = Path(__file__).resolve().parents[1] / "uploads/Transactions-instructions-only.pdf"
        with TemporaryDirectory() as directory:
            self.processor.upload_dir = Path(directory)
            documents, stats = self.processor._process_file(str(source))
            metadata = DocumentMetadata(**stats)
            self.assertEqual(metadata.table_count, 1)
            self.assertEqual(metadata.hierarchy_levels, 2)
            self.assertEqual(metadata.heading_count, 27)
            self.assertEqual(metadata.tables_requiring_ocr, 1)
            self.assertEqual(metadata.image_count, 3)
            tables = [d for d in documents if d.metadata["content_type"] == "table"]
            self.assertEqual(tables[0].metadata["page_number"], 2)
            self.assertTrue(Path(tables[0].metadata["image_path"]).is_file())
            prose = "\n".join(d.page_content for d in documents if d.metadata["content_type"] == "text")
            self.assertIn("Abstract", prose)
            self.assertIn("conclusion", prose)

    def test_two_real_tables_are_preserved(self):
        def build(doc):
            page = doc.new_page()
            draw_table(page)
            draw_table(page, top=300)

        extraction = self.extract_document(build)
        self.assertEqual(len(extraction.tables), 2)

    def test_identical_tables_on_different_pages_are_preserved(self):
        def build(doc):
            draw_table(doc.new_page())
            draw_table(doc.new_page())

        extraction = self.extract_document(build)
        self.assertEqual(len(extraction.tables), 2)
        self.assertEqual([table["page_number"] for table in extraction.tables], [1, 2])

    def test_table_text_is_not_duplicated_in_prose(self):
        def build(doc):
            page = doc.new_page()
            page.insert_text((50, 50), "Introduction outside the table.")
            draw_table(page)
            page.insert_text((50, 240), "Conclusion outside the table.")

        extraction = self.extract_document(build)
        self.assertEqual(len(extraction.tables), 1)
        prose = extraction.pages[0][1]
        self.assertIn("Introduction", prose)
        self.assertIn("Conclusion", prose)
        self.assertNotIn("Apples", prose)


if __name__ == "__main__":
    unittest.main()
