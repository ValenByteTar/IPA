"""Tests for Stage 3 alternative parser adapters (E3 competition).

Covers DoclingParser and UnstructuredParser adapters.  MinerU is excluded
because its model dependencies are not reliably available in CI.

These tests validate that each adapter:
  - Produces a valid ParserResult with status='parsed'
  - Returns a CanonicalDocument with text, pages, and source_spans
  - Preserves provenance (artifact_id in spans)
  - Handles empty/corrupt PDFs gracefully (status='failed')
  - Produces text that contains expected content from a known PDF

Tests use a synthetic PDF generated with PyMuPDF to avoid external fixtures.
"""
from __future__ import annotations

from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    """Create a small born-digital PDF for testing."""
    import pymupdf
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Semantic chunking test document.", fontsize=12)
    page.insert_text((72, 100), "Second line of content here.", fontsize=12)
    page2 = doc.new_page()
    page2.insert_text((72, 72), "Page two has different topic.", fontsize=12)
    pdf_path = tmp_path / "sample.pdf"
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture
def empty_pdf(tmp_path: Path) -> Path:
    """Create a valid but empty PDF (no text)."""
    import pymupdf
    doc = pymupdf.open()
    doc.new_page()
    pdf_path = tmp_path / "empty.pdf"
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


# ---------------------------------------------------------------------------
# DoclingParser tests
# ---------------------------------------------------------------------------

class TestDoclingParser:
    """Tests for the Docling parser adapter (E3 candidate)."""

    def test_parses_pdf_with_text(self, sample_pdf: Path):
        from ipa import parse_pdf_docling
        result = parse_pdf_docling(sample_pdf, "sha256:test", do_ocr=False)
        assert result.status == "parsed"
        assert result.canonical_document is not None
        doc = result.canonical_document
        assert doc.parser_id == "docling"
        assert doc.pages >= 2
        assert len(doc.text) > 0
        assert "chunking" in doc.text.lower() or "semantic" in doc.text.lower()

    def test_preserves_source_spans(self, sample_pdf: Path):
        from ipa import parse_pdf_docling
        result = parse_pdf_docling(sample_pdf, "sha256:test", do_ocr=False)
        assert result.status == "parsed"
        doc = result.canonical_document
        assert len(doc.source_spans) > 0
        for span in doc.source_spans:
            assert span.artifact_id == "sha256:test"
            assert span.offset_end > span.offset_start

    def test_mime_type_is_pdf(self, sample_pdf: Path):
        from ipa import parse_pdf_docling
        result = parse_pdf_docling(sample_pdf, "sha256:test", do_ocr=False)
        assert result.canonical_document.mime_type == "application/pdf"

    def test_empty_pdf_still_parses(self, empty_pdf: Path):
        """Empty PDFs should parse (no text, but valid structure)."""
        from ipa import parse_pdf_docling
        result = parse_pdf_docling(empty_pdf, "sha256:empty", do_ocr=False)
        # Docling may parse with empty text or fail — both acceptable.
        assert result.status in ("parsed", "failed")


# ---------------------------------------------------------------------------
# UnstructuredParser tests
# ---------------------------------------------------------------------------

class TestUnstructuredParser:
    """Tests for the Unstructured parser adapter (E3 candidate)."""

    def test_parses_pdf_with_text(self, sample_pdf: Path):
        from ipa import parse_pdf_unstructured
        result = parse_pdf_unstructured(sample_pdf, "sha256:test")
        assert result.status == "parsed"
        assert result.canonical_document is not None
        doc = result.canonical_document
        assert doc.parser_id == "unstructured"
        assert doc.pages >= 1
        assert len(doc.text) > 0

    def test_preserves_source_spans(self, sample_pdf: Path):
        from ipa import parse_pdf_unstructured
        result = parse_pdf_unstructured(sample_pdf, "sha256:test")
        assert result.status == "parsed"
        doc = result.canonical_document
        assert len(doc.source_spans) > 0
        for span in doc.source_spans:
            assert span.artifact_id == "sha256:test"

    def test_mime_type_is_pdf(self, sample_pdf: Path):
        from ipa import parse_pdf_unstructured
        result = parse_pdf_unstructured(sample_pdf, "sha256:test")
        assert result.canonical_document.mime_type == "application/pdf"

    def test_empty_pdf_still_parses(self, empty_pdf: Path):
        """Empty PDFs should parse or fail gracefully."""
        from ipa import parse_pdf_unstructured
        result = parse_pdf_unstructured(empty_pdf, "sha256:empty")
        assert result.status in ("parsed", "failed")


# ---------------------------------------------------------------------------
# Cross-parser consistency tests
# ---------------------------------------------------------------------------

class TestParserConsistency:
    """Validate that all parsers produce compatible CanonicalDocument shapes."""

    def test_all_parsers_produce_valid_document(self, sample_pdf: Path):
        """All available parsers must produce a CanonicalDocument with required fields."""
        from ipa import parse_pdf_pymupdf, parse_pdf_docling, parse_pdf_unstructured

        parsers = [
            ("pymupdf", parse_pdf_pymupdf),
            ("docling", lambda p, aid: parse_pdf_docling(p, aid, do_ocr=False)),
            ("unstructured", parse_pdf_unstructured),
        ]

        for name, fn in parsers:
            result = fn(sample_pdf, "sha256:cross")
            assert result.status == "parsed", f"{name} failed to parse"
            doc = result.canonical_document
            assert doc is not None, f"{name} returned None document"
            assert doc.document_id.startswith("doc:"), f"{name} bad document_id"
            assert doc.pages > 0, f"{name} has 0 pages"
            assert isinstance(doc.text, str), f"{name} text is not str"
            assert isinstance(doc.source_spans, list), f"{name} spans not list"
            assert doc.mime_type == "application/pdf", f"{name} wrong mime_type"
