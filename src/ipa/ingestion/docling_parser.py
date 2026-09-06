"""DoclingParser â€” PDF parser backed by IBM Docling.

Competitor to PyMuPDF in E3.  Produces CanonicalDocument records with
per-page text and source spans, same interface as parse_pdf_pymupdf.

Docling uses AI models for layout detection, table extraction, and
reading order.  Heavier than PyMuPDF but produces structured output.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

from ipa.contracts import CanonicalDocument, ParserResult, SourceSpan
from ipa.ingestion.parsers import _normalize_text, _doc_id


def parse_pdf_docling(path: Path, artifact_id: str, do_ocr: bool = False) -> ParserResult:
    """Parse a PDF using Docling.  Produces per-page text and source spans.

    Args:
        path: Path to the PDF file.
        artifact_id: Unique artifact identifier.
        do_ocr: Enable OCR for scanned/image-based PDFs.  Defaults to False
            because most PDFs in the corpus are born-digital with embedded
            text.  OCR adds ~3-5s/page overhead and runs RapidOCR on every
            image found in each page (logos, charts, decorations), producing
            many empty results.  Enable only for known scanned PDFs.
    """
    try:
        from docling.document_converter import DocumentConverter
        from docling.datamodel.pipeline_options import PdfPipelineOptions
    except ImportError:
        return ParserResult(
            artifact_id=artifact_id, parser_id="docling",
            status="failed", canonical_document=None,
        )

    # Docling reads DOCLING_DEVICE and supports CUDA for model inference.
    # Keep this explicit so the E3 benchmark does not silently fall back to CPU.
    import os
    os.environ.setdefault("DOCLING_DEVICE", "cuda")

    # Configure pipeline: layout detection + table structure, OCR only if needed.
    from docling.document_converter import PdfFormatOption
    from docling.datamodel.base_models import InputFormat

    pipeline_options = PdfPipelineOptions(
        do_ocr=do_ocr,
        do_table_structure=True,
    )
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
            )
        }
    )
    result = converter.convert(str(path))
    doc = result.document

    # Export to markdown â€” Docling's primary text representation.
    full_text = doc.export_to_markdown()
    full_text = _normalize_text(full_text)

    # Docling doesn't expose per-page offsets directly in markdown export.
    # We create a single span covering the whole document.
    pages = doc.pages
    num_pages = len(pages) if pages else 1

    # Build per-page elements metadata.
    page_elements: list[dict[str, Any]] = []
    for page_no in sorted(pages.keys()) if pages else [1]:
        page_info = pages.get(page_no)
        page_elements.append({
            "page": page_no,
            "size": {"width": page_info.size.width, "height": page_info.size.height}
            if page_info and page_info.size else None,
        })

    spans = [SourceSpan(
        artifact_id=artifact_id,
        page=1,
        offset_start=0,
        offset_end=len(full_text),
    )]

    canonical = CanonicalDocument(
        document_id=_doc_id(artifact_id, "docling"),
        pages=num_pages,
        elements=page_elements,
        source_spans=spans,
        text=full_text,
        mime_type="application/pdf",
        parser_id="docling",
    )
    return ParserResult(
        artifact_id=artifact_id,
        parser_id="docling",
        status="parsed",
        canonical_document=canonical,
    )

