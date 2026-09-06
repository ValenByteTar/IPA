"""UnstructuredParser â€” PDF parser backed by Unstructured.io.

Competitor to PyMuPDF and Docling in E3.  Unstructured is a popular RAG
ingestion library that supports many formats.  For PDFs it uses
unstructured-inference (layout detection) + pdfminer for text extraction.

By default uses the "hi_res" strategy with layout detection for better
reading order and table extraction.  Falls back to "fast" strategy if
models are unavailable.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ipa.contracts import CanonicalDocument, ParserResult, SourceSpan
from ipa.ingestion.parsers import _normalize_text, _doc_id


def parse_pdf_unstructured(path: Path, artifact_id: str) -> ParserResult:
    """Parse a PDF using Unstructured.  Produces per-page text and source spans.

    Uses hi_res strategy with YOLOX layout detection by default for better
    reading order and table extraction.  Falls back to fast strategy on error.
    """
    try:
        from unstructured.partition.pdf import partition_pdf
    except ImportError:
        return ParserResult(
            artifact_id=artifact_id, parser_id="unstructured",
            status="failed", canonical_document=None,
        )

    # Use fast strategy by default (no layout detection models).
    # hi_res strategy with YOLOX layout detection is ~10x slower (273s vs ~20s
    # per PDF) and the text extraction quality is similar for born-digital PDFs.
    # Switch to hi_res only for scanned/complex PDFs that need layout detection.
    try:
        elements = partition_pdf(
            filename=str(path),
            strategy="fast",
        )
    except Exception:
        return ParserResult(
            artifact_id=artifact_id, parser_id="unstructured",
            status="failed", canonical_document=None,
        )

    # Group elements by page and extract text.
    page_texts: dict[int, list[str]] = {}
    for el in elements:
        page_num = el.metadata.page_number or 1
        text = str(el.text)
        if text.strip():
            page_texts.setdefault(page_num, []).append(text)

    # Build full text with per-page spans.
    full_text_parts: list[str] = []
    page_elements: list[dict[str, Any]] = []
    spans: list[SourceSpan] = []
    offset = 0

    for page_no in sorted(page_texts.keys()):
        page_text = _normalize_text("\n".join(page_texts[page_no]))
        page_elements.append({"page": page_no, "char_count": len(page_text)})
        spans.append(SourceSpan(
            artifact_id=artifact_id,
            page=page_no,
            offset_start=offset,
            offset_end=offset + len(page_text),
        ))
        full_text_parts.append(page_text)
        offset += len(page_text) + 1

    full_text = "\n".join(full_text_parts)
    num_pages = len(page_texts) if page_texts else 1

    canonical = CanonicalDocument(
        document_id=_doc_id(artifact_id, "unstructured"),
        pages=num_pages,
        elements=page_elements,
        source_spans=spans,
        text=full_text,
        mime_type="application/pdf",
        parser_id="unstructured",
    )
    return ParserResult(
        artifact_id=artifact_id,
        parser_id="unstructured",
        status="parsed",
        canonical_document=canonical,
    )

