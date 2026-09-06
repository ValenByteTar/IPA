"""Parser adapters â€” convert raw artifacts into CanonicalDocument records.

Each parser is an independent adapter behind the contract boundary.
The fast path uses PyMuPDF for PDFs and stdlib parsers for text/html/json.
No parser imports an enrichment or embedding dependency.

Stage 1 limitations (documented):
  - PDF: PyMuPDF ``get_text("text")`` extracts flat text; no table/column
    detection, no reading-order heuristics, no OCR.  Adequate for baseline
    lexical indexing.
  - HTML: regex-based tag stripping.  Works for simple/synthetic HTML.
    Does NOT preserve semantic structure (headings, tables, nav vs article).
    For real-world HTML with complex layout, a structural parser
    (selectolax/trafilatura) is required â€” planned for E2/E5.
  - JSON: extracts the ``text`` field as primary content (if present),
    remaining keys become metadata in ``elements``.  Falls back to
    re-serialization for JSON without a ``text`` field.
  - TXT: reads UTF-8 as-is.  No structure to extract.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
import unicodedata
from pathlib import Path
from typing import Any

from ipa.contracts import CanonicalDocument, ParserResult, SourceSpan
from ipa.ingestion.content_safety import safe_open_pdf, SafeParseConfig, SafeParseError


def _doc_id(artifact_id: str, parser_id: str) -> str:
    raw = f"{artifact_id}:{parser_id}".encode("utf-8")
    return f"doc:{hashlib.sha256(raw).hexdigest()[:16]}"


# ---------------------------------------------------------------------------
# Shared text normalization
# ---------------------------------------------------------------------------

# Unicode private-use area (U+E000â€“U+F8FF) â€” common in PDF fonts (Wingdings,
# Symbol, etc.).  These glyphs carry no semantic meaning for lexical search.
_PUA_RANGE = re.compile(r"[\ue000-\uf8ff]")

# Multiple consecutive blank lines / spaces.
_MULTI_BLANK = re.compile(r"\n{3,}")
_MULTI_SPACE = re.compile(r"[ \t]{2,}")


def _normalize_text(text: str) -> str:
    """Normalize text for lexical indexing.

    - Replace Unicode PUA characters (Wingdings, Symbol fonts) with spaces.
    - Normalize common replacement chars (U+FFFD) to nothing.
    - Collapse excessive whitespace (3+ newlines â†’ 1 blank line, 2+ spaces â†’ 1).
    - Strip leading/trailing whitespace.

    Does NOT remove meaningful content â€” only noise that hurts BM25 quality.
    """
    # Remove private-use area glyphs (Wingdings bullets, Symbol font chars).
    text = _PUA_RANGE.sub(" ", text)
    # Drop replacement characters.
    text = text.replace("\ufffd", "")
    # Normalize unicode to NFC for consistent tokenization.
    text = unicodedata.normalize("NFC", text)
    # Collapse excessive whitespace.
    text = _MULTI_BLANK.sub("\n\n", text)
    text = _MULTI_SPACE.sub(" ", text)
    return text.strip()


def parse_pdf_pymupdf(path: Path, artifact_id: str) -> ParserResult:
    """Parse a PDF using PyMuPDF with OCR only for genuinely scanned PDFs.

    Text extraction flow:
      1. Scan all pages: extract text with ``get_text("text")`` and count
         how many pages have extractable text.
      2. If >=10% of pages have text â†’ text-based PDF â†’ no OCR at all.
         (Images in text-based PDFs are figures/charts, not scanned text.)
      3. If <10% of pages have text â†’ scanned PDF â†’ OCR every page.

    This eliminates OCR for the vast majority of PDFs (system cards,
    arxiv papers, CISA advisories) while still handling genuinely
    scanned documents.
    """
    import pymupdf

    # Layer 3: safe PDF open (no JS, no actions, no embedded files, page limit)
    safe_config = SafeParseConfig(max_pages=2000, timeout_seconds=120)
    try:
        doc_ctx = safe_open_pdf(path, safe_config)
        doc = doc_ctx.__enter__()
    except SafeParseError:
        raise  # Re-raise safety errors

    total_pages = len(doc)

    # --- Phase 1: assess whether this PDF needs OCR at all ---
    pages_with_text = 0
    for page_num in range(total_pages):
        text = doc[page_num].get_text("text").strip()
        if len(text) > 50:
            pages_with_text += 1

    text_page_ratio = pages_with_text / max(total_pages, 1)
    pdf_is_scanned = text_page_ratio < 0.1  # <10% of pages have text

    # --- Phase 2: extract text (with OCR only if scanned) ---
    pages: list[dict[str, Any]] = []
    spans: list[SourceSpan] = []
    full_text_parts: list[str] = []
    offset = 0

    _ocr_adapter = None

    for page_num in range(total_pages):
        page = doc[page_num]
        raw_text = page.get_text("text")
        text = _normalize_text(raw_text)

        # Only OCR if the entire PDF is scanned (no extractable text)
        if pdf_is_scanned and not text.strip():
            if _ocr_adapter is None:
                from ipa.acquisition.ocr_adapter import OCRAdapter
                _ocr_adapter = OCRAdapter(gpu=True, paragraph=True)

            # Render page to image
            max_dim_px = max(page.rect.width, page.rect.height)
            if max_dim_px > 0:
                dpi = min(200, 2000 / max_dim_px * 72)
            else:
                dpi = 150
            mat = pymupdf.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat)
            img_bytes = pix.tobytes("png")

            import tempfile, os
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".png", prefix="pdf_page_")
            tmp_path = Path(tmp_path)
            try:
                os.write(tmp_fd, img_bytes)
                os.close(tmp_fd)
                tmp_fd = -1

                ocr_result = _ocr_adapter.extract_text(tmp_path)
                if ocr_result.success and ocr_result.text.strip():
                    text = f"[OCR] {ocr_result.text}"
            except Exception:
                pass  # OCR failure is non-fatal
            finally:
                if tmp_fd >= 0:
                    try:
                        os.close(tmp_fd)
                    except OSError:
                        pass
                for _ in range(3):
                    try:
                        if tmp_path.exists():
                            tmp_path.unlink()
                        break
                    except (OSError, PermissionError):
                        time.sleep(0.5)

        pages.append({"page": page_num + 1, "char_count": len(text)})
        spans.append(SourceSpan(
            artifact_id=artifact_id,
            page=page_num + 1,
            offset_start=offset,
            offset_end=offset + len(text),
        ))
        full_text_parts.append(text)
        offset += len(text) + 1  # +1 for the page separator

    # Clean up OCR adapter if it was loaded
    if _ocr_adapter is not None:
        _ocr_adapter.close()

    # Close safe PDF context (Layer 3)
    doc_ctx.__exit__(None, None, None)
    full_text = "\n".join(full_text_parts)
    canonical = CanonicalDocument(
        document_id=_doc_id(artifact_id, "pymupdf"),
        pages=len(pages),
        elements=pages,
        source_spans=spans,
        text=full_text,
        mime_type="application/pdf",
        parser_id="pymupdf",
    )
    return ParserResult(
        artifact_id=artifact_id,
        parser_id="pymupdf",
        status="parsed",
        canonical_document=canonical,
    )


def parse_text(path: Path, artifact_id: str) -> ParserResult:
    """Parse a plain-text file."""
    raw = path.read_text(encoding="utf-8")
    text = _normalize_text(raw)
    canonical = CanonicalDocument(
        document_id=_doc_id(artifact_id, "text"),
        pages=1,
        elements=[{"page": 1, "char_count": len(text)}],
        source_spans=[SourceSpan(
            artifact_id=artifact_id, page=1,
            offset_start=0, offset_end=len(text),
        )],
        text=text,
        mime_type="text/plain",
        parser_id="text",
    )
    return ParserResult(
        artifact_id=artifact_id, parser_id="text",
        status="parsed", canonical_document=canonical,
    )


def parse_html(path: Path, artifact_id: str) -> ParserResult:
    """Parse an HTML file â€” strip tags with a simple regex-based extractor.

    Stage 1 limitation: this regex parser does not preserve semantic structure
    (headings, tables, nav vs article).  For real-world HTML with complex
    layout, a structural parser (selectolax/trafilatura) is required.
    See experiments E2/E5 in the roadmap.
    """
    raw = path.read_text(encoding="utf-8")
    # Remove script/style blocks first.
    cleaned = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", raw, flags=re.DOTALL | re.IGNORECASE)
    # Remove HTML comments.
    cleaned = re.sub(r"<!--.*?-->", "", cleaned, flags=re.DOTALL)
    # Remove remaining tags.
    text = re.sub(r"<[^>]+>", " ", cleaned)
    # Normalize whitespace + PUA chars.
    text = _normalize_text(text)

    canonical = CanonicalDocument(
        document_id=_doc_id(artifact_id, "html"),
        pages=1,
        elements=[{"page": 1, "char_count": len(text)}],
        source_spans=[SourceSpan(
            artifact_id=artifact_id, page=1,
            offset_start=0, offset_end=len(text),
        )],
        text=text,
        mime_type="text/html",
        parser_id="html",
    )
    return ParserResult(
        artifact_id=artifact_id, parser_id="html",
        status="parsed", canonical_document=canonical,
    )


def parse_json(path: Path, artifact_id: str) -> ParserResult:
    """Parse a JSON file â€” extract ``text`` field as content, rest as metadata.

    If the JSON has a ``text`` field (string), it becomes the primary content
    for chunking and indexing.  All other keys are stored in ``elements`` as
    metadata, preserving structure without polluting the lexical index.

    If there is no ``text`` field, falls back to re-serializing the entire
    JSON with ``indent=2`` and ``sort_keys=True`` for deterministic text.
    """
    data = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(data, dict) and isinstance(data.get("text"), str):
        # Structured JSON with a text field â€” extract it.
        text = _normalize_text(data["text"])
        metadata = {k: v for k, v in data.items() if k != "text"}
        elements = [{"page": 1, "char_count": len(text), "keys": list(metadata.keys()), "metadata": metadata}]
    else:
        # No text field â€” fall back to re-serialization.
        text = _normalize_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True))
        elements = [{"page": 1, "char_count": len(text), "keys": list(data.keys()) if isinstance(data, dict) else []}]

    canonical = CanonicalDocument(
        document_id=_doc_id(artifact_id, "json"),
        pages=1,
        elements=elements,
        source_spans=[SourceSpan(
            artifact_id=artifact_id, page=1,
            offset_start=0, offset_end=len(text),
        )],
        text=text,
        mime_type="application/json",
        parser_id="json",
    )
    return ParserResult(
        artifact_id=artifact_id, parser_id="json",
        status="parsed", canonical_document=canonical,
    )


_PARSERS: dict[str, Any] = {
    "pymupdf": parse_pdf_pymupdf,
    "text": parse_text,
    "html": parse_html,
    "json": parse_json,
}


def parse(path: Path, artifact_id: str, parser_id: str) -> ParserResult:
    """Dispatch to the named parser.  Raises ValueError for unknown parsers."""
    parser = _PARSERS.get(parser_id)
    if parser is None:
        return ParserResult(
            artifact_id=artifact_id, parser_id=parser_id,
            status="failed", canonical_document=None,
        )
    return parser(path, artifact_id)

