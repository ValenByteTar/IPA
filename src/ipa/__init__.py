"""IPA public package with bounded-context lazy exports."""
from __future__ import annotations

from importlib import import_module
from typing import Any

__version__ = "0.2.1"

_EXPORTS = {
    "ArtifactRef": ("ipa.contracts", "ArtifactRef"),
    "CanonicalDocument": ("ipa.contracts", "CanonicalDocument"),
    "DocumentChunk": ("ipa.contracts", "DocumentChunk"),
    "ParserResult": ("ipa.contracts", "ParserResult"),
    "SearchHit": ("ipa.contracts", "SearchHit"),
    "SourceSpan": ("ipa.contracts", "SourceSpan"),
    "chunk_document": ("ipa.ingestion.chunker", "chunk_document"),
    "chunk_document_recursive": ("ipa.ingestion.alt_chunkers", "chunk_document_recursive"),
    "chunk_document_token": ("ipa.ingestion.alt_chunkers", "chunk_document_token"),
    "chunk_document_semantic": ("ipa.ingestion.alt_chunkers", "chunk_document_semantic"),
    "parse": ("ipa.ingestion.parsers", "parse"),
    "parse_pdf_pymupdf": ("ipa.ingestion.parsers", "parse_pdf_pymupdf"),
    "parse_pdf_docling": ("ipa.ingestion.docling_parser", "parse_pdf_docling"),
    "parse_pdf_unstructured": ("ipa.ingestion.unstructured_parser", "parse_pdf_unstructured"),
    "LandingZone": ("ipa.ingestion.landing_zone", "LandingZone"),
    "detect_mime": ("ipa.ingestion.mime_router", "detect_mime"),
    "route_to_parser": ("ipa.ingestion.mime_router", "route_to_parser"),
    "FastPathRunner": ("ipa.ingestion.fast_path", "FastPathRunner"),
    "FastPathResult": ("ipa.ingestion.fast_path", "FastPathResult"),
    "DocumentStore": ("ipa.storage.document_store", "DocumentStore"),
    "BM25Index": ("ipa.indexes.bm25_index", "BM25Index"),
    "TantivyIndex": ("ipa.indexes.tantivy_index", "TantivyIndex"),
    "EmbeddingAdapter": ("ipa.indexes.embedding_adapter", "EmbeddingAdapter"),
    "LanceDBIndex": ("ipa.indexes.lancedb_index", "LanceDBIndex"),
    "SQLiteVecIndex": ("ipa.indexes.sqlite_vec_index", "SQLiteVecIndex"),
    "TraceLog": ("ipa.observability.trace_log", "TraceLog"),
    "TraceEvent": ("ipa.observability.trace_log", "TraceEvent"),
    "make_event": ("ipa.observability.trace_log", "make_event"),
    "WebScraper": ("ipa.acquisition.web_scraper", "WebScraper"),
    "ScrapeSite": ("ipa.acquisition.web_scraper", "ScrapeSite"),
    "ScrapeResult": ("ipa.acquisition.web_scraper", "ScrapeResult"),
    "ScrapeSummary": ("ipa.acquisition.web_scraper", "ScrapeSummary"),
    "PlaywrightBackend": ("ipa.acquisition.web_scraper", "PlaywrightBackend"),
    "OCRAdapter": ("ipa.acquisition.ocr_adapter", "OCRAdapter"),
    "OCRResult": ("ipa.acquisition.ocr_adapter", "OCRResult"),
    "ExL3Provider": ("ipa.providers.exl3_provider", "ExL3Provider"),
    "GenerationResult": ("ipa.providers.exl3_provider", "GenerationResult"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module 'ipa' has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attr)
    globals()[name] = value
    return value


__all__ = sorted(_EXPORTS)
