"""Deprecated compatibility facade for the former res023_lab package."""
from __future__ import annotations

from importlib import import_module

__version__ = "0.1.0"

_SEARCH_MODULES = (
    "ipa.contracts",
    "ipa.ingestion.chunker",
    "ipa.ingestion.alt_chunkers",
    "ipa.ingestion.fast_path",
    "ipa.ingestion.landing_zone",
    "ipa.ingestion.mime_router",
    "ipa.ingestion.parsers",
    "ipa.ingestion.docling_parser",
    "ipa.ingestion.unstructured_parser",
    "ipa.acquisition.web_scraper",
    "ipa.acquisition.ocr_adapter",
    "ipa.observability.trace_log",
    "ipa.indexes.bm25_index",
    "ipa.indexes.tantivy_index",
    "ipa.indexes.lancedb_index",
    "ipa.indexes.sqlite_vec_index",
    "ipa.indexes.embedding_adapter",
    "ipa.reporter.reporter_contracts",
    "ipa.reporter.reporter_pipeline",
    "ipa.tutor.tutor_contracts",
    "ipa.providers.exl3_provider",
)


def __getattr__(name: str):
    ipa = import_module("ipa")
    try:
        return getattr(ipa, name)
    except AttributeError:
        for module_name in _SEARCH_MODULES:
            module = import_module(module_name)
            if hasattr(module, name):
                value = getattr(module, name)
                globals()[name] = value
                return value
        raise


__all__ = []
