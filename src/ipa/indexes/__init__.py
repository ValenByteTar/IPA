"""Derived index adapters (transitional facade)."""
from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "BM25Index": ("bm25_index", "BM25Index"),
    "TantivyIndex": ("tantivy_index", "TantivyIndex"),
    "LanceDBIndex": ("lancedb_index", "LanceDBIndex"),
    "SQLiteVecIndex": ("sqlite_vec_index", "SQLiteVecIndex"),
    "EmbeddingAdapter": ("embedding_adapter", "EmbeddingAdapter"),
    "RerankerAdapter": ("reranker_adapter", "RerankerAdapter"),
}


def __getattr__(name: str):
    try:
        module_name, attr = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    return getattr(import_module(f"ipa.indexes.{module_name}"), attr)


__all__ = sorted(_EXPORTS)

