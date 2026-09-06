"""TantivyIndex â€” lexical index backed by Tantivy (Rust).

Competitor to SQLite FTS5 in E6.  Provides the same interface as BM25Index:
add_chunks, search, count, is_queryable, close.

Tantivy is a full-text search engine written in Rust.  The Python binding
(`tantivy`) provides BM25 scoring out of the box.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import tantivy
from typing import Iterator

from ipa.contracts import DocumentChunk, SearchHit, SourceSpan


def _span_to_json(span: SourceSpan | None) -> str:
    if span is None:
        return "{}"
    return json.dumps({
        "artifact_id": span.artifact_id,
        "page": span.page,
        "offset_start": span.offset_start,
        "offset_end": span.offset_end,
    })


def _json_to_span(s: str) -> SourceSpan | None:
    if not s or s == "{}":
        return None
    d = json.loads(s)
    return SourceSpan(
        artifact_id=d["artifact_id"], page=d["page"],
        offset_start=d["offset_start"], offset_end=d["offset_end"],
    )


class TantivyIndex:
    """Incremental lexical index backed by Tantivy with BM25 ranking."""

    def __init__(self, db_path: str | Path, read_only: bool = False) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._read_only = read_only

        # Define schema: chunk_id and document_id are stored (retrievable),
        # text is the indexed field, span_json is stored but not indexed.
        # In tantivy 0.26, fields are indexed by default when a tokenizer is
        # specified.  Stored-only fields use stored=True without tokenizer.
        schema_builder = tantivy.SchemaBuilder()
        schema_builder.add_text_field("chunk_id", stored=True, tokenizer_name="raw")
        schema_builder.add_text_field("document_id", stored=True, tokenizer_name="raw")
        schema_builder.add_text_field("content_hash", stored=True, tokenizer_name="raw")
        schema_builder.add_text_field("text", stored=True, tokenizer_name="default")
        schema_builder.add_text_field("span_json", stored=True, tokenizer_name="raw")
        self.schema = schema_builder.build()

        # Open or create the index.  Tantivy requires the directory to exist.
        self.db_path.mkdir(parents=True, exist_ok=True)

        self.index = tantivy.Index(self.schema, path=str(self.db_path))
        if read_only:
            # Read-only mode: no writer, no lock â€” safe for concurrent queries
            self.index_writer = None
        else:
            self.index_writer = self.index.writer()
        self._count = 0

    def _get_searcher(self) -> tantivy.Searcher:
        """Get a fresh searcher.  Must be called per search to see new commits."""
        self.index.reload()
        return self.index.searcher()

    def add_chunk(self, chunk: DocumentChunk) -> None:
        """Insert a single chunk.  Not idempotent â€” use add_chunks for batches."""
        doc = tantivy.Document()
        doc.add_text("chunk_id", chunk.chunk_id)
        doc.add_text("document_id", chunk.document_id)
        doc.add_text("content_hash", chunk.content_hash)
        doc.add_text("text", chunk.text)
        doc.add_text("span_json", _span_to_json(chunk.source_span))
        self.index_writer.add_document(doc)
        self._count += 1

    def add_chunks(self, chunks: list[DocumentChunk], commit: bool = True) -> None:
        """Batch-insert chunks.  Idempotent: deletes existing docs with
        matching chunk_id before inserting.  Commits by default."""
        if not chunks:
            return
        # Delete existing docs with same chunk_id (idempotency on reprocess)
        for chunk in chunks:
            self.index_writer.delete_documents_by_term("chunk_id", chunk.chunk_id)
        for chunk in chunks:
            doc = tantivy.Document()
            doc.add_text("chunk_id", chunk.chunk_id)
            doc.add_text("document_id", chunk.document_id)
            doc.add_text("content_hash", chunk.content_hash)
            doc.add_text("text", chunk.text)
            doc.add_text("span_json", _span_to_json(chunk.source_span))
            self.index_writer.add_document(doc)
        self._count += len(chunks)
        if commit:
            self.commit()

    def commit(self) -> None:
        """Flush pending writes and make them searchable."""
        self.index_writer.commit()

    def search(self, query: str, limit: int = 10) -> list[SearchHit]:
        """Search using Tantivy BM25 ranking.  Returns SearchHit records."""
        if limit <= 0:
            raise ValueError("limit must be positive")
        searcher = self._get_searcher()
        # Sanitize query: Tantivy's query parser chokes on special chars
        # like apostrophes, colons, and parentheses.  Strip them and keep
        # only alphanumeric tokens and spaces.
        import re
        safe_query = re.sub(r'[^\w\s]', ' ', query).strip()
        if not safe_query:
            return []
        # Parse query targeting the "text" field.
        try:
            parsed = self.index.parse_query(safe_query, default_field_names=["text"])
        except (ValueError, Exception):
            # If parsing still fails, fall back to a simple term search.
            terms = safe_query.split()
            if not terms:
                return []
            parsed = self.index.parse_query(
                " ".join(terms[:5]), default_field_names=["text"]
            )
        search_result = searcher.search(parsed, limit=limit)
        results: list[SearchHit] = []
        for score, doc_address in search_result.hits:
            doc = searcher.doc(doc_address)
            chunk_id = doc.get_first("chunk_id") or ""
            span_json = doc.get_first("span_json") or "{}"
            results.append(SearchHit(
                chunk_id=chunk_id,
                score=score,
                source_span=_json_to_span(span_json),
                retrieval_backend="tantivy",
            ))
        return results

    def count(self) -> int:
        """Return the number of indexed documents."""
        if self._count > 0:
            return self._count
        searcher = self._get_searcher()
        return searcher.num_docs

    def is_queryable(self) -> bool:
        """True if at least one document is indexed."""
        return self.count() > 0

    def close(self) -> None:
        """Flush and close the writer."""
        try:
            self.index_writer.commit()
        except Exception:
            pass

    def __enter__(self) -> "TantivyIndex":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

