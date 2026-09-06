"""Deterministic chunker â€” splits CanonicalDocument into DocumentChunks.

Uses fixed-size character windows with configurable overlap.  Chunk IDs and
content hashes are deterministic: the same input always produces the same
chunks.  Source spans are mapped from the document's spans to each chunk.
"""
from __future__ import annotations

import hashlib
from typing import Any

from ipa.contracts import CanonicalDocument, DocumentChunk, SourceSpan


def _chunk_id(document_id: str, index: int) -> str:
    raw = f"{document_id}:{index}".encode("utf-8")
    return f"chunk:{hashlib.sha256(raw).hexdigest()[:16]}"


def _content_hash(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _map_span(
    chunk_start: int, chunk_end: int, doc_spans: list[SourceSpan]
) -> SourceSpan | None:
    """Find the document source span that contains this chunk's character range."""
    for span in doc_spans:
        if span.offset_start <= chunk_start < span.offset_end:
            return SourceSpan(
                artifact_id=span.artifact_id,
                page=span.page,
                offset_start=chunk_start,
                offset_end=min(chunk_end, span.offset_end),
            )
    return doc_spans[0] if doc_spans else None


def chunk_document(
    doc: CanonicalDocument,
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[DocumentChunk]:
    """Split a CanonicalDocument into deterministic overlapping chunks.

    Args:
        doc: The canonical document to chunk.
        chunk_size: Target character window size.
        overlap: Number of characters to overlap between consecutive chunks.

    Returns:
        List of DocumentChunk records with stable IDs and source spans.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be in [0, chunk_size)")

    text = doc.text
    chunks: list[DocumentChunk] = []
    if not text:
        return chunks

    step = chunk_size - overlap
    index = 0
    pos = 0
    while pos < len(text):
        end = min(pos + chunk_size, len(text))
        chunk_text = text[pos:end]
        span = _map_span(pos, end, doc.source_spans)
        chunks.append(DocumentChunk(
            chunk_id=_chunk_id(doc.document_id, index),
            document_id=doc.document_id,
            content_hash=_content_hash(chunk_text),
            text=chunk_text,
            metadata={
                "chunk_index": index,
                "char_start": pos,
                "char_end": end,
                "chunk_size": chunk_size,
                "overlap": overlap,
            },
            source_span=span,
        ))
        index += 1
        if end >= len(text):
            break
        pos += step

    return chunks

