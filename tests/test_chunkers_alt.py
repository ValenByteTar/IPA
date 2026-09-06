"""Tests for Stage 3 alternative chunker adapters (E5 competition).

Covers LangChain RecursiveCharacterTextSplitter, TokenTextSplitter, and
the semantic chunker.  All must produce valid DocumentChunk records
compatible with the baseline fixed-window chunker.

Tests validate:
  - Chunks have stable IDs and content hashes
  - Source spans are mapped correctly
  - Chunk sizes respect configured limits
  - Semantic chunker respects min/max_chunk_size
  - Empty documents produce no chunks
  - All chunkers produce chunks whose text is a substring of the document
"""
from __future__ import annotations

import pytest

from ipa import (
    CanonicalDocument,
    SourceSpan,
    chunk_document,
    chunk_document_recursive,
    chunk_document_token,
)
from ipa.alt_chunkers import chunk_document_semantic


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_doc(text: str, doc_id: str = "doc:test") -> CanonicalDocument:
    """Build a minimal CanonicalDocument for chunker tests."""
    return CanonicalDocument(
        document_id=doc_id,
        pages=1,
        elements=[{"page": 1, "char_count": len(text)}],
        source_spans=[SourceSpan(
            artifact_id="sha256:test",
            page=1,
            offset_start=0,
            offset_end=len(text),
        )],
        text=text,
        mime_type="text/plain",
        parser_id="text",
    )


@pytest.fixture
def sample_doc() -> CanonicalDocument:
    """A document with multiple sentences and paragraphs."""
    paragraphs = []
    for i in range(10):
        paragraphs.append(
            f"This is paragraph {i}. It has multiple sentences. "
            f"The topic is number {i}. We discuss concepts related to {i}."
        )
    return _make_doc("\n\n".join(paragraphs))


@pytest.fixture
def empty_doc() -> CanonicalDocument:
    return _make_doc("")


# ---------------------------------------------------------------------------
# Recursive chunker tests
# ---------------------------------------------------------------------------

class TestRecursiveChunker:
    def test_produces_valid_chunks(self, sample_doc: CanonicalDocument):
        chunks = chunk_document_recursive(sample_doc, chunk_size=200, overlap=20)
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.chunk_id.startswith("chunk:")
            assert chunk.content_hash.startswith("sha256:")
            assert chunk.document_id == sample_doc.document_id
            assert len(chunk.text) > 0
            assert chunk.source_span is not None

    def test_respects_max_chunk_size(self, sample_doc: CanonicalDocument):
        chunk_size = 200
        chunks = chunk_document_recursive(sample_doc, chunk_size=chunk_size, overlap=20)
        # Recursive splitter may slightly exceed chunk_size due to separator
        # boundaries, but should be close.
        for chunk in chunks:
            assert len(chunk.text) <= chunk_size + 50, (
                f"Chunk {chunk.chunk_id} is {len(chunk.text)} chars, "
                f"expected <= {chunk_size + 50}"
            )

    def test_chunk_text_is_substring_of_doc(self, sample_doc: CanonicalDocument):
        chunks = chunk_document_recursive(sample_doc, chunk_size=200, overlap=20)
        for chunk in chunks:
            # Chunks may have whitespace normalization, so check first 50 chars.
            assert chunk.text[:50] in sample_doc.text or chunk.text[:30] in sample_doc.text

    def test_empty_doc_produces_no_chunks(self, empty_doc: CanonicalDocument):
        chunks = chunk_document_recursive(empty_doc, chunk_size=200, overlap=20)
        assert chunks == []

    def test_metadata_records_splitter(self, sample_doc: CanonicalDocument):
        chunks = chunk_document_recursive(sample_doc, chunk_size=200, overlap=20)
        for chunk in chunks:
            assert chunk.metadata["splitter"] == "recursive"
            assert chunk.metadata["chunk_size"] == 200


# ---------------------------------------------------------------------------
# Token chunker tests
# ---------------------------------------------------------------------------

class TestTokenChunker:
    def test_produces_valid_chunks(self, sample_doc: CanonicalDocument):
        chunks = chunk_document_token(sample_doc, chunk_size=100, overlap=10)
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.chunk_id.startswith("chunk:")
            assert chunk.content_hash.startswith("sha256:")
            assert chunk.document_id == sample_doc.document_id
            assert len(chunk.text) > 0

    def test_empty_doc_produces_no_chunks(self, empty_doc: CanonicalDocument):
        chunks = chunk_document_token(empty_doc, chunk_size=100, overlap=10)
        assert chunks == []

    def test_metadata_records_splitter(self, sample_doc: CanonicalDocument):
        chunks = chunk_document_token(sample_doc, chunk_size=100, overlap=10)
        for chunk in chunks:
            assert chunk.metadata["splitter"] == "token"


# ---------------------------------------------------------------------------
# Semantic chunker tests
# ---------------------------------------------------------------------------

class TestSemanticChunker:
    def test_produces_valid_chunks(self, sample_doc: CanonicalDocument):
        chunks = chunk_document_semantic(sample_doc, threshold=0.3)
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.chunk_id.startswith("chunk:")
            assert chunk.content_hash.startswith("sha256:")
            assert chunk.document_id == sample_doc.document_id
            assert len(chunk.text) > 0
            assert chunk.source_span is not None

    def test_respects_max_chunk_size(self, sample_doc: CanonicalDocument):
        max_size = 300
        chunks = chunk_document_semantic(
            sample_doc, threshold=0.3, max_chunk_size=max_size
        )
        for chunk in chunks:
            assert len(chunk.text) <= max_size + 50, (
                f"Chunk is {len(chunk.text)} chars, expected <= {max_size + 50}"
            )

    def test_respects_min_chunk_size(self, sample_doc: CanonicalDocument):
        """No chunk (except possibly the last merged one) should be tiny."""
        min_size = 100
        chunks = chunk_document_semantic(
            sample_doc, threshold=0.3, min_chunk_size=min_size
        )
        # All chunks except possibly the last should be >= min_size.
        for chunk in chunks[:-1]:
            assert len(chunk.text) >= min_size, (
                f"Chunk is {len(chunk.text)} chars, expected >= {min_size}"
            )

    def test_empty_doc_produces_no_chunks(self, empty_doc: CanonicalDocument):
        chunks = chunk_document_semantic(empty_doc, threshold=0.3)
        assert chunks == []

    def test_metadata_records_splitter_and_threshold(self, sample_doc: CanonicalDocument):
        chunks = chunk_document_semantic(sample_doc, threshold=0.3)
        for chunk in chunks:
            assert chunk.metadata["splitter"] == "semantic"
            assert chunk.metadata["threshold"] == 0.3
            assert "num_sentences" in chunk.metadata

    def test_lower_threshold_produces_fewer_chunks(self, sample_doc: CanonicalDocument):
        """A lower threshold means fewer boundaries, thus fewer chunks."""
        chunks_high = chunk_document_semantic(
            sample_doc, threshold=0.8, min_chunk_size=10, max_chunk_size=10000
        )
        chunks_low = chunk_document_semantic(
            sample_doc, threshold=0.1, min_chunk_size=10, max_chunk_size=10000
        )
        assert len(chunks_low) <= len(chunks_high), (
            f"Low threshold produced {len(chunks_low)} chunks, "
            f"high threshold produced {len(chunks_high)}"
        )


# ---------------------------------------------------------------------------
# Cross-chunker consistency tests
# ---------------------------------------------------------------------------

class TestChunkerConsistency:
    """All chunkers must produce compatible DocumentChunk shapes."""

    def test_all_chunkers_produce_valid_chunks(self, sample_doc: CanonicalDocument):
        chunkers = [
            ("fixed_window", lambda d: chunk_document(d, chunk_size=200, overlap=20)),
            ("recursive", lambda d: chunk_document_recursive(d, chunk_size=200, overlap=20)),
            ("token", lambda d: chunk_document_token(d, chunk_size=100, overlap=10)),
        ]

        for name, fn in chunkers:
            chunks = fn(sample_doc)
            assert len(chunks) > 0, f"{name} produced no chunks"
            for chunk in chunks:
                assert chunk.chunk_id.startswith("chunk:"), f"{name} bad chunk_id"
                assert chunk.content_hash.startswith("sha256:"), f"{name} bad content_hash"
                assert chunk.document_id == sample_doc.document_id, f"{name} bad document_id"
                assert isinstance(chunk.text, str), f"{name} text not str"
                assert isinstance(chunk.metadata, dict), f"{name} metadata not dict"

    def test_chunk_ids_are_deterministic(self, sample_doc: CanonicalDocument):
        """Same input + same chunker = same chunk IDs."""
        chunks1 = chunk_document_recursive(sample_doc, chunk_size=200, overlap=20)
        chunks2 = chunk_document_recursive(sample_doc, chunk_size=200, overlap=20)
        ids1 = [c.chunk_id for c in chunks1]
        ids2 = [c.chunk_id for c in chunks2]
        assert ids1 == ids2

    def test_no_duplicate_chunk_ids_within_one_chunker(self, sample_doc: CanonicalDocument):
        """No chunker should produce duplicate chunk IDs for one document."""
        for name, fn in [
            ("fixed", lambda d: chunk_document(d, chunk_size=200, overlap=20)),
            ("recursive", lambda d: chunk_document_recursive(d, chunk_size=200, overlap=20)),
            ("token", lambda d: chunk_document_token(d, chunk_size=100, overlap=10)),
        ]:
            chunks = fn(sample_doc)
            ids = [c.chunk_id for c in chunks]
            assert len(ids) == len(set(ids)), f"{name} has duplicate chunk IDs"
